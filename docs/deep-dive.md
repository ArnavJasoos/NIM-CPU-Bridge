# nim-cpu-bridge: Technical Deep Dive

## Why this is hard and how we solve it

NIM containers are tightly coupled to CUDA at three levels:

| Level | Where | What NIM does |
|---|---|---|
| Driver API | `libcuda.so.1` | `cuInit`, `cuDeviceGetCount`, context/stream create |
| Runtime API | `libcudart.so.12` | `cudaMalloc`, `cudaMemcpy`, `cudaLaunchKernel` |
| NVML | `libnvidia-ml.so.1` | Queries GPU name, VRAM, compute capability |
| cuBLAS | `libcublas.so.12` | GEMM operations for attention / FFN |
| NCCL | `libnccl.so.2` | Multi-GPU comms (probed at init even for single-GPU) |

At startup, vLLM / Triton / TRT-LLM call into all five. If any returns an error,
the NIM entrypoint aborts.

### The shim approach

We compile a single shared library (`libcuda_shim.so`) that exports every symbol
from all five libraries. It's loaded via `LD_PRELOAD`, which makes the dynamic
linker prefer it over any real CUDA libraries the container might have. Key properties:

- **Device enumeration** returns 1 fake GPU with properties matching a Tesla T4
  (CUDA compute 7.5, 16 GB VRAM by default — configurable via env vars).
- **Memory allocation** backs "device" pointers with real heap memory. NIM passes
  `CUdeviceptr` values around opaquely; since NIM never dereferences them directly
  (only passes them to kernel launches), host pointers work transparently.
- **Kernel launches** (`cuLaunchKernel`, `cudaLaunchKernel`) dispatch to the
  Python tensor bridge over a Unix domain socket instead of the GPU.

### The tensor bridge

The shim and the Python backend communicate over a Unix socket at
`/tmp/ncb_bridge.sock` (configurable). The protocol is a simple TLV:

```
[4 bytes: msg_type (big-endian)] [4 bytes: payload_len] [payload: JSON or raw bytes]
```

For LLM workloads, we take a higher-level approach: rather than intercepting
individual CUDA kernels (which would require reimplementing the entire cuBLAS
GEMM protocol), we patch vLLM's `LLMEngine.generate()` at the Python level via
a monkey-patch injected by the orchestrator before NIM's Python code imports vLLM.
This gives us clean control over the inference loop without having to understand
every kernel the model runner calls.

The low-level kernel dispatch in `cuLaunchKernel` remains as a safety net for
NIM images that call into CUDA at a lower level than vLLM.

### Model weight handling

NIM downloads weights in SafeTensors format from NGC. Our converter:

1. Detects the NIM cache directory (usually `~/.cache/nim/<model_slug>/`)
2. Locates all `.safetensors` shards
3. Invokes `llama.cpp`'s `convert_hf_to_gguf.py` to produce an F16 GGUF
4. Quantises with `llama-quantize` to the requested level (default Q4_K_M)

The quantised GGUF is cached in `<nim_cache>/ncb_converted/` and reused on
subsequent starts.

### Why not just run llama.cpp directly?

The NIM API surface is more than `/v1/chat/completions`. NIM images expose:
- Custom NVIDIA API extensions (`/v1/models`, `/health`, streaming protocols)
- NGC credential management and model profile selection
- Multi-LoRA serving, guided decoding, and speculative decoding APIs

Clients targeting NIM may call these endpoints. By wrapping NIM's own container
rather than replacing it with a plain llama.cpp server, we maintain full API
compatibility for applications already integrated with NIM — they need only
change the base URL, not the client code.

---

## CUDA symbol coverage

The shim implements ~180 CUDA symbols. The complete list is in
`src/shim/cuda_shim.c`. Key categories:

| Category | Symbols | Notes |
|---|---|---|
| Initialisation | `cuInit`, `cuDriverGetVersion` | Returns success immediately |
| Device | `cuDeviceGet*`, `cudaGetDevice*` | Returns 1 fake T4 |
| Context | `cuCtxCreate/Destroy/Get/Set` | Dummy handle |
| Memory | `cuMemAlloc/Free`, `cudaMalloc/Free`, managed | Host-backed |
| Memcpy | `cuMemcpy*`, `cudaMemcpy*` | Host memcpy |
| Streams | `cuStream*`, `cudaStream*` | Dummy handles |
| Kernels | `cuLaunchKernel`, `cudaLaunchKernel` | Bridge dispatch |
| Modules | `cuModule*`, `cuFunc*` | Dummy handles |
| cuBLAS | `cublasCreate/Destroy`, `cublasGemm*` | Bridge dispatch |
| NVML | `nvmlInit/DeviceGet*` | Returns fake device props |
| NCCL | `ncclCommInit/Destroy`, `ncclGetVersion` | No-op stubs |
| Error | `cudaGetErrorString`, `cuGetErrorName` | Returns "no error" |

---

## Performance expectations

On a 2024 server-class CPU (96-core Xeon, AVX-512):

| Model | Quant | Tokens/sec (approx) |
|---|---|---|
| Llama 3.1 8B | Q4_K_M | 8–15 |
| Llama 3.1 8B | Q8_0   | 4–8  |
| Phi-3 Mini   | ONNX F32 | 20–40 |

For comparison, a single H100 achieves 1000+ tokens/sec for the 8B model.
CPU inference is appropriate for development, testing, low-traffic internal
services, and air-gapped deployments where GPU access is not available.

To improve throughput:
- Prefer Q4_K_M or Q4_K_S — they hit the best quality-per-speed tradeoff
- Set `NCB_THREADS` to the number of physical cores (not hyperthreads)
- On AVX-512 hosts, rebuild llama-cpp-python with `CMAKE_ARGS="-DLLAMA_AVX512=on"`
- Enable huge pages: `echo always > /sys/kernel/mm/transparent_hugepage/enabled`

---

## Security considerations

- The shim does not disable NIM's safeguards — it only intercepts hardware probing.
- NGC API keys are passed through to the NIM container unchanged.
- The bridge socket (`/tmp/ncb_bridge.sock`) is host-local. In multi-tenant
  environments, bind it to a per-container tempdir with restricted permissions.
- Running NIM images implies accepting NVIDIA's NIM terms of service and the
  model license. nim-cpu-bridge does not alter those obligations.

---

## Limitations and known issues

| Limitation | Detail |
|---|---|
| Embedding / reranker NIMs | Not yet supported — planned for v0.2 |
| TRT-LLM engines | Pre-compiled TRT engines are GPU-specific; NIM falls back to vLLM on unsupported GPUs, which our shim then intercepts |
| Multi-node inference | NCCL stubs are no-ops; single-node only |
| Streaming speed | Token generation is synchronous in the first version; async batching planned |
| Windows | Shim uses POSIX APIs; Linux only |

---

## Contributing

Pull requests welcome. Please:
1. Add a test for any new CUDA symbol stubs
2. Run `pytest tests/ -v` against a tiny model before submitting
3. Keep the shim in C (no C++ to minimise binary size and ABI risk)

See [CONTRIBUTING.md](../CONTRIBUTING.md) for details.
