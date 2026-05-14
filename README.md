# nim-cpu-bridge

> Run unmodified NVIDIA NIM containers on CPU-only or non-NVIDIA hardware — same API, same UX, no driver required.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-compose-ready-blue)]()
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)]()

---

## What this is

NVIDIA NIM containers expose an OpenAI-compatible inference API and are optimised for NVIDIA GPUs. This project wraps any NIM image with a minimal shim layer so it boots and serves on:

- CPU-only Linux servers (x86-64, AVX2/AVX-512)
- Apple Silicon Macs (Metal via ctransformers/llama.cpp)
- Non-NVIDIA cloud instances (AWS Graviton, Azure Epyc, GCP C3)
- CI/CD pipelines with no GPUs at all

**Zero changes to the NIM image.** The shim is injected at container startup via `LD_PRELOAD` and a `docker compose` wrapper. NIM thinks it found a CUDA device; the shim silently routes inference calls to a CPU backend.

Performance will be slower than GPU — this is expected and acknowledged. The goal is API fidelity and NIM workflow compatibility, not throughput.

---

## Architecture

```
Client app (OpenAI SDK / curl)
        │  OpenAI-compatible REST
        ▼
 ┌────────────────────────────────────────┐
 │  NIM Container (unmodified image)      │
 │  ┌──────────────────────────────────┐  │
 │  │  NIM entrypoint + vLLM/Triton   │  │
 │  │       dlopen("libcuda.so")       │  │
 │  └────────────┬─────────────────────┘  │
 │               │  LD_PRELOAD intercept  │
 │  ┌────────────▼─────────────────────┐  │
 │  │   nim-cpu-bridge CUDA shim       │  │  ← key layer
 │  │   implements CUDA Driver API +   │  │
 │  │   Runtime API stubs in pure C    │  │
 │  └─────┬──────────┬──────────┬──────┘  │
 │        │          │          │          │
 │  ┌─────▼──┐  ┌───▼────┐  ┌──▼──────┐  │
 │  │llama.cpp│  │ ONNX   │  │ctransf. │  │
 │  │GGUF+AVX │  │Runtime │  │OpenBLAS │  │
 │  └─────────┘  └────────┘  └─────────┘  │
 └────────────────────────────────────────┘
          │  model weights (volume)
 ┌────────▼───────────────────────────────┐
 │  Model cache  (NGC → GGUF auto-convert)│
 └────────────────────────────────────────┘
          │
 ┌────────▼───────────────────────────────┐
 │  nim-cpu-bridge orchestrator sidecar   │
 │  • injects shim env vars               │
 │  • relays NGC API token                │
 │  • Prometheus metrics passthrough      │
 │  • health probe rewrite                │
 └────────────────────────────────────────┘
```

---

## Quick start

### Prerequisites

- Docker 24+ with compose v2
- An NGC API key (for pulling NIM images and weights)
- 16 GB RAM minimum; 32 GB recommended for 8B-class models

### Run

```bash
git clone https://github.com/your-org/nim-cpu-bridge
cd nim-cpu-bridge

export NGC_API_KEY=<your key>
export NIM_IMAGE=nvcr.io/nim/meta/llama-3.1-8b-instruct:latest

docker compose -f docker/compose.yml up
```

The service starts on `http://localhost:8000` with the standard NIM/OpenAI API surface.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta/llama-3.1-8b-instruct",
    "messages": [{"role":"user","content":"Hello!"}]
  }'
```

---

## Configuration

All runtime options are environment variables, mirroring NIM's own env surface:

| Variable | Default | Description |
|---|---|---|
| `NCB_BACKEND` | `auto` | `llama_cpp` / `onnxruntime` / `ctransformers` / `auto` |
| `NCB_THREADS` | `$(nproc)` | CPU threads for inference |
| `NCB_CONTEXT_LEN` | `4096` | KV cache context window |
| `NCB_QUANT` | `Q4_K_M` | GGUF quantisation level |
| `NCB_CONVERT_ON_START` | `true` | Auto-convert NGC weights to GGUF |
| `NCB_FAKE_GPU_NAME` | `Tesla T4` | GPU name reported to NIM |
| `NCB_FAKE_VRAM_MB` | `16384` | Fake VRAM size in MB |
| `NCB_LOG_LEVEL` | `INFO` | Shim + orchestrator log level |
| `NGC_API_KEY` | _(required)_ | NGC credential for weight download |

---

## How it works in depth

See [docs/deep-dive.md](docs/deep-dive.md) for full technical detail. Summary:

1. **CUDA shim** (`src/shim/cuda_shim.c`) — a shared library compiled to `libcuda.so.1` and `libcudart.so.12` that implements the ~80 CUDA Driver API functions NIM's stack actually calls at startup (device enumeration, memory allocation stubs, context creation). All compute calls (`cuLaunchKernel`, `cudaMemcpy` for device buffers) are intercepted and redirected via a Unix socket to the backend process.

2. **Backend router** (`src/backends/router.py`) — selects the best available CPU backend at startup. Probes for AVX-512, AVX2, NEON. Loads quantised weights via the chosen engine.

3. **Tensor bridge** (`src/backends/tensor_bridge.py`) — receives tensor payloads from the shim over a local socket, runs forward passes on CPU, and returns results. The shim maps these to fake CUDA device memory pointers so the NIM runtime never sees CPU memory addresses.

4. **Weight converter** (`src/converter/ngc_to_gguf.py`) — on first startup, downloads SafeTensors from NGC and converts to GGUF using `llama.cpp`'s `convert.py`. Subsequent starts use the cache.

5. **Orchestrator sidecar** (`src/orchestrator/main.py`) — a FastAPI process that sits in front of the NIM HTTP port, rewrites health checks, forwards NGC credentials, and exposes Prometheus metrics.

---

## Supported NIM images (tested)

| NIM image | Backend used | Notes |
|---|---|---|
| `nim/meta/llama-3.1-8b-instruct` | llama.cpp | Full support |
| `nim/meta/llama-3.1-70b-instruct` | llama.cpp | Requires 64 GB RAM |
| `nim/mistralai/mistral-7b-instruct-v0.3` | llama.cpp | Full support |
| `nim/microsoft/phi-3-mini-4k-instruct` | ONNX Runtime | Better perf via ONNX |
| `nim/nvidia/nemotron-mini-4b-instruct` | llama.cpp | Full support |

Embedding and reranker NIMs are not yet supported (planned).

---

## Project layout

```
nim-cpu-bridge/
├── src/
│   ├── shim/           # C CUDA stub library (the core intercept)
│   ├── backends/       # Python backend router + tensor bridge
│   ├── converter/      # NGC SafeTensors → GGUF converter
│   └── orchestrator/   # FastAPI sidecar + health/metrics
├── docker/
│   ├── compose.yml     # Main entry point for users
│   ├── Dockerfile.shim # Builds the shim .so
│   └── Dockerfile.sidecar
├── configs/            # Per-model backend profiles
├── scripts/            # Helper scripts (pull-weights, benchmark)
├── tests/              # Integration tests (no GPU required)
└── docs/               # Deep-dive docs
```

---

## License

Apache 2.0. See [LICENSE](LICENSE).

NVIDIA NIM is a commercial product subject to NVIDIA's terms. This project does not redistribute NIM software; it only provides tooling to wrap externally-pulled NIM containers.
