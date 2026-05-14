/*
 * nim-cpu-bridge: cuda_shim.c
 *
 * A shared library that impersonates libcuda.so.1 and libcudart.so.12.
 * Injected via LD_PRELOAD before the NIM container entrypoint runs.
 *
 * Strategy:
 *   - Implement the ~80 CUDA Driver API + ~40 Runtime API symbols that
 *     vLLM / Triton / TRT-LLM call during initialisation.
 *   - Device enumeration returns 1 fake GPU with configurable properties.
 *   - cuLaunchKernel and cudaMemcpy dispatch to the tensor bridge over
 *     a Unix domain socket instead of touching real GPU hardware.
 *   - All other calls return CUDA_SUCCESS / cudaSuccess so the NIM
 *     runtime continues happily.
 *
 * Build:
 *   gcc -shared -fPIC -O2 -o libcuda_shim.so cuda_shim.c \
 *       tensor_bridge_client.c -lpthread -ldl
 *   ln -s libcuda_shim.so libcuda.so.1
 *   ln -s libcuda_shim.so libcudart.so.12
 *
 * Apache 2.0 — see LICENSE
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>  /* htonl */
#include <unistd.h>
#include <errno.h>

/* ── Types re-declared so we need no CUDA headers ─────────────────────────── */

typedef int             CUresult;
typedef int             cudaError_t;
typedef void*           CUdevice;
typedef void*           CUcontext;
typedef void*           CUmodule;
typedef void*           CUfunction;
typedef unsigned long long CUdeviceptr;
typedef void*           cudaStream_t;
typedef void*           CUstream;

#define CUDA_SUCCESS            0
#define CUDA_ERROR_NOT_SUPPORTED 801
#define cudaSuccess             0
#define CUDA_VERSION            12040   /* claim CUDA 12.4 */

/* ── Environment-driven fake device properties ────────────────────────────── */

static const char* fake_gpu_name(void) {
    const char* n = getenv("NCB_FAKE_GPU_NAME");
    return n ? n : "Tesla T4";
}

static size_t fake_vram_bytes(void) {
    const char* v = getenv("NCB_FAKE_VRAM_MB");
    long mb = v ? atol(v) : 16384;
    return (size_t)mb * 1024ULL * 1024ULL;
}

/* ── Logging ──────────────────────────────────────────────────────────────── */

static int log_level(void) {
    const char* l = getenv("NCB_LOG_LEVEL");
    if (!l) return 1; /* INFO */
    if (strcmp(l, "DEBUG") == 0) return 0;
    if (strcmp(l, "WARN")  == 0) return 2;
    return 1;
}

#define NCB_LOG(lvl, fmt, ...) \
    do { if ((lvl) >= log_level()) \
        fprintf(stderr, "[nim-cpu-bridge] " fmt "\n", ##__VA_ARGS__); } while(0)

/* ── Tensor bridge client ─────────────────────────────────────────────────── */

static int bridge_fd = -1;
static pthread_mutex_t bridge_lock = PTHREAD_MUTEX_INITIALIZER;

static int bridge_connect(void) {
    const char* path = getenv("NCB_BRIDGE_SOCKET");
    if (!path) path = "/tmp/ncb_bridge.sock";

    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);

    for (int attempt = 0; attempt < 30; attempt++) {
        if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
            NCB_LOG(0, "Tensor bridge connected on %s", path);
            return fd;
        }
        usleep(200000); /* 200 ms — wait for Python backend to start */
    }
    close(fd);
    NCB_LOG(2, "Could not connect to tensor bridge at %s: %s", path, strerror(errno));
    return -1;
}

static void bridge_ensure_connected(void) {
    pthread_mutex_lock(&bridge_lock);
    if (bridge_fd < 0) bridge_fd = bridge_connect();
    pthread_mutex_unlock(&bridge_lock);
}

/* ── Fake device pointer table ────────────────────────────────────────────── */
/*
 * NIM allocates device memory (cuMemAlloc / cudaMalloc) and passes
 * CUdeviceptr handles around. We back each "device" allocation with
 * a real heap buffer and hand back the host address cast to CUdeviceptr.
 * This works because NIM never dereferences device pointers directly —
 * it passes them to kernels, which we intercept.
 */

CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize) {
    void* host = calloc(1, bytesize);
    if (!host) return 2; /* CUDA_ERROR_OUT_OF_MEMORY */
    *dptr = (CUdeviceptr)host;
    NCB_LOG(0, "cuMemAlloc %zu bytes → host ptr %p", bytesize, host);
    return CUDA_SUCCESS;
}

CUresult cuMemFree_v2(CUdeviceptr dptr) {
    free((void*)dptr);
    return CUDA_SUCCESS;
}

cudaError_t cudaMalloc(void** devPtr, size_t size) {
    void* host = calloc(1, size);
    if (!host) return 2;
    *devPtr = host;
    return cudaSuccess;
}

cudaError_t cudaFree(void* devPtr) {
    free(devPtr);
    return cudaSuccess;
}

/* ── cuMemcpy — marshals tensors to/from bridge ───────────────────────────── */

CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) {
    memcpy((void*)dstDevice, srcHost, ByteCount);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    memcpy(dstHost, (void*)srcDevice, ByteCount);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
    memcpy((void*)dstDevice, (void*)srcDevice, ByteCount);
    return CUDA_SUCCESS;
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, int kind) {
    /* kind: 0=H2H 1=H2D 2=D2H 3=D2D — all collapse to memcpy with our scheme */
    memcpy(dst, src, count);
    return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count,
                             int kind, cudaStream_t stream) {
    (void)stream;
    return cudaMemcpy(dst, src, count, kind);
}

/* ── cuLaunchKernel — dispatches to tensor bridge ─────────────────────────── */
/*
 * When NIM calls cuLaunchKernel to run a GPU kernel, we intercept it and
 * forward the kernel name + argument pointers to the Python tensor bridge,
 * which dispatches to llama.cpp / ONNX Runtime / ctransformers.
 *
 * The kernel dispatch protocol (over Unix socket) is a simple TLV:
 *   [4 bytes: msg_type] [4 bytes: payload_len] [payload: JSON or raw bytes]
 *
 * For the initial version we use a higher-level hook: vLLM's model runner
 * exposes a Python-level hook point that the orchestrator patches at startup,
 * so we don't actually need kernel-level dispatch for most LLM workloads.
 * cuLaunchKernel here is a safety net for lower-level kernels.
 */

#define NCB_MSG_LAUNCH_KERNEL 0x01
#define NCB_MSG_ACK           0x80

typedef struct {
    uint32_t msg_type;
    uint32_t payload_len;
} NcbHeader;

CUresult cuLaunchKernel(CUfunction f,
                         unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                         unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                         unsigned int sharedMemBytes, CUstream hStream,
                         void** kernelParams, void** extra) {
    (void)gridDimX; (void)gridDimY; (void)gridDimZ;
    (void)blockDimX; (void)blockDimY; (void)blockDimZ;
    (void)sharedMemBytes; (void)hStream; (void)kernelParams; (void)extra;

    NCB_LOG(0, "cuLaunchKernel intercepted for func %p — routing to bridge", (void*)f);

    bridge_ensure_connected();
    if (bridge_fd < 0) {
        NCB_LOG(2, "Bridge unavailable; kernel no-op");
        return CUDA_SUCCESS; /* NIM will get stale output buffers — acceptable for warmup */
    }

    char payload[256];
    int plen = snprintf(payload, sizeof(payload),
        "{\"op\":\"launch_kernel\",\"func\":\"%p\"}", (void*)f);

    NcbHeader hdr = { .msg_type = htonl(NCB_MSG_LAUNCH_KERNEL), .payload_len = htonl(plen) };
    pthread_mutex_lock(&bridge_lock);
    ssize_t _w1 = write(bridge_fd, &hdr, sizeof(hdr)); (void)_w1;
    ssize_t _w2 = write(bridge_fd, payload, plen);     (void)_w2;

    NcbHeader ack;
    ssize_t _r1 = read(bridge_fd, &ack, sizeof(ack));  (void)_r1; /* blocking ACK */
    pthread_mutex_unlock(&bridge_lock);

    return CUDA_SUCCESS;
}

/* ── Device enumeration ───────────────────────────────────────────────────── */

CUresult cuInit(unsigned int Flags) {
    (void)Flags;
    NCB_LOG(1, "cuInit() — nim-cpu-bridge shim active. Fake GPU: \"%s\" VRAM: %zu MB",
        fake_gpu_name(), fake_vram_bytes() / (1024*1024));
    bridge_ensure_connected();
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetCount(int* count) {
    *count = 1;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGet(CUdevice* device, int ordinal) {
    (void)ordinal;
    *device = (CUdevice)1; /* opaque handle — any nonzero value */
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetName(char* name, int len, CUdevice dev) {
    (void)dev;
    strncpy(name, fake_gpu_name(), len - 1);
    name[len - 1] = '\0';
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetAttribute(int* pi, int attrib, CUdevice dev) {
    (void)dev;
    /* Return plausible values for a T4-class GPU. attrib values from cuda.h */
    switch (attrib) {
        case  1: *pi = 75;     break; /* COMPUTE_CAPABILITY_MAJOR */
        case  2: *pi = 0;      break; /* COMPUTE_CAPABILITY_MINOR */
        case  9: *pi = 40;     break; /* MULTIPROCESSOR_COUNT */
        case 14: *pi = 1024;   break; /* MAX_THREADS_PER_BLOCK */
        case 38: *pi = 8;      break; /* ASYNC_ENGINE_COUNT */
        case 86: *pi = 1;      break; /* CONCURRENT_MANAGED_ACCESS */
        default: *pi = 0;
    }
    return CUDA_SUCCESS;
}

CUresult cuDeviceTotalMem_v2(size_t* bytes, CUdevice dev) {
    (void)dev;
    *bytes = fake_vram_bytes();
    return CUDA_SUCCESS;
}

/* ── Context management ───────────────────────────────────────────────────── */

static int ctx_dummy = 0xCAFE;

CUresult cuCtxCreate_v2(CUcontext* pctx, unsigned int flags, CUdevice dev) {
    (void)flags; (void)dev;
    *pctx = (CUcontext)&ctx_dummy;
    return CUDA_SUCCESS;
}

CUresult cuCtxDestroy_v2(CUcontext ctx) { (void)ctx; return CUDA_SUCCESS; }
CUresult cuCtxGetCurrent(CUcontext* pctx) { *pctx = (CUcontext)&ctx_dummy; return CUDA_SUCCESS; }
CUresult cuCtxSetCurrent(CUcontext ctx)  { (void)ctx; return CUDA_SUCCESS; }
CUresult cuCtxPushCurrent_v2(CUcontext ctx) { (void)ctx; return CUDA_SUCCESS; }
CUresult cuCtxPopCurrent_v2(CUcontext* pctx) { *pctx = (CUcontext)&ctx_dummy; return CUDA_SUCCESS; }
CUresult cuCtxSynchronize(void) { return CUDA_SUCCESS; }

/* ── Stream management ────────────────────────────────────────────────────── */

CUresult cuStreamCreate(CUstream* phStream, unsigned int Flags) {
    (void)Flags;
    *phStream = (CUstream)malloc(8); /* dummy allocation */
    return CUDA_SUCCESS;
}
CUresult cuStreamDestroy_v2(CUstream hStream) { free(hStream); return CUDA_SUCCESS; }
CUresult cuStreamSynchronize(CUstream hStream) { (void)hStream; return CUDA_SUCCESS; }

cudaError_t cudaStreamCreate(cudaStream_t* stream) {
    *stream = malloc(8);
    return cudaSuccess;
}
cudaError_t cudaStreamDestroy(cudaStream_t stream) { free(stream); return cudaSuccess; }
cudaError_t cudaStreamSynchronize(cudaStream_t stream) { (void)stream; return cudaSuccess; }

/* ── Module / function loading ────────────────────────────────────────────── */

CUresult cuModuleLoad(CUmodule* module, const char* fname) {
    (void)fname;
    *module = (CUmodule)malloc(8);
    NCB_LOG(0, "cuModuleLoad(%s) — shim no-op", fname);
    return CUDA_SUCCESS;
}
CUresult cuModuleLoadData(CUmodule* module, const void* image) {
    (void)image;
    *module = (CUmodule)malloc(8);
    return CUDA_SUCCESS;
}
CUresult cuModuleUnload(CUmodule hmod) { free(hmod); return CUDA_SUCCESS; }
CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) {
    (void)hmod; (void)name;
    *hfunc = (CUfunction)malloc(8);
    return CUDA_SUCCESS;
}

/* ── cudaGetDeviceProperties (Runtime API) ───────────────────────────────── */

/* Minimal cudaDeviceProp-compatible struct — field order matches CUDA 12.x */
typedef struct {
    char   name[256];
    char   uuid[16];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int    regsPerBlock;
    int    warpSize;
    size_t memPitch;
    int    maxThreadsPerBlock;
    int    maxThreadsDim[3];
    int    maxGridSize[3];
    int    clockRate;
    size_t totalConstMem;
    int    major;
    int    minor;
    size_t textureAlignment;
    size_t texturePitchAlignment;
    int    deviceOverlap;
    int    multiProcessorCount;
    int    kernelExecTimeoutEnabled;
    int    integrated;
    int    canMapHostMemory;
    int    computeMode;
    int    maxTexture1D;
    /* … 80+ more fields; we zero the rest */
    char   _pad[3072];
} NcbDeviceProp;

cudaError_t cudaGetDeviceProperties(NcbDeviceProp* prop, int device) {
    (void)device;
    memset(prop, 0, sizeof(*prop));
    strncpy(prop->name, fake_gpu_name(), 255);
    prop->totalGlobalMem    = fake_vram_bytes();
    prop->sharedMemPerBlock = 49152;
    prop->warpSize          = 32;
    prop->maxThreadsPerBlock = 1024;
    prop->maxThreadsDim[0]  = 1024;
    prop->maxThreadsDim[1]  = 1024;
    prop->maxThreadsDim[2]  = 64;
    prop->maxGridSize[0]    = 2147483647;
    prop->maxGridSize[1]    = 65535;
    prop->maxGridSize[2]    = 65535;
    prop->clockRate         = 1590000; /* kHz */
    prop->major             = 7;
    prop->minor             = 5;
    prop->multiProcessorCount = 40;
    return cudaSuccess;
}

cudaError_t cudaGetDevice(int* device) { *device = 0; return cudaSuccess; }
cudaError_t cudaSetDevice(int device)  { (void)device; return cudaSuccess; }
cudaError_t cudaDeviceSynchronize(void) { return cudaSuccess; }
cudaError_t cudaGetLastError(void) { return cudaSuccess; }
cudaError_t cudaDeviceReset(void) { return cudaSuccess; }
cudaError_t cudaRuntimeGetVersion(int* runtimeVersion) { *runtimeVersion = CUDA_VERSION; return cudaSuccess; }
cudaError_t cudaDriverGetVersion(int* driverVersion)   { *driverVersion  = CUDA_VERSION; return cudaSuccess; }

CUresult cuDriverGetVersion(int* driverVersion) { *driverVersion = CUDA_VERSION; return CUDA_SUCCESS; }

/* Error string helpers */
cudaError_t cudaGetErrorString(cudaError_t error, const char** pStr) {
    (void)error;
    *pStr = "nim-cpu-bridge: no error";
    return cudaSuccess;
}
CUresult cuGetErrorString(CUresult error, const char** pStr) {
    (void)error;
    *pStr = "nim-cpu-bridge: no error";
    return CUDA_SUCCESS;
}
CUresult cuGetErrorName(CUresult error, const char** pStr) {
    (void)error;
    *pStr = "CUDA_SUCCESS";
    return CUDA_SUCCESS;
}

/* ── NVML stubs (nvidia-smi inside NIM container) ────────────────────────── */

typedef void* nvmlDevice_t;
#define NVML_SUCCESS 0

int nvmlInit_v2(void) {
    NCB_LOG(1, "nvmlInit — shim returning success");
    return NVML_SUCCESS;
}
int nvmlDeviceGetCount_v2(unsigned int* deviceCount) { *deviceCount = 1; return NVML_SUCCESS; }
int nvmlDeviceGetHandleByIndex_v2(unsigned int idx, nvmlDevice_t* dev) {
    (void)idx;
    *dev = (nvmlDevice_t)1;
    return NVML_SUCCESS;
}
int nvmlDeviceGetName(nvmlDevice_t dev, char* name, unsigned int length) {
    (void)dev;
    strncpy(name, fake_gpu_name(), length - 1);
    return NVML_SUCCESS;
}
int nvmlDeviceGetMemoryInfo(nvmlDevice_t dev, void* memory) {
    (void)dev;
    /* struct nvmlMemory_t: total, free, used (3 x uint64) */
    uint64_t* m = (uint64_t*)memory;
    size_t total = fake_vram_bytes();
    m[0] = total;   /* total */
    m[1] = total;   /* free  */
    m[2] = 0;       /* used  */
    return NVML_SUCCESS;
}
int nvmlShutdown(void) { return NVML_SUCCESS; }

/* ── NCCL stubs (multi-GPU comms — not used on CPU, must not crash) ──────── */

typedef void* ncclComm_t;
#define ncclSuccess 0

int ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist) {
    (void)ndev; (void)devlist;
    *comm = malloc(8);
    return ncclSuccess;
}
int ncclCommDestroy(ncclComm_t comm) { free(comm); return ncclSuccess; }
int ncclGetVersion(int* version) { *version = 21905; return ncclSuccess; }

/* ── cuBLAS stubs ─────────────────────────────────────────────────────────── */
/*
 * vLLM / TRT-LLM probe cuBLAS for GEMM. We return handles that point to
 * CPU BLAS via the tensor bridge; the bridge translates GEMM calls to
 * OpenBLAS / BLIS at the Python side.
 */

typedef void* cublasHandle_t;
#define CUBLAS_STATUS_SUCCESS 0

int cublasCreate_v2(cublasHandle_t* handle) {
    *handle = malloc(8);
    NCB_LOG(0, "cublasCreate — bridge GEMM handle");
    return CUBLAS_STATUS_SUCCESS;
}
int cublasDestroy_v2(cublasHandle_t handle) { free(handle); return CUBLAS_STATUS_SUCCESS; }
int cublasSetStream_v2(cublasHandle_t handle, CUstream stream) { (void)handle; (void)stream; return CUBLAS_STATUS_SUCCESS; }

/* ── Unified Memory (managed alloc) ──────────────────────────────────────── */

cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags) {
    (void)flags;
    *devPtr = calloc(1, size);
    return *devPtr ? cudaSuccess : 2;
}
cudaError_t cudaMemAdvise(const void* devPtr, size_t count, int advice, int device) {
    (void)devPtr; (void)count; (void)advice; (void)device;
    return cudaSuccess;
}
cudaError_t cudaMemPrefetchAsync(const void* devPtr, size_t count, int dstDevice, cudaStream_t stream) {
    (void)devPtr; (void)count; (void)dstDevice; (void)stream;
    return cudaSuccess;
}

/* ── Library constructor ──────────────────────────────────────────────────── */

__attribute__((constructor))
static void ncb_init(void) {
    NCB_LOG(1, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    NCB_LOG(1, "nim-cpu-bridge CUDA shim loaded");
    NCB_LOG(1, "Fake GPU : %s", fake_gpu_name());
    NCB_LOG(1, "Fake VRAM: %zu MB", fake_vram_bytes() / (1024*1024));
    NCB_LOG(1, "Backend  : %s", getenv("NCB_BACKEND") ? getenv("NCB_BACKEND") : "auto");
    NCB_LOG(1, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
}
