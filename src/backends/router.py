"""
nim-cpu-bridge: backend/router.py

Selects the best available CPU inference backend at startup, loads
quantised model weights, and serves tensor requests from the CUDA shim
over the local Unix bridge socket.

Supported backends (in priority order):
  1. llama_cpp     — GGUF quantised weights, best CPU performance via AVX2/AVX-512
  2. onnxruntime   — ONNX graph, good for small/medium models (Phi-3, Mistral)
  3. ctransformers — GGUF fallback with OpenBLAS; also supports Apple Metal

Apache 2.0 — see LICENSE
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import struct
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger("ncb.router")


# ── Backend enum ──────────────────────────────────────────────────────────────

class BackendKind(Enum):
    LLAMA_CPP       = auto()
    ONNX_RUNTIME    = auto()
    CTRANSFORMERS   = auto()


# ── Hardware probe ────────────────────────────────────────────────────────────

def _cpu_flags() -> set[str]:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    return set(line.split(":")[1].split())
    except OSError:
        pass
    return set()


def _probe_hardware() -> dict[str, Any]:
    flags = _cpu_flags()
    info: dict[str, Any] = {
        "avx512f":  "avx512f"  in flags,
        "avx2":     "avx2"     in flags,
        "avx":      "avx"      in flags,
        "neon":     _is_arm(),
        "core_count": os.cpu_count() or 1,
        "ram_gb":   _ram_gb(),
    }
    logger.info("Hardware: %s", info)
    return info


def _is_arm() -> bool:
    import platform
    return platform.machine().lower() in ("aarch64", "arm64")


def _ram_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    return round(kb / (1024 * 1024), 1)
    except OSError:
        pass
    return 0.0


# ── Backend availability check ────────────────────────────────────────────────

def _backend_available(kind: BackendKind) -> bool:
    if kind == BackendKind.LLAMA_CPP:
        try:
            import llama_cpp  # noqa: F401
            return True
        except ImportError:
            return False
    elif kind == BackendKind.ONNX_RUNTIME:
        try:
            import onnxruntime  # noqa: F401
            return True
        except ImportError:
            return False
    elif kind == BackendKind.CTRANSFORMERS:
        try:
            import ctransformers  # noqa: F401
            return True
        except ImportError:
            return False
    return False


def _select_backend(env_override: str | None, hw: dict[str, Any]) -> BackendKind:
    if env_override and env_override != "auto":
        mapping = {
            "llama_cpp":      BackendKind.LLAMA_CPP,
            "onnxruntime":    BackendKind.ONNX_RUNTIME,
            "ctransformers":  BackendKind.CTRANSFORMERS,
        }
        chosen = mapping.get(env_override)
        if chosen:
            logger.info("Backend override: %s", env_override)
            return chosen
        logger.warning("Unknown backend override '%s', falling back to auto", env_override)

    # Auto-selection priority
    for candidate in [BackendKind.LLAMA_CPP, BackendKind.ONNX_RUNTIME, BackendKind.CTRANSFORMERS]:
        if _backend_available(candidate):
            logger.info("Auto-selected backend: %s", candidate.name)
            return candidate

    raise RuntimeError(
        "No CPU inference backend found. Install at least one of: "
        "llama-cpp-python, onnxruntime, ctransformers"
    )


# ── Model loading ─────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    model_path: Path
    context_len: int      = 4096
    n_threads: int        = field(default_factory=lambda: os.cpu_count() or 4)
    quant_type: str       = "Q4_K_M"
    max_batch_size: int   = 1


class BackendHandle:
    """Wraps whichever backend was chosen."""

    def __init__(self, kind: BackendKind, cfg: ModelConfig):
        self.kind = kind
        self.cfg  = cfg
        self._model: Any = None
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        logger.info("Loading model %s with backend %s …", self.cfg.model_path, self.kind.name)
        if self.kind == BackendKind.LLAMA_CPP:
            self._load_llama_cpp()
        elif self.kind == BackendKind.ONNX_RUNTIME:
            self._load_onnxruntime()
        elif self.kind == BackendKind.CTRANSFORMERS:
            self._load_ctransformers()

    def _load_llama_cpp(self) -> None:
        from llama_cpp import Llama  # type: ignore
        self._model = Llama(
            model_path=str(self.cfg.model_path),
            n_ctx=self.cfg.context_len,
            n_threads=self.cfg.n_threads,
            n_gpu_layers=0,   # CPU only
            verbose=False,
        )
        logger.info("llama.cpp model loaded; ctx=%d threads=%d",
                    self.cfg.context_len, self.cfg.n_threads)

    def _load_onnxruntime(self) -> None:
        import onnxruntime as ort  # type: ignore
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.cfg.n_threads
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self._model = ort.InferenceSession(
            str(self.cfg.model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        logger.info("ONNX Runtime session created")

    def _load_ctransformers(self) -> None:
        from ctransformers import AutoModelForCausalLM  # type: ignore
        self._model = AutoModelForCausalLM.from_pretrained(
            str(self.cfg.model_path),
            model_type="llama",
            threads=self.cfg.n_threads,
            gpu_layers=0,
        )
        logger.info("ctransformers model loaded")

    # ── Inference ──────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
        stop: list[str] | None = None,
        stream: bool = False,
    ) -> Iterator[str]:
        """Yield text chunks. For non-streaming backends, yields once."""
        with self._lock:
            if self.kind == BackendKind.LLAMA_CPP:
                yield from self._gen_llama_cpp(prompt, max_tokens, temperature, top_p, stop, stream)
            elif self.kind == BackendKind.ONNX_RUNTIME:
                yield from self._gen_onnxruntime(prompt, max_tokens, temperature, top_p, stop)
            elif self.kind == BackendKind.CTRANSFORMERS:
                yield from self._gen_ctransformers(prompt, max_tokens, temperature, top_p, stop, stream)

    def _gen_llama_cpp(self, prompt, max_tokens, temperature, top_p, stop, stream):
        output = self._model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
            stream=stream,
            echo=False,
        )
        if stream:
            for chunk in output:
                yield chunk["choices"][0]["text"]
        else:
            yield output["choices"][0]["text"]

    def _gen_onnxruntime(self, prompt, max_tokens, temperature, top_p, stop):
        # ONNX Runtime path requires tokenizer; this is handled by the
        # orchestrator which wraps the ONNX session with HuggingFace tokenizer.
        # Here we delegate back to the orchestrator via a local call.
        from .onnx_gen import generate_onnx  # lazy import
        yield from generate_onnx(self._model, prompt, max_tokens, temperature, top_p, stop)

    def _gen_ctransformers(self, prompt, max_tokens, temperature, top_p, stop, stream):
        tokens = self._model.tokenize(prompt)
        output_tokens = self._model.generate(
            tokens,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        text = self._model.detokenize(list(output_tokens))
        yield text


# ── Bridge socket server ──────────────────────────────────────────────────────
#
# The CUDA shim connects here to dispatch kernel launches and tensor ops.
# Protocol: 4-byte big-endian msg_type + 4-byte big-endian payload_len + payload JSON

MSG_LAUNCH_KERNEL = 0x01
MSG_ACK           = 0x80


class BridgeServer(threading.Thread):
    """Listens on the Unix socket for shim messages."""

    def __init__(self, handle: BackendHandle, socket_path: str):
        super().__init__(daemon=True, name="ncb-bridge")
        self.handle      = handle
        self.socket_path = socket_path

    def run(self) -> None:
        import os
        try:
            os.unlink(self.socket_path)
        except OSError:
            pass

        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self.socket_path)
        srv.listen(8)
        logger.info("Bridge server listening on %s", self.socket_path)

        while True:
            conn, _ = srv.accept()
            t = threading.Thread(target=self._handle_conn, args=(conn,), daemon=True)
            t.start()

    def _handle_conn(self, conn: socket.socket) -> None:
        try:
            while True:
                header = self._recv_exact(conn, 8)
                if not header:
                    break
                msg_type, payload_len = struct.unpack(">II", header)
                payload_bytes = self._recv_exact(conn, payload_len) if payload_len else b""
                self._dispatch(conn, msg_type, payload_bytes)
        except (OSError, ConnectionResetError):
            pass
        finally:
            conn.close()

    @staticmethod
    def _recv_exact(conn: socket.socket, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                return b""
            buf += chunk
        return buf

    def _dispatch(self, conn: socket.socket, msg_type: int, payload: bytes) -> None:
        if msg_type == MSG_LAUNCH_KERNEL:
            try:
                data = json.loads(payload)
                logger.debug("Kernel dispatch: %s", data)
            except json.JSONDecodeError:
                pass
            # Send ACK
            conn.sendall(struct.pack(">II", MSG_ACK, 0))
        else:
            logger.warning("Unknown bridge message type: 0x%02x", msg_type)
            conn.sendall(struct.pack(">II", MSG_ACK, 0))


# ── Entry point ───────────────────────────────────────────────────────────────

def build_router() -> "Router":
    hw      = _probe_hardware()
    env_be  = os.getenv("NCB_BACKEND", "auto")
    kind    = _select_backend(env_be, hw)

    model_path_str = os.getenv("NCB_MODEL_PATH", "")
    if not model_path_str:
        raise ValueError("NCB_MODEL_PATH must be set to the GGUF/ONNX model file")

    model_path = Path(model_path_str)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    cfg = ModelConfig(
        model_path=model_path,
        context_len=int(os.getenv("NCB_CONTEXT_LEN", "4096")),
        n_threads=int(os.getenv("NCB_THREADS", str(hw["core_count"]))),
        quant_type=os.getenv("NCB_QUANT", "Q4_K_M"),
    )

    handle = BackendHandle(kind, cfg)

    socket_path = os.getenv("NCB_BRIDGE_SOCKET", "/tmp/ncb_bridge.sock")
    bridge = BridgeServer(handle, socket_path)
    bridge.start()

    return Router(handle, bridge)


@dataclass
class Router:
    handle: BackendHandle
    bridge: BridgeServer

    def generate(self, **kwargs) -> Iterator[str]:
        return self.handle.generate(**kwargs)
