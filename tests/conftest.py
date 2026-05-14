"""
nim-cpu-bridge: tests/conftest.py

Shared pytest fixtures used by test_integration.py.
All tests run without a real GPU, real NIM image, or real model weights.

Apache 2.0 — see LICENSE
"""

from __future__ import annotations

import os
import struct
import threading
from pathlib import Path

import pytest


# ── Environment ───────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def ncb_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Set all required NCB_* env vars to safe test values.
    Applied automatically to every test in the suite.
    """
    fake_gguf = tmp_path / "model.Q4_K_M.gguf"
    fake_gguf.touch()  # zero-byte placeholder; backends are mocked

    monkeypatch.setenv("NCB_BACKEND",        "llama_cpp")
    monkeypatch.setenv("NCB_MODEL_PATH",     str(fake_gguf))
    monkeypatch.setenv("NCB_QUANT",          "Q4_K_M")
    monkeypatch.setenv("NCB_CONTEXT_LEN",    "512")
    monkeypatch.setenv("NCB_THREADS",        "1")
    monkeypatch.setenv("NCB_LOG_LEVEL",      "DEBUG")
    monkeypatch.setenv("NCB_FAKE_GPU_NAME",  "Tesla T4")
    monkeypatch.setenv("NCB_FAKE_VRAM_MB",   "16384")
    monkeypatch.setenv("NCB_BRIDGE_SOCKET",  str(tmp_path / "bridge.sock"))
    monkeypatch.setenv("NIM_MODEL_NAME",     "meta/llama-3.1-8b-instruct")
    monkeypatch.setenv("NIM_CACHE_PATH",     str(tmp_path / "nim_cache"))
    monkeypatch.setenv("NCB_CONVERT_ON_START", "false")


# ── Fake bridge socket server ─────────────────────────────────────────────────

class _FakeBridge(threading.Thread):
    """
    Minimal Unix socket server that ACKs every message from the shim.
    Mirrors the real BridgeServer protocol so shim tests don't block.
    """

    MSG_ACK = 0x80

    def __init__(self, socket_path: str) -> None:
        super().__init__(daemon=True, name="fake-bridge")
        self.socket_path = socket_path
        self._ready = threading.Event()

    def run(self) -> None:
        import socket as _sock

        try:
            os.unlink(self.socket_path)
        except OSError:
            pass

        srv = _sock.socket(_sock.AF_UNIX, _sock.SOCK_STREAM)
        srv.bind(self.socket_path)
        srv.listen(4)
        self._ready.set()

        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                break
            threading.Thread(
                target=self._handle, args=(conn,), daemon=True
            ).start()

    def _handle(self, conn) -> None:  # type: ignore[no-untyped-def]
        import socket as _sock
        try:
            while True:
                header = b""
                while len(header) < 8:
                    chunk = conn.recv(8 - len(header))
                    if not chunk:
                        return
                    header += chunk
                _msg_type, payload_len = struct.unpack(">II", header)
                if payload_len:
                    conn.recv(payload_len)
                # Send ACK
                conn.sendall(struct.pack(">II", self.MSG_ACK, 0))
        except (OSError, ConnectionResetError):
            pass
        finally:
            conn.close()

    def wait_ready(self, timeout: float = 2.0) -> None:
        if not self._ready.wait(timeout):
            raise RuntimeError("FakeBridge did not start in time")


@pytest.fixture()
def fake_bridge(tmp_path: Path) -> _FakeBridge:
    """Start a fake bridge server and return it."""
    sock_path = str(tmp_path / "bridge.sock")
    bridge = _FakeBridge(sock_path)
    bridge.start()
    bridge.wait_ready()
    return bridge


# ── NIM cache directory ───────────────────────────────────────────────────────

@pytest.fixture()
def nim_cache(tmp_path: Path) -> Path:
    """
    Create a minimal NIM-style cache directory with a fake config.json
    and a placeholder .safetensors file.
    """
    import json

    slug = "meta--llama-3.1-8b-instruct"
    model_dir = tmp_path / slug / "v1"
    model_dir.mkdir(parents=True)

    (model_dir / "config.json").write_text(
        json.dumps({
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
            "num_hidden_layers": 32,
        })
    )
    (model_dir / "model.safetensors").write_bytes(b"\x00" * 64)

    return tmp_path


# ── FastAPI test client ───────────────────────────────────────────────────────

@pytest.fixture()
def mock_router(monkeypatch: pytest.MonkeyPatch):
    """
    Replace build_router() with a mock so orchestrator tests
    never try to load a real model.
    """
    from unittest.mock import MagicMock, patch

    mock = MagicMock()
    mock.handle.kind.name = "LLAMA_CPP"
    mock.generate.return_value = iter(["Hello from mock backend!"])

    with patch("src.orchestrator.main.build_router", return_value=mock):
        with patch("src.orchestrator.main.prepare_model", return_value=Path("/fake/model.gguf")):
            yield mock


@pytest.fixture()
def app_client(mock_router):
    """HTTPX async test client wired to the FastAPI orchestrator app."""
    from httpx import AsyncClient, ASGITransport
    from src.orchestrator.main import app

    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
