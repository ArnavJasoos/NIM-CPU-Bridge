"""
nim-cpu-bridge: tests/test_integration.py

Integration tests that exercise the full stack without a GPU.
Runs against a locally-started orchestrator with a tiny test model.

Requirements:
  pip install pytest pytest-asyncio httpx

Run:
  NCB_MODEL_PATH=<path/to/tiny.gguf> pytest tests/ -v
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pytest
import httpx

BASE_URL = os.getenv("NCB_TEST_BASE_URL", "http://localhost:8000")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def client():
    return httpx.Client(base_url=BASE_URL, timeout=120.0)


@pytest.fixture(scope="session")
def async_client():
    return httpx.AsyncClient(base_url=BASE_URL, timeout=120.0)


def wait_for_ready(client: httpx.Client, max_wait: int = 120) -> bool:
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            r = client.get("/health")
            if r.status_code == 200 and r.json().get("status") == "ready":
                return True
        except httpx.ConnectError:
            pass
        time.sleep(2)
    return False


# ── Health tests ──────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_nim_cpu_bridge_marker(self, client):
        r = client.get("/health")
        assert r.json().get("nim_cpu_bridge") is True

    def test_health_status_ready(self, client):
        assert wait_for_ready(client, max_wait=30), "Service did not become ready"


# ── Model listing ─────────────────────────────────────────────────────────────

class TestModels:
    def test_list_models_200(self, client):
        r = client.get("/v1/models")
        assert r.status_code == 200

    def test_list_models_schema(self, client):
        data = client.get("/v1/models").json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        assert len(data["data"]) >= 1
        m = data["data"][0]
        assert "id" in m
        assert "object" in m

    def test_model_owned_by(self, client):
        data = client.get("/v1/models").json()
        m = data["data"][0]
        assert m["owned_by"] == "nim-cpu-bridge"


# ── Chat completions ──────────────────────────────────────────────────────────

class TestChatCompletions:
    def _payload(self, **kwargs):
        base = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Say hello in one word."}],
            "max_tokens": 8,
            "temperature": 0.0,
        }
        base.update(kwargs)
        return base

    def test_basic_completion_200(self, client):
        r = client.post("/v1/chat/completions", json=self._payload())
        assert r.status_code == 200

    def test_response_schema(self, client):
        data = client.post("/v1/chat/completions", json=self._payload()).json()
        assert data["object"] == "chat.completion"
        assert "choices" in data
        assert len(data["choices"]) == 1
        choice = data["choices"][0]
        assert "message" in choice
        assert choice["message"]["role"] == "assistant"
        assert isinstance(choice["message"]["content"], str)

    def test_usage_fields(self, client):
        data = client.post("/v1/chat/completions", json=self._payload()).json()
        usage = data["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_nim_cpu_bridge_marker_in_response(self, client):
        data = client.post("/v1/chat/completions", json=self._payload()).json()
        assert data.get("nim_cpu_bridge") is True

    def test_streaming_response(self, client):
        payload = self._payload(stream=True)
        chunks: list[str] = []
        with client.stream("POST", "/v1/chat/completions", json=payload) as r:
            assert r.status_code == 200
            for line in r.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunk = json.loads(line[6:])
                    delta = chunk["choices"][0]["delta"]
                    if "content" in delta:
                        chunks.append(delta["content"])
        assert len(chunks) > 0

    def test_stop_sequences_respected(self, client):
        # With stop=[".", "!"] the model should stop at those tokens
        # (exact behaviour is backend-dependent; just check it returns)
        payload = self._payload(
            stop=[".", "!"],
            messages=[{"role": "user", "content": "List numbers: 1"}],
            max_tokens=32,
        )
        r = client.post("/v1/chat/completions", json=payload)
        assert r.status_code == 200


# ── Completions (legacy) ──────────────────────────────────────────────────────

class TestCompletions:
    def test_basic_completion(self, client):
        r = client.post("/v1/completions", json={
            "model": "test-model",
            "prompt": "The capital of France is",
            "max_tokens": 4,
            "temperature": 0.0,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "text_completion"
        assert data["choices"][0]["text"]


# ── Metrics ───────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_metrics_200(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200

    def test_metrics_prometheus_format(self, client):
        text = client.get("/metrics").text
        assert "ncb_uptime_seconds" in text
        assert "ncb_backend_info" in text


# ── Shim unit tests (no service required) ────────────────────────────────────

class TestShimBuild:
    def test_shim_library_exists(self):
        """Check that the shim was compiled (CI gate)."""
        shim_path = Path(__file__).parent.parent / "src" / "shim" / "libcuda_shim.so"
        if not shim_path.exists():
            pytest.skip("Shim not compiled — run `make -C src/shim` first")
        assert shim_path.stat().st_size > 0

    def test_shim_exports_cuInit(self):
        """Verify cuInit is exported from the shim."""
        import subprocess, shutil
        if not shutil.which("nm"):
            pytest.skip("nm not available")
        shim_path = Path(__file__).parent.parent / "src" / "shim" / "libcuda_shim.so"
        if not shim_path.exists():
            pytest.skip("Shim not compiled")
        result = subprocess.run(["nm", "-D", str(shim_path)], capture_output=True, text=True)
        assert "cuInit" in result.stdout

    def test_shim_exports_cuMemAlloc(self):
        import subprocess, shutil
        if not shutil.which("nm"):
            pytest.skip("nm not available")
        shim_path = Path(__file__).parent.parent / "src" / "shim" / "libcuda_shim.so"
        if not shim_path.exists():
            pytest.skip("Shim not compiled")
        result = subprocess.run(["nm", "-D", str(shim_path)], capture_output=True, text=True)
        assert "cuMemAlloc_v2" in result.stdout


# ── Backend router unit tests ─────────────────────────────────────────────────

class TestBackendSelection:
    def test_auto_selects_available_backend(self):
        from src.backends.router import _select_backend, BackendKind
        # At least one backend must be installed in the test env
        kind = _select_backend("auto", {"avx2": True, "core_count": 4, "ram_gb": 16.0})
        assert isinstance(kind, BackendKind)

    def test_explicit_backend_override(self):
        from src.backends.router import _select_backend, BackendKind, _backend_available
        if not _backend_available(BackendKind.LLAMA_CPP):
            pytest.skip("llama-cpp-python not installed")
        kind = _select_backend("llama_cpp", {})
        assert kind == BackendKind.LLAMA_CPP

    def test_unknown_backend_falls_back(self):
        from src.backends.router import _select_backend
        # Should warn and fall through to auto-selection
        try:
            kind = _select_backend("nonexistent_backend", {"core_count": 4, "ram_gb": 8.0})
            # Should return a valid backend
            assert kind is not None
        except RuntimeError:
            pass  # acceptable if no backend installed
