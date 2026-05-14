"""
nim-cpu-bridge: orchestrator/main.py

FastAPI sidecar that:
  1. Boots the CPU backend router and bridge socket server
  2. Auto-converts model weights if NCB_CONVERT_ON_START=true
  3. Proxies NIM's OpenAI-compatible HTTP API with CPU inference
  4. Rewrites /health and /v1/models to return plausible NIM responses
  5. Exposes Prometheus metrics at /metrics
  6. Injects the CUDA shim env vars via subprocess before launching NIM

This process replaces NIM's vLLM/Triton server for LLM inference.
For non-LLM NIM tasks (embedding, reranking), it may optionally proxy
to a compatible CPU-side replacement.

Apache 2.0 — see LICENSE
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..backends.router import Router, build_router
from ..converter.ngc_to_gguf import prepare_model, ConversionError

logger = logging.getLogger("ncb.orchestrator")

# ── Startup state ─────────────────────────────────────────────────────────────

_router: Router | None = None
_model_id: str = ""
_start_time: float = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _router, _model_id

    model_tag = os.getenv("NIM_MODEL_NAME", "unknown")
    _model_id = model_tag
    quant     = os.getenv("NCB_QUANT",    "Q4_K_M")
    backend   = os.getenv("NCB_BACKEND",  "auto")

    convert_on_start = os.getenv("NCB_CONVERT_ON_START", "true").lower() == "true"

    logger.info("nim-cpu-bridge orchestrator starting for model: %s", model_tag)

    if convert_on_start and model_tag != "unknown":
        try:
            model_path = prepare_model(model_tag, quant_type=quant, backend=backend)
            os.environ["NCB_MODEL_PATH"] = str(model_path)
            logger.info("Model path set to %s", model_path)
        except ConversionError as e:
            logger.warning("Weight conversion skipped: %s", e)
            logger.warning("Set NCB_MODEL_PATH manually if using a pre-converted GGUF.")

    try:
        _router = build_router()
        logger.info("CPU backend router ready")
    except Exception as e:
        logger.error("Failed to start backend router: %s", e)
        raise

    yield  # Application runs

    logger.info("Shutting down nim-cpu-bridge")


app = FastAPI(title="nim-cpu-bridge", version="0.1.0", lifespan=lifespan)


# ── Request / response schemas (OpenAI-compatible) ────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int                  = Field(default=256, ge=1, le=8192)
    temperature: float               = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float                     = Field(default=0.95, ge=0.0, le=1.0)
    stream: bool                     = False
    stop: list[str] | None           = None
    n: int                           = 1


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int                  = Field(default=256, ge=1)
    temperature: float               = Field(default=0.7)
    top_p: float                     = Field(default=0.95)
    stream: bool                     = False
    stop: list[str] | None           = None


# ── Prompt formatting ─────────────────────────────────────────────────────────

def _apply_chat_template(messages: list[ChatMessage]) -> str:
    """
    Minimal llama-2 / ChatML template.
    Production use should load tokenizer_config.json from the model dir.
    """
    parts: list[str] = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"<<SYS>>\n{msg.content}\n<</SYS>>\n")
        elif msg.role == "user":
            parts.append(f"[INST] {msg.content} [/INST]")
        elif msg.role == "assistant":
            parts.append(f" {msg.content} ")
    return "".join(parts)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_chat_chunk(delta_text: str, finish: str | None, req_id: str) -> str:
    chunk = {
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": _model_id,
        "choices": [{
            "index": 0,
            "delta": {"content": delta_text} if delta_text else {},
            "finish_reason": finish,
        }],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def _make_chat_response(text: str, req_id: str, prompt_tokens: int, gen_tokens: int) -> dict[str, Any]:
    return {
        "id": req_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _model_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": gen_tokens,
            "total_tokens": prompt_tokens + gen_tokens,
        },
        "nim_cpu_bridge": True,  # marker so clients can detect shim mode
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> JSONResponse:
    ready = _router is not None
    return JSONResponse(
        {"status": "ready" if ready else "loading", "nim_cpu_bridge": True},
        status_code=200 if ready else 503,
    )


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    return JSONResponse({
        "object": "list",
        "data": [{
            "id": _model_id,
            "object": "model",
            "created": int(_start_time),
            "owned_by": "nim-cpu-bridge",
            "permission": [],
        }],
    })


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest) -> Any:
    if not _router:
        raise HTTPException(503, "Backend not ready")

    prompt = _apply_chat_template(req.messages)
    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    if req.stream:
        async def _stream() -> AsyncIterator[str]:
            yield _make_chat_chunk("", None, req_id)  # role delta
            loop = asyncio.get_event_loop()
            gen = await loop.run_in_executor(
                None,
                lambda: list(_router.generate(  # type: ignore[union-attr]
                    prompt=prompt,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    stop=req.stop,
                    stream=True,
                )),
            )
            for chunk_text in gen:
                yield _make_chat_chunk(chunk_text, None, req_id)
            yield _make_chat_chunk("", "stop", req_id)
            yield "data: [DONE]\n\n"

        return StreamingResponse(_stream(), media_type="text/event-stream")

    # Non-streaming
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(
        None,
        lambda: "".join(_router.generate(  # type: ignore[union-attr]
            prompt=prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stop=req.stop,
        )),
    )
    prompt_tokens = len(prompt.split())
    gen_tokens    = len(text.split())
    return JSONResponse(_make_chat_response(text, req_id, prompt_tokens, gen_tokens))


@app.post("/v1/completions")
async def completions(req: CompletionRequest) -> Any:
    if not _router:
        raise HTTPException(503, "Backend not ready")

    req_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(
        None,
        lambda: "".join(_router.generate(  # type: ignore[union-attr]
            prompt=req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stop=req.stop,
        )),
    )
    return JSONResponse({
        "id": req_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": _model_id,
        "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": len(req.prompt.split()),
            "completion_tokens": len(text.split()),
        },
    })


@app.get("/metrics")
async def metrics() -> Any:
    """Minimal Prometheus-compatible metrics."""
    uptime = time.time() - _start_time
    backend_name = _router.handle.kind.name if _router else "none"
    lines = [
        "# HELP ncb_uptime_seconds Orchestrator uptime",
        "# TYPE ncb_uptime_seconds gauge",
        f"ncb_uptime_seconds {uptime:.1f}",
        "# HELP ncb_backend_info Active CPU backend",
        f'ncb_backend_info{{backend="{backend_name}"}} 1',
        "# HELP ncb_model_info Active model",
        f'ncb_model_info{{model="{_model_id}"}} 1',
    ]
    from starlette.responses import PlainTextResponse
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


@app.middleware("http")
async def log_requests(request: Request, call_next: Any) -> Any:
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    logger.info("%s %s → %d (%.3fs)", request.method, request.url.path,
                response.status_code, elapsed)
    return response


# ── CLI entry ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, os.getenv("NCB_LOG_LEVEL", "INFO")),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    host = os.getenv("NCB_HOST", "0.0.0.0")
    port = int(os.getenv("NCB_PORT", "8000"))
    uvicorn.run(app, host=host, port=port, log_config=None)
