"""
FastAPI entrypoint for the PACT interactive web demo.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .demo_service import (
    DEFAULT_DEMO_TIMEOUT_SECONDS,
    DemoValidationError,
    build_demo_config,
    list_sample_texts,
    run_demo_rewrite,
)

DEMO_RATE_LIMIT_WINDOW_SECONDS = 60
DEMO_RATE_LIMIT_MAX_REQUESTS = 12


class DemoRewriteRequest(BaseModel):
    text: str = Field(default="", max_length=6000)
    target_ratio: float = 0.4
    mixing_mode: str = "block_replace"
    model: str = "MiniMax-M2.7"
    seed: int = Field(default=42, ge=0, le=1_000_000)
    language_hint: str = Field(default="English", max_length=32)


def create_app() -> FastAPI:
    app = FastAPI(
        title="PACT Interactive Demo",
        description="Interactive single-text construction demo for PACT.",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    request_log: dict[str, deque[float]] = defaultdict(deque)

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        if request.url.path == "/api/rewrite":
            rate_limit_error = _rate_limit_error(request, request_log)
            if rate_limit_error:
                return JSONResponse(
                    status_code=429,
                    content={"detail": rate_limit_error},
                )
        return await call_next(request)

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/config")
    async def config() -> dict[str, Any]:
        return build_demo_config()

    @app.get("/api/samples")
    async def samples() -> dict[str, Any]:
        return {"samples": list_sample_texts()}

    @app.post("/api/rewrite")
    async def rewrite(payload: DemoRewriteRequest) -> dict[str, Any]:
        try:
            result = await run_demo_rewrite(
                text=payload.text,
                target_ratio=payload.target_ratio,
                mixing_mode=payload.mixing_mode,
                model=payload.model,
                seed=payload.seed,
                language_hint=payload.language_hint,
                timeout_seconds=DEFAULT_DEMO_TIMEOUT_SECONDS,
            )
            return _jsonable(asdict(result))
        except DemoValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    _mount_frontend(app)
    return app


def _client_key(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _rate_limit_error(
    request: Request,
    request_log: dict[str, deque[float]],
) -> str:
    now = time.monotonic()
    key = _client_key(request)
    bucket = request_log[key]
    while bucket and now - bucket[0] > DEMO_RATE_LIMIT_WINDOW_SECONDS:
        bucket.popleft()
    if len(bucket) >= DEMO_RATE_LIMIT_MAX_REQUESTS:
        return (
            "请求过于频繁，请稍后再试。公开 Demo 默认限制为 "
            f"{DEMO_RATE_LIMIT_MAX_REQUESTS} 次/分钟。"
        )
    bucket.append(now)
    return ""


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _mount_frontend(app: FastAPI) -> None:
    dist_dir = Path(__file__).resolve().parent.parent / "web" / "dist"
    if not dist_dir.exists():
        return

    assets_dir = dist_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str) -> FileResponse:
        candidate = dist_dir / full_path
        if full_path and candidate.exists() and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(dist_dir / "index.html")


app = create_app()
