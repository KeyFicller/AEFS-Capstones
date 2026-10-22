"""FastAPI entry for the three stages: one POST returns all three replies.

`GET /` is the form for asking by hand; it is `serve.py`'s page, loaded by path
so the markup lives in one place. `/docs` is the machine-readable surface.

"At the same time" is one request, not three parallel forwards. `Bundle.run`
walks the stages in order and seeds torch globally per stage: the seed contract
in `generate.py` and the MPS backend both need one model running at a time.

    ./start.sh run uvicorn --app-dir projects/tiny-llm-pipeline/eval api:app --port 8000
"""

from __future__ import annotations

import importlib.util
import os
import secrets
import time
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from tiny_llm_pipeline.generate import Bundle, GenSettings

BUNDLE = Path(__file__).resolve().parents[1] / "checkpoints"

app = FastAPI(title="tiny-llm-pipeline", description="pretrain / sft / dpo, one question")


@app.exception_handler(RequestValidationError)
def _invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
    """`{"error": ...}`, the shape `serve.py`'s page reads.

    Its page reports `data.error` and leaves the previous cards in place on any
    non-200, so FastAPI's default `{"detail": [...]}` would show "request
    failed" with no reason and stale replies -- which looks like the knob did
    nothing.
    """
    return JSONResponse(
        {
            "error": "; ".join(
                f"{'.'.join(map(str, e['loc'][1:]))}: {e['msg']}" for e in exc.errors()
            )
        },
        status_code=422,
    )


@lru_cache(maxsize=1)
def get_bundle() -> Bundle:
    """Loaded once, on the first request. `TINY_LLM_BUNDLE` picks another directory."""
    return Bundle.load(os.environ.get("TINY_LLM_BUNDLE", BUNDLE))


class Ask(BaseModel):
    """One question and the three knobs the page exposes.

    Ranges are rejected with a 422, not clamped: `/docs` is the test UI, so a
    bad number should be visible as a bad number.
    """

    question: str = Field(min_length=1, description="Anything; pretrain continues it.")
    max_new: int = Field(128, ge=1, le=512)
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="0 is greedy.")
    rep_pen: float = Field(1.5, ge=1.0, le=2.0, description="Delivered value is 1.5.")
    seed: int | None = Field(
        None,
        ge=0,
        description="Pin it to replay a reply; leave it out for a fresh sample each call.",
    )


@lru_cache(maxsize=1)
def _page():
    """`serve.py` loaded by path: `eval/` is not a package."""
    spec = importlib.util.spec_from_file_location(
        "serve_page", Path(__file__).with_name("serve.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@app.get("/", response_class=HTMLResponse)
def form(bundle: Annotated[Bundle, Depends(get_bundle)]) -> str:
    """The page: one textarea, and three cards that fill from one POST."""
    return _page().render_page(bundle)


@app.post("/ask")
def ask(body: Ask, bundle: Annotated[Bundle, Depends(get_bundle)]) -> dict[str, object]:
    """Every loaded stage's reply to `question`, in stage order."""
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "question must not be blank")
    # `Bundle.run` seeds torch from (seed, stage, rep_pen, prompt, index) and
    # always uses index 0, so a pinned seed replays the same text forever. A
    # page that hits 生成 twice wants two samples, so the seed is fresh unless
    # the caller pins one -- and it comes back in the reply to be replayed.
    seed = body.seed if body.seed is not None else secrets.randbelow(2**31)
    settings = GenSettings(
        max_new=body.max_new, temperature=body.temperature, rep_pen=body.rep_pen, seed=seed
    )
    started = time.perf_counter()
    try:
        replies = bundle.run(question, settings)
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc
    return {
        "device": bundle.device.type,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
        "seed": seed,
        "stages": [asdict(reply) for reply in replies],
    }
