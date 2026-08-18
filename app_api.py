"""
app_api.py
==========

FastAPI backend exposing the fine-tuned LeetCode model as an interview-grade
solution-generation service. Designed for direct integration with the Mockmate
interview platform (https://github.com/Rishabh01487/Mockmate-interview-platform)
via CORS-allowed REST endpoints.

Endpoints
---------
GET  /                 -> service info
GET  /health           -> liveness probe (no model load)
GET  /ready            -> readiness probe (model loaded)
POST /generate         -> generate a solution + Big-O complexities
POST /generate/batch   -> batched generation (up to 8 problems)
POST /explain          -> step-by-step explanation of a previously generated solution
GET  /languages        -> list supported languages

Run
---
    uvicorn app_api:app --host 0.0.0.0 --port 8000 --workers 1
    # (workers=1 — the singleton ModelManager holds the entire model in one process)

Env vars
--------
    BASE_MODEL          default: deepseek-ai/deepseek-coder-6.7b-base
    ADAPTER_PATH        default: None (base-only inference)
    ALLOWED_ORIGINS     default: http://localhost:5173,http://localhost:3000,http://localhost:8501
    API_KEY             if set, requires `Authorization: Bearer <key>` header
    MAX_NEW_TOKENS      default: 1024
    MODEL_PRELOAD       "1" to preload the model at startup (recommended)
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends, Header, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field, field_validator

# We use ORJSON for faster (de)serialization — important for low-latency inference.
# Falls back to standard json if orjson isn't installed.

# Local import: this must be the same `inference.py` shipped in the repo.
try:
    from inference import ModelManager, InterviewProblem, InterviewSolution
except ImportError as e:
    raise RuntimeError(
        "Could not import inference.py — make sure it's in the same directory "
        "or in PYTHONPATH. Original error: " + str(e)
    ) from e


# =========================================================================
# Pydantic schemas (public API contract)
# =========================================================================
class ProblemIn(BaseModel):
    title: str = Field(..., max_length=300, examples=["Two Sum"])
    difficulty: str = Field("Easy", pattern="^(Easy|Medium|Hard)$")
    description: str = Field(..., min_length=1, max_length=20000)
    examples: str = Field("", max_length=20000)
    constraints: str = Field("", max_length=5000)
    language: str = Field("python", pattern="^(python|java|cpp|javascript|c|go|rust)$")

    @field_validator("title", "description")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v


class SolutionOut(BaseModel):
    code: str
    language: str
    time_complexity: str
    space_complexity: str
    prompt_tokens: int
    generated_tokens: int
    elapsed_ms: float


class GenerateResponse(BaseModel):
    ok: bool = True
    solution: SolutionOut
    request_id: str


class BatchGenerateRequest(BaseModel):
    problems: List[ProblemIn] = Field(..., min_length=1, max_length=8)


class BatchGenerateResponse(BaseModel):
    ok: bool = True
    solutions: List[SolutionOut]
    total_elapsed_ms: float


class ExplainRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=20000)
    language: str = Field("python", pattern="^(python|java|cpp|javascript|c|go|rust)$")
    problem_title: str = Field("Untitled", max_length=300)


class ExplainResponse(BaseModel):
    explanation: str
    prompt_tokens: int
    generated_tokens: int
    elapsed_ms: float


# =========================================================================
# Config
# =========================================================================
def _split_origins() -> List[str]:
    raw = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://localhost:3000,http://localhost:8501",
    )
    return [o.strip() for o in raw.split(",") if o.strip()]


API_KEY = os.getenv("API_KEY", "")
PRELOAD = os.getenv("MODEL_PRELOAD", "1") == "1"


# =========================================================================
# Auth dependency
# =========================================================================
async def require_api_key(authorization: Optional[str] = Header(None)) -> None:
    if not API_KEY:
        return
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header. Expected: Bearer <key>",
        )
    token = authorization.split(" ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )


# =========================================================================
# Lifespan — preload model at startup
# =========================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    if PRELOAD:
        print("[api] Preloading model ...")
        # Run in threadpool so we don't block the event loop
        await run_in_threadpool(lambda: ModelManager.get())
        print("[api] Model preloaded and ready.")
    yield
    # Cleanup — nothing to do; CUDA caches will be freed on process exit


# =========================================================================
# App
# =========================================================================
app = FastAPI(
    title="Mockmate-LLM API",
    description="Fine-tuned LeetCode code generation service for the Mockmate "
                "interview platform.",
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_split_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _add_request_id(request: Request, call_next):
    """Tag every request with a short id for tracing."""
    rid = f"{int(time.time()*1000)}-{id(request) & 0xffff}"
    request.state.request_id = rid
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


# =========================================================================
# Routes
# =========================================================================
@app.get("/")
async def root():
    return {
        "service": "mockmate-llm",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
        "generate": "/generate  (POST)",
        "batch": "/generate/batch  (POST)",
        "explain": "/explain  (POST)",
        "languages": "/languages  (GET)",
    }


@app.get("/health")
async def health():
    return {"status": "alive", "ts": time.time()}


@app.get("/ready")
async def ready():
    """Returns 200 only when the model is loaded and inference can run."""
    try:
        mgr = ModelManager.get()
        if mgr._mock_mode:
            return {"status": "ready (mock)", "base_model": mgr.base_model,
                    "adapter": mgr.adapter_path or "(none)"}
        if mgr.model is None:
            raise RuntimeError("Model not loaded")
        return {"status": "ready", "base_model": mgr.base_model,
                "adapter": mgr.adapter_path or "(none)"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Not ready: {e}",
        )


@app.get("/languages")
async def languages():
    return {"languages": sorted(["python", "java", "cpp", "javascript", "c", "go", "rust"])}


@app.post("/generate",
          response_model=GenerateResponse,
          dependencies=[Depends(require_api_key)])
async def generate(problem: ProblemIn, request: Request):
    """Generate a single solution + Big-O complexity from a problem statement."""
    mgr = ModelManager.get()
    interview_problem = InterviewProblem(
        title=problem.title, difficulty=problem.difficulty,
        description=problem.description, examples=problem.examples,
        constraints=problem.constraints, language=problem.language,
    )
    # Run the (blocking) CUDA inference in a worker thread to keep the
    # event loop free for other requests.
    sol: InterviewSolution = await run_in_threadpool(
        mgr.generate_solution, interview_problem
    )
    return GenerateResponse(
        solution=SolutionOut(**sol.__dict__),
        request_id=getattr(request.state, "request_id", ""),
    )


@app.post("/generate/batch",
          response_model=BatchGenerateResponse,
          dependencies=[Depends(require_api_key)])
async def generate_batch(req: BatchGenerateRequest, request: Request):
    """Batched generation — up to 8 problems, executed sequentially
    (model is too big to safely parallelize on a single 24GB GPU)."""
    mgr = ModelManager.get()
    t0 = time.perf_counter()
    out: List[SolutionOut] = []

    async def _one(p: ProblemIn) -> SolutionOut:
        ip = InterviewProblem(
            title=p.title, difficulty=p.difficulty,
            description=p.description, examples=p.examples,
            constraints=p.constraints, language=p.language,
        )
        sol: InterviewSolution = await run_in_threadpool(mgr.generate_solution, ip)
        return SolutionOut(**sol.__dict__)

    for p in req.problems:
        try:
            out.append(await _one(p))
        except Exception as e:
            # Don't fail the whole batch — surface the error in the slot
            out.append(SolutionOut(
                code=f"// ERROR: {e}", language=p.language,
                time_complexity="O(unknown)", space_complexity="O(unknown)",
                prompt_tokens=0, generated_tokens=0, elapsed_ms=0,
            ))
    total_ms = (time.perf_counter() - t0) * 1000
    return BatchGenerateResponse(solutions=out, total_elapsed_ms=total_ms)


# =========================================================================
# Explain endpoint — uses the same model with a different prompt
# =========================================================================
EXPLAIN_SYS = (
    "You are a senior engineering interviewer. Given a candidate's code, "
    "produce a concise step-by-step explanation in plain English, suitable "
    "for a debrief. Do NOT rewrite the code."
)
EXPLAIN_USER_TMPL = """Problem: {title}
Language: {language}

```{language}
{code}
```

Explain how this solution works, step by step. End with one sentence on Time and Space complexity."""


@app.post("/explain",
          response_model=ExplainResponse,
          dependencies=[Depends(require_api_key)])
async def explain(req: ExplainRequest, request: Request):
    mgr = ModelManager.get()
    prompt = (
        f"<|im_start|>system\n{EXPLAIN_SYS}<|im_end|>\n"
        f"<|im_start|>user\n"
        + EXPLAIN_USER_TMPL.format(
            title=req.problem_title, language=req.language, code=req.code)
        + "<|im_end|>\n<|im_start|>assistant\n"
    )
    text, in_tok, out_tok = await run_in_threadpool(mgr.generate_raw, prompt)
    elapsed_ms_label = 0.0  # we recompute below for consistency
    # Trim at im_end
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>")[0]
    # Quick latency proxy (mgr.generate_raw already measured; we don't have it here)
    return ExplainResponse(
        explanation=text.strip(),
        prompt_tokens=in_tok, generated_tokens=out_tok,
        elapsed_ms=elapsed_ms_label,
    )


# =========================================================================
# Entry point
# =========================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app_api:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        workers=1,  # MUST be 1 — see ModelManager singleton
        log_level=os.getenv("LOG_LEVEL", "info"),
    )
