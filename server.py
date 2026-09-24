"""
Eclipse web server: the browser interface and a small JSON API around one
loaded model.

The model is loaded once, before the server starts listening. If the
checkpoint is missing or does not check out, the server does not start.

    venv/bin/python server.py                      # http://127.0.0.1:8000
    venv/bin/python server.py --checkpoint PATH --port 8080

    GET  /               the web interface (web/)
    GET  /model          how the model works: architecture and attention (web/model.html)
    GET  /model-3d       3D views of the token embeddings and of attention (web/model3d.html)
    GET  /api/info       what is loaded: checkpoint, architecture, provenance, limits
    GET  /api/embedding-projection?count=300&k=3   token embeddings: 3D PCA + nearest neighbors
    GET  /api/embedding-neighbors?token_id=ID&k=10  one token's neighbors in the whole vocabulary
    GET  /api/architecture   the loaded model's structure and parameter counts
    POST /api/attention  {"prompt": "..."}: attention weights from one forward pass
    POST /api/generate   {"prompt": "...", "max_new_tokens": 150, "temperature": 1.0,
                          "top_k": 40, "seed": null}
"""

import argparse
import sys
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from eclipse.attention import (MAX_ATTENTION_CHARS, AttentionCaptureError, capture_attention,
                               describe_architecture)
from eclipse.embedding import (DEFAULT_COUNT, DEFAULT_K, DEFAULT_NEIGHBORS, MAX_COUNT, MAX_K,
                               MAX_NEIGHBORS, MIN_COUNT, embedding_neighbors,
                               embedding_projection)
from eclipse.inference import (DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_K,
                               MAX_NEW_TOKENS_LIMIT, MAX_PROMPT_CHARS, MAX_TEMPERATURE,
                               SEED_LIMIT, Eclipse, EclipseError, GenerationRequestError,
                               GenerationSettings)

WEB_DIR = Path(__file__).resolve().parent / "web"


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    max_new_tokens: int = Field(DEFAULT_MAX_NEW_TOKENS, ge=1, le=MAX_NEW_TOKENS_LIMIT)
    temperature: float = Field(DEFAULT_TEMPERATURE, ge=0.0, le=MAX_TEMPERATURE,
                               allow_inf_nan=False)
    top_k: int | None = Field(DEFAULT_TOP_K, ge=1)
    seed: int | None = Field(None, ge=0, lt=SEED_LIMIT)


class AttentionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: str = Field(min_length=1, max_length=MAX_ATTENTION_CHARS)


class GenerateResponse(BaseModel):
    prompt: str
    completion: str
    prompt_tokens: int
    prompt_unknown_tokens: int
    generated_tokens: int
    stop_reason: str
    seed: int
    settings: dict
    elapsed_ms: float


def _error(status, message):
    return JSONResponse(status_code=status, content={"error": message})


def create_app(engine: Eclipse) -> FastAPI:
    app = FastAPI(title="Eclipse", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.engine = engine

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        problems = []
        for err in exc.errors():
            location = ".".join(str(part) for part in err["loc"] if part != "body")
            problems.append(f"{location}: {err['msg']}" if location else err["msg"])
        return _error(422, "; ".join(problems) or "invalid request")

    @app.exception_handler(GenerationRequestError)
    async def invalid_generation(request: Request, exc: GenerationRequestError):
        return _error(422, str(exc))

    @app.get("/api/info")
    def info():
        return engine.info.to_dict() if engine.info else {"name": "Eclipse"}

    @app.post("/api/generate", response_model=GenerateResponse)
    def generate(body: GenerateRequest):
        settings = GenerationSettings(max_new_tokens=body.max_new_tokens,
                                      temperature=body.temperature,
                                      top_k=body.top_k, seed=body.seed)
        result = engine.generate(body.prompt, settings)
        return GenerateResponse(
            prompt=result.prompt, completion=result.completion,
            prompt_tokens=result.prompt_tokens,
            prompt_unknown_tokens=result.prompt_unknown_tokens,
            generated_tokens=result.generated_tokens, stop_reason=result.stop_reason,
            seed=result.seed, settings=result.to_dict()["settings"],
            elapsed_ms=round(result.elapsed_seconds * 1000, 1))

    @app.exception_handler(AttentionCaptureError)
    async def unverified_attention(request: Request, exc: AttentionCaptureError):
        return _error(500, str(exc))

    @app.get("/api/architecture")
    def architecture():
        return describe_architecture(engine)

    @app.post("/api/attention")
    def attention(body: AttentionRequest):
        return capture_attention(engine, body.prompt)

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/model", include_in_schema=False)
    def model_page():
        return FileResponse(WEB_DIR / "model.html")

    @app.get("/api/embedding-projection")
    def projection(count: int = Query(DEFAULT_COUNT, ge=MIN_COUNT, le=MAX_COUNT),
                   k: int = Query(DEFAULT_K, ge=1, le=MAX_K)):
        return embedding_projection(engine, count, k)

    @app.get("/api/embedding-neighbors")
    def neighbors(token_id: int = Query(), k: int = Query(DEFAULT_NEIGHBORS, ge=1, le=MAX_NEIGHBORS)):
        return embedding_neighbors(engine, token_id, k)

    @app.get("/model-3d", include_in_schema=False)
    def model_3d_page():
        return FileResponse(WEB_DIR / "model3d.html")

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description="Serve Eclipse in the browser")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint path (default: $ECLIPSE_CHECKPOINT, else "
                             "checkpoints/sr_data200mb_s1337_best.pt)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="interface to listen on (default 127.0.0.1: this machine only)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cpu", help="cpu (default), mps, or auto")
    parser.add_argument("--skip-provenance-check", action="store_true",
                        help="serve weights even if they cannot be matched to the research record")
    args = parser.parse_args(argv)

    try:
        engine = Eclipse.load(args.checkpoint, device=args.device,
                              verify_provenance=not args.skip_provenance_check)
    except EclipseError as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(1)

    info = engine.info
    arch = info.architecture
    print(f"Eclipse loaded: {info.checkpoint} (step {info.step}), {info.parameters:,} parameters, "
          f"d_model {arch['d_model']}, {arch['num_heads']} heads, {arch['num_blocks']} blocks, "
          f"device {info.device}")
    print(f"provenance: {info.provenance['status']} - {info.provenance['detail']}")

    import uvicorn
    uvicorn.run(create_app(engine), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
