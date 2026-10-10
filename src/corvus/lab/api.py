"""FastAPI inference service and local recorded-signal sandbox."""
from contextlib import asynccontextmanager
import math
from pathlib import Path
import time
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from corvus.data import transform
from .runtime import LabService

MAX_BODY_BYTES = 512 * 1024


class BodyLimit:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        try:
            too_large = int(headers.get(b"content-length", b"0")) > MAX_BODY_BYTES
        except ValueError:
            too_large = True
        chunks, size = [], 0
        if not too_large:
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                size += len(chunk)
                if size > MAX_BODY_BYTES:
                    too_large = True
                    break
                chunks.append(chunk)
                if not message.get("more_body", False):
                    break
        if too_large:
            return await JSONResponse({"detail": "Request body exceeds 512 KiB"}, 413)(scope, receive, send)
        delivered = False

        async def replay_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReplayInput(StrictModel):
    record_id: str = Field(min_length=1, max_length=40)
    model_id: str = Field(default="ecg-onnx", min_length=1, max_length=40)
    corruption: Literal["clean", "noise", "drift", "dropout", "sampling", "gain"] = "noise"
    severity: FiniteFloat = Field(default=0.5, ge=0, le=1)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)


class PredictInput(StrictModel):
    signal: list[Annotated[FiniteFloat, Field(ge=-1000, le=1000)]] = Field(min_length=1000, max_length=10000)
    sample_rate_hz: FiniteFloat = Field(ge=100, le=1000)
    model_id: str = Field(default="ecg-onnx", min_length=1, max_length=40)

    @model_validator(mode="after")
    def duration(self):
        if abs(len(self.signal) / self.sample_rate_hz - 10.0) > 1.01 / self.sample_rate_hz:
            raise ValueError("Expected a ten-second single-lead ECG in millivolts")
        return self


def create_app(directory="artifacts/lab", service_factory=LabService):
    @asynccontextmanager
    async def lifespan(app):
        app.state.service = service_factory(directory)
        yield

    app = FastAPI(title="Corvus Model Lab", version="0.2.0", lifespan=lifespan,
                  description="Recorded ECG stress tests and versioned inference. Research demonstration with weak machine labels.")
    app.add_middleware(BodyLimit)
    web = Path(__file__).parent / "web"

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # Never echo raw waveforms or nonfinite input values in error responses.
        detail = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        return JSONResponse({"detail": detail}, status_code=422)

    @app.middleware("http")
    async def observe(request: Request, call_next):
        response = await call_next(request)
        service = getattr(app.state, "service", None)
        if service:
            with service.lock:
                service.metrics["requests"] += 1
                service.metrics["errors"] += int(response.status_code >= 400)
        response.headers["X-Content-Type-Options"] = "nosniff"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        if request.url.path in ("/", "/assets/app.js", "/assets/styles.css"):
            response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'"
        return response

    @app.get("/health")
    def health():
        s = app.state.service
        return {"status": "ready", "models_loaded": len(s.specs),
                "models_passing_gates": sum(v["gate"]["passed"] for v in s.specs.values())}

    @app.get("/api/v1/summary")
    def summary():
        s = app.state.service
        report = {k: v for k, v in s.report.items() if k not in {"test_inventory", "models"}}
        return {**report, "models": list(s.specs.values()), "benchmarks": s.report["models"]}

    @app.get("/api/v1/records")
    def records():
        return app.state.service.records

    @app.post("/api/v1/replay")
    def replay(body: ReplayInput):
        try:
            return app.state.service.replay(body.record_id, body.model_id, body.corruption, body.severity, body.seed)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/v1/predict")
    def predict(body: PredictInput):
        s = app.state.service
        if body.model_id not in s.specs:
            raise HTTPException(404, "Unknown model")
        spec = s.specs[body.model_id]
        if not spec["gate"]["passed"]:
            raise HTTPException(409, {"code": "model_not_promoted", "message": "This model failed release checks. Explore it through the sandbox replay endpoint.",
                                      "checks": spec["gate"]["checks"]})
        start = time.perf_counter()
        try:
            x = transform(np.asarray(body.signal, dtype=np.float32))[None]
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        preprocessing_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        score = float(s.predictors[body.model_id].score(x)[0])
        inference_ms = (time.perf_counter() - start) * 1000
        if not math.isfinite(score):
            raise HTTPException(500, "Model returned a nonfinite score")
        with s.lock:
            s.metrics["inferences"] += 1
            s.metrics["inference_seconds"] += inference_ms / 1000
        return {"model_id": body.model_id, "model_version": spec["version"], "score": score,
                "threshold": spec["threshold"], "prediction": int(score > spec["threshold"]),
                "inference_ms": inference_ms, "preprocessing_ms": preprocessing_ms,
                "score_meaning": "PVC-report class score; not a calibrated diagnostic probability",
                "release_scope": "Engineering demonstration only"}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics():
        s = app.state.service
        with s.lock:
            m = dict(s.metrics)
        return "\n".join([
            "# TYPE corvus_http_requests_total counter", f"corvus_http_requests_total {m['requests']}",
            "# TYPE corvus_http_errors_total counter", f"corvus_http_errors_total {m['errors']}",
            "# TYPE corvus_inferences_total counter", f"corvus_inferences_total {m['inferences']}",
            "# TYPE corvus_inference_seconds_total counter", f"corvus_inference_seconds_total {m['inference_seconds']:.9f}",
            "# TYPE corvus_models_loaded gauge", f"corvus_models_loaded {len(s.specs)}", ""])

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(web / "index.html")

    app.mount("/assets", StaticFiles(directory=web), name="assets")
    return app
