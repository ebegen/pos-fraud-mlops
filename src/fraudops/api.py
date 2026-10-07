"""Model serving API with Prometheus metrics and live drift monitoring.

Endpoints
  POST /predict          one ticket or a batch -> fraud probability + decision
  GET  /health           liveness + loaded model version
  GET  /model            model card (metrics, leaderboard, threshold)
  POST /admin/reload     reload the champion bundle (the Airflow DAG calls this after a promotion)
  GET  /metrics          Prometheus scrape endpoint

Metrics
  fraud_requests_total{status}, fraud_request_latency_seconds (histogram),
  fraud_predictions_total{decision}, fraud_score (histogram),
  fraud_feature_psi{feature} and fraud_drift_max_psi (rolling window vs training reference),
  fraud_model_info{version,model,sampler}

Drift: the last DRIFT_WINDOW tickets are kept in memory; PSI per feature is recomputed every
DRIFT_EVERY requests against the reference bins saved with the model.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from fraudops.drift import drift_report

DRIFT_WINDOW = int(os.environ.get("DRIFT_WINDOW", "2000"))
DRIFT_EVERY = int(os.environ.get("DRIFT_EVERY", "200"))

REQUESTS = Counter("fraud_requests_total", "Prediction requests", ["status"])
LATENCY = Histogram("fraud_request_latency_seconds", "Prediction latency", buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
DECISIONS = Counter("fraud_predictions_total", "Scored tickets by decision", ["decision"])
SCORE = Histogram("fraud_score", "Predicted fraud probability", buckets=[i / 20 for i in range(1, 21)])
PSI = Gauge("fraud_feature_psi", "Population stability index vs training data", ["feature"])
MAX_PSI = Gauge("fraud_drift_max_psi", "Largest feature PSI in the current window")
WINDOW_SIZE = Gauge("fraud_drift_window_size", "Tickets in the drift window")
MODEL_INFO = Gauge("fraud_model_info", "Loaded model", ["version", "model", "sampler"])
THRESHOLD = Gauge("fraud_model_threshold", "Decision threshold of the loaded model")


class Ticket(BaseModel):
    ticket_total: float = Field(ge=0)
    n_items: int = Field(ge=0)
    n_voids: int = Field(ge=0)
    void_amount: float = Field(ge=0)
    discount_pct: float = Field(ge=0, le=100)
    refund_amount: float = Field(ge=0)
    is_cash: int = Field(ge=0, le=1)
    drawer_opens: int = Field(ge=0)
    minutes_since_shift_start: float = Field(ge=0)
    hour: int = Field(ge=0, le=23)
    cashier_tenure_days: float = Field(ge=0)
    cashier_void_rate_7d: float = Field(ge=0, le=1)


class Batch(BaseModel):
    tickets: list[Ticket] = Field(min_length=1, max_length=5000)


class Scorer:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.lock = threading.Lock()
        self.window: deque[dict] = deque(maxlen=DRIFT_WINDOW)
        self.since_check = 0
        self.load()

    def load(self) -> dict:
        pipe = joblib.load(self.model_dir / "model.joblib")
        meta = json.loads((self.model_dir / "meta.json").read_text())
        with self.lock:
            self.pipe, self.meta = pipe, meta
            self.window.clear()
        MODEL_INFO.clear()
        MODEL_INFO.labels(meta["version"], meta["model"], meta["sampler"]).set(1)
        THRESHOLD.set(meta["threshold"])
        return meta

    def score(self, rows: list[dict]) -> list[dict]:
        df = pd.DataFrame(rows)[self.meta["features"]]
        with self.lock:
            p = self.pipe.predict_proba(df)[:, 1]
            thr = self.meta["threshold"]
        out = []
        for prob in p:
            decision = "fraud" if prob >= thr else "ok"
            DECISIONS.labels(decision).inc()
            SCORE.observe(float(prob))
            out.append({"fraud_probability": round(float(prob), 4), "decision": decision})
        self.window.extend(rows)
        self.since_check += len(rows)
        if self.since_check >= DRIFT_EVERY:
            self.since_check = 0
            self.check_drift()
        return out

    def check_drift(self) -> dict[str, float]:
        if len(self.window) < 200:
            return {}
        rep = drift_report(self.meta["reference"], pd.DataFrame(list(self.window)))
        for f, v in rep.items():
            PSI.labels(f).set(v)
        MAX_PSI.set(max(rep.values(), default=0.0))
        WINDOW_SIZE.set(len(self.window))
        return rep


scorer: Scorer | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global scorer
    scorer = Scorer(Path(os.environ.get("MODEL_DIR", "models/champion")))
    yield


app = FastAPI(title="POS fraud scoring", version="0.1.0", lifespan=lifespan)


def _scorer() -> Scorer:
    if scorer is None:
        raise HTTPException(503, "model not loaded")
    return scorer


@app.post("/predict")
def predict(body: Ticket | Batch):
    t0 = time.perf_counter()
    try:
        rows = [t.model_dump() for t in body.tickets] if isinstance(body, Batch) else [body.model_dump()]
        res = _scorer().score(rows)
        REQUESTS.labels("ok").inc()
        return {"model_version": _scorer().meta["version"], "results": res}
    except HTTPException:
        REQUESTS.labels("error").inc()
        raise
    except Exception as e:
        REQUESTS.labels("error").inc()
        raise HTTPException(500, str(e)) from e
    finally:
        LATENCY.observe(time.perf_counter() - t0)


@app.get("/health")
def health():
    s = _scorer()
    return {"status": "ok", "model_version": s.meta["version"], "window": len(s.window)}


@app.get("/model")
def model_card():
    m = dict(_scorer().meta)
    m.pop("reference", None)
    return m


@app.get("/drift")
def drift():
    return _scorer().check_drift()


@app.post("/admin/reload")
def reload_model():
    meta = _scorer().load()
    return {"reloaded": meta["version"]}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
