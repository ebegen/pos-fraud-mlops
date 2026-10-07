"""Train and compare fraud models, pick a decision threshold, save a model bundle.

Grid: {RandomForest, XGBoost, LightGBM} x {no resampling, ADASYN, Tomek links}.
Selection metric: PR-AUC on the validation split (accuracy is meaningless at ~1.5% fraud).
The threshold maximises F1 on validation; test metrics are reported once, at that threshold.

Bundle layout (models/<name>/):
  model.joblib  - fitted imblearn/sklearn pipeline
  meta.json     - features, threshold, metrics, PSI reference bins, version, trained_at

If MLFLOW_TRACKING_URI is set, every grid cell is logged as an MLflow run and the winner is tagged.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import TomekLinks
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from fraudops.data import FEATURES, TARGET, generate
from fraudops.drift import reference_bins

# Bounded thread count: OpenMP libraries oversubscribe badly in shared containers/Airflow workers
JOBS = int(os.environ.get("FRAUDOPS_JOBS", min(4, os.cpu_count() or 1)))

MODELS = {
    "random_forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=JOBS, random_state=0),
    "xgboost": lambda: XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.06, subsample=0.9, colsample_bytree=0.9, eval_metric="aucpr", n_jobs=JOBS, random_state=0
    ),
    "lightgbm": lambda: LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, n_jobs=JOBS, verbose=-1, random_state=0),
}
SAMPLERS = {
    "none": lambda: None,
    "adasyn": lambda: ADASYN(random_state=0),
    "tomek": lambda: TomekLinks(),
}


@dataclass
class Result:
    model: str
    sampler: str
    val_pr_auc: float
    threshold: float
    pipe: Pipeline
    fit_seconds: float


def build(model: str, sampler: str) -> Pipeline:
    steps = []
    s = SAMPLERS[sampler]()
    if s is not None:
        steps.append(("resample", s))
    steps.append(("clf", MODELS[model]()))
    return Pipeline(steps)


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    return float(thr[int(np.argmax(f1[:-1]))]) if len(thr) else 0.5


def evaluate(pipe: Pipeline, X: pd.DataFrame, y: np.ndarray, threshold: float) -> dict[str, float]:
    p = pipe.predict_proba(X)[:, 1]
    yhat = (p >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
    }


def run_grid(train: pd.DataFrame, val: pd.DataFrame, models: list[str], samplers: list[str]) -> list[Result]:
    mlflow = _mlflow()
    out = []
    for m in models:
        for s in samplers:
            t0 = time.time()
            pipe = build(m, s).fit(train[FEATURES], train[TARGET])
            secs = time.time() - t0
            p = pipe.predict_proba(val[FEATURES])[:, 1]
            r = Result(m, s, float(average_precision_score(val[TARGET], p)), best_threshold(val[TARGET].to_numpy(), p), pipe, secs)
            out.append(r)
            print(f"  {m:14s} {s:7s} val PR-AUC {r.val_pr_auc:.4f}  ({secs:.1f}s)")
            if mlflow:
                with mlflow.start_run(run_name=f"{m}-{s}"):
                    mlflow.log_params({"model": m, "sampler": s})
                    mlflow.log_metrics({"val_pr_auc": r.val_pr_auc, "fit_seconds": secs})
    return sorted(out, key=lambda r: r.val_pr_auc, reverse=True)


def train(df: pd.DataFrame, out_dir: Path, models: list[str] | None = None, samplers: list[str] | None = None) -> dict:
    train_df, rest = train_test_split(df, test_size=0.4, stratify=df[TARGET], random_state=0)
    val_df, test_df = train_test_split(rest, test_size=0.5, stratify=rest[TARGET], random_state=0)
    results = run_grid(train_df, val_df, models or list(MODELS), samplers or list(SAMPLERS))
    best = results[0]
    test = evaluate(best.pipe, test_df[FEATURES], test_df[TARGET].to_numpy(), best.threshold)
    meta = {
        "version": datetime.now(UTC).strftime("%Y%m%d-%H%M%S"),
        "trained_at": datetime.now(UTC).isoformat(),
        "model": best.model,
        "sampler": best.sampler,
        "threshold": best.threshold,
        "features": FEATURES,
        "n_train": len(train_df),
        "fraud_rate_train": float(train_df[TARGET].mean()),
        "val_pr_auc": best.val_pr_auc,
        "test": test,
        "leaderboard": [{"model": r.model, "sampler": r.sampler, "val_pr_auc": round(r.val_pr_auc, 4)} for r in results],
        "reference": reference_bins(train_df, FEATURES),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best.pipe, out_dir / "model.joblib")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    mlflow = _mlflow()
    if mlflow:
        with mlflow.start_run(run_name=f"winner-{best.model}-{best.sampler}"):
            mlflow.set_tag("winner", "true")
            mlflow.log_params({"model": best.model, "sampler": best.sampler, "threshold": round(best.threshold, 4)})
            mlflow.log_metrics({f"test_{k}": v for k, v in test.items()} | {"val_pr_auc": best.val_pr_auc})
            mlflow.log_artifacts(str(out_dir), artifact_path="bundle")
    print(f"winner: {best.model} + {best.sampler} · test {json.dumps({k: round(v, 4) for k, v in test.items()})} -> {out_dir}")
    return meta


def promote(candidate: Path, champion: Path, eval_df: pd.DataFrame, min_gain: float = 0.0) -> bool:
    """Champion/challenger gate: the candidate replaces the champion only if PR-AUC on fresh data is at least as good."""
    cand = joblib.load(candidate / "model.joblib")
    cand_score = average_precision_score(eval_df[TARGET], cand.predict_proba(eval_df[FEATURES])[:, 1])
    if (champion / "model.joblib").exists():
        champ = joblib.load(champion / "model.joblib")
        champ_score = average_precision_score(eval_df[TARGET], champ.predict_proba(eval_df[FEATURES])[:, 1])
    else:
        champ_score = -1.0
    print(f"promotion check: candidate PR-AUC {cand_score:.4f} vs champion {champ_score:.4f}")
    if cand_score < champ_score + min_gain:
        return False
    tmp = champion.with_name(champion.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(candidate, tmp)
    shutil.rmtree(champion, ignore_errors=True)
    tmp.rename(champion)  # swap the whole directory so the API never sees a half-written bundle
    return True


def _mlflow():
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return None
    import mlflow

    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "pos-fraud"))
    return mlflow


def main() -> None:
    ap = argparse.ArgumentParser(description="Train POS fraud models")
    ap.add_argument("--data", type=Path, help="parquet/csv; generated if omitted")
    ap.add_argument("--n", type=int, default=60_000)
    ap.add_argument("--drift", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("models/candidate"))
    ap.add_argument("--promote-to", type=Path, help="e.g. models/champion")
    ap.add_argument("--fast", action="store_true", help="xgboost only, no resampling grid (CI)")
    a = ap.parse_args()
    if a.data:
        df = pd.read_parquet(a.data) if a.data.suffix == ".parquet" else pd.read_csv(a.data)
    else:
        df = generate(a.n, drift=a.drift, seed=a.seed)
    meta = train(df, a.out, ["xgboost"] if a.fast else None, ["none"] if a.fast else None)
    if a.promote_to:
        fresh = generate(20_000, drift=a.drift, seed=a.seed + 1000)
        print("promoted" if promote(a.out, a.promote_to, fresh) else "kept champion", meta["version"])


if __name__ == "__main__":
    main()
