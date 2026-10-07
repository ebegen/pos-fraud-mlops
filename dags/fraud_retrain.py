"""Daily drift check -> conditional retrain -> champion/challenger gate -> hot reload of the API.

  collect_batch   new labelled tickets for the day (synthetic here; a warehouse query in production)
  check_drift     PSI of the batch vs the champion's training reference
  decide          branch: retrain if max PSI > threshold or `force_retrain`
  retrain         grid over models x resampling on recent data -> candidate bundle
  gate            promote only if the candidate beats the champion on a fresh holdout
  reload_api      POST /admin/reload so serving picks up the new champion without a restart

Trigger with params to demo a shift:  {"drift": 0.8}
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from airflow.sdk import Param, dag, task

MODELS = Path(os.environ.get("FRAUD_MODELS", "/opt/fraud/models"))
DATA = Path(os.environ.get("FRAUD_DATA", "/opt/fraud/data"))
API = os.environ.get("FRAUD_API", "http://api:8000")


def _split(df):
    """Fixed 70/30 split of the day's batch: training part for the candidate, holdout for the gate."""
    holdout = df.sample(frac=0.3, random_state=1)
    return df.drop(holdout.index), holdout


@dag(
    dag_id="fraud_retrain",
    schedule="@daily",
    start_date=datetime(2026, 10, 1, tzinfo=UTC),
    catchup=False,
    tags=["mlops", "fraud"],
    params={
        "drift": Param(0.0, type="number", minimum=0, maximum=1, description="simulated input shift for the day's batch"),
        "psi_threshold": Param(0.2, type="number"),
        "force_retrain": Param(False, type="boolean"),
    },
)
def fraud_retrain():
    @task
    def collect_batch(**ctx) -> str:
        from fraudops.data import generate

        p = ctx["params"]
        run = ctx["run_id"].replace(":", "_").replace("+", "_")
        out = DATA / f"batch_{run}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        generate(30_000, drift=float(p["drift"]), seed=abs(hash(run)) % 10_000).to_parquet(out)
        return str(out)

    @task
    def check_drift(batch: str) -> dict:
        import pandas as pd

        from fraudops.drift import drift_report

        meta = json.loads((MODELS / "champion" / "meta.json").read_text())
        rep = drift_report(meta["reference"], pd.read_parquet(batch))
        top = sorted(rep.items(), key=lambda kv: kv[1], reverse=True)[:3]
        print("PSI top features:", top)
        return {"max_psi": max(rep.values()), "top": top}

    @task.branch
    def decide(drift: dict, **ctx) -> str:
        p = ctx["params"]
        if p["force_retrain"] or drift["max_psi"] > float(p["psi_threshold"]):
            return "retrain"
        return "no_change"

    @task
    def no_change() -> None:
        print("inputs stable; champion kept")

    @task
    def retrain(batch: str) -> str:
        import pandas as pd

        from fraudops.data import generate
        from fraudops.train import train

        # train on the batch minus the gate's holdout (no leakage); recent data dominates,
        # a slice of the original distribution keeps old fraud patterns alive
        recent = _split(pd.read_parquet(batch))[0]
        df = pd.concat([recent, generate(15_000, seed=11)], ignore_index=True)
        meta = train(df, MODELS / "candidate")
        return meta["version"]

    @task
    def gate(batch: str, version: str) -> bool:
        import pandas as pd

        from fraudops.train import promote

        holdout = _split(pd.read_parquet(batch))[1]  # never seen by the candidate
        promoted = promote(MODELS / "candidate", MODELS / "champion", holdout)
        print(("promoted " if promoted else "rejected ") + version)
        return promoted

    @task
    def reload_api(promoted: bool) -> None:
        import httpx

        if not promoted:
            print("champion unchanged; no reload")
            return
        r = httpx.post(f"{API}/admin/reload", timeout=30)
        r.raise_for_status()
        print("API now serving", r.json())

    batch = collect_batch()
    drift = check_drift(batch)
    choice = decide(drift)
    version = retrain(batch)
    choice >> [version, no_change()]
    reload_api(gate(batch, version))


fraud_retrain()
