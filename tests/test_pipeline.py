import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fraudops.data import FEATURES, TARGET, generate
from fraudops.drift import drift_report, reference_bins
from fraudops.train import promote, train


def test_generate_shape_and_rate():
    df = generate(5000, seed=1)
    assert list(df.columns) == FEATURES + [TARGET]
    assert 0.01 <= df[TARGET].mean() <= 0.02


def test_psi_detects_drift_but_not_noise():
    base = generate(8000, seed=1)
    ref = reference_bins(base, FEATURES)
    same = drift_report(ref, generate(8000, seed=2))
    shifted = drift_report(ref, generate(8000, drift=0.9, seed=3))
    assert max(same.values()) < 0.1
    assert max(shifted.values()) > 0.2
    assert shifted["ticket_total"] > same["ticket_total"]


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    out = tmp_path_factory.mktemp("models") / "candidate"
    meta = train(generate(12000, seed=3), out, ["xgboost"], ["none"])
    return out, meta


def test_train_writes_bundle(bundle):
    out, meta = bundle
    assert (out / "model.joblib").exists()
    saved = json.loads((out / "meta.json").read_text())
    assert saved["features"] == FEATURES
    assert 0 < saved["threshold"] < 1
    assert meta["test"]["pr_auc"] > 0.5  # far above the ~1.5% base rate


def test_promotion_gate(bundle, tmp_path):
    out, _ = bundle
    champion = tmp_path / "champion"
    fresh = generate(6000, seed=9)
    assert promote(out, champion, fresh) is True  # empty slot -> promoted
    assert promote(out, champion, fresh, min_gain=0.01) is False  # same model must beat champion by min_gain


def test_api_predict_metrics_and_drift(bundle, monkeypatch):
    out, _ = bundle
    monkeypatch.setenv("MODEL_DIR", str(out))
    from fraudops import api

    with TestClient(api.app) as c:
        assert c.get("/health").json()["status"] == "ok"
        one = generate(1, seed=4)[FEATURES].iloc[0].to_dict()
        one = {k: (int(v) if k in ("n_items", "n_voids", "is_cash", "drawer_opens", "hour") else float(v)) for k, v in one.items()}
        r = c.post("/predict", json=one).json()
        assert r["results"][0]["decision"] in ("ok", "fraud")
        rows = generate(600, drift=0.9, seed=5)[FEATURES].to_dict("records")
        rows = [{k: (int(v) if isinstance(v, (np.integer,)) else float(v)) for k, v in x.items()} for x in rows]
        assert c.post("/predict", json={"tickets": rows}).status_code == 200
        assert max(c.get("/drift").json().values()) > 0.2
        m = c.get("/metrics").text
        assert "fraud_requests_total" in m and "fraud_drift_max_psi" in m
        assert c.post("/predict", json={"ticket_total": -1}).status_code == 422
