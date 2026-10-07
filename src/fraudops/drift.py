"""Population Stability Index (PSI) drift checks.

Reference bins come from the training data and are saved with the model, so the API and the
Airflow DAG compare live traffic against exactly what the model was trained on.

Rule of thumb: PSI < 0.1 stable · 0.1-0.2 watch · > 0.2 significant shift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-6


def reference_bins(df: pd.DataFrame, features: list[str], bins: int = 10) -> dict:
    ref = {}
    for f in features:
        x = df[f].to_numpy(dtype=float)
        values = np.unique(x)
        if len(values) <= 20:  # binary / small integer feature: one bucket per value
            edges = np.concatenate([[values[0] - 0.5], values + 0.5])
        else:
            edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
            if len(edges) < bins // 2:  # zero-inflated (e.g. refunds): own bucket for the spike, quantiles for the rest
                vals, cnt = np.unique(x, return_counts=True)
                mode = vals[np.argmax(cnt)]
                rest = x[x != mode]
                edges = np.unique(np.concatenate([[x.min(), mode + 1e-9], np.quantile(rest, np.linspace(0, 1, bins)), [x.max()]]))
        counts, _ = np.histogram(x, bins=edges)
        ref[f] = {"edges": edges.tolist(), "dist": (counts / max(counts.sum(), 1)).tolist()}
    return ref


def psi(ref: dict, x: np.ndarray) -> float:
    edges = np.asarray(ref["edges"], dtype=float)
    expected = np.asarray(ref["dist"], dtype=float)
    clipped = np.clip(x, edges[0], edges[-1])
    counts, _ = np.histogram(clipped, bins=edges)
    actual = counts / max(counts.sum(), 1)
    e = np.clip(expected, EPS, None)
    a = np.clip(actual, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(ref: dict, df: pd.DataFrame) -> dict[str, float]:
    return {f: psi(r, df[f].to_numpy(dtype=float)) for f, r in ref.items() if f in df}
