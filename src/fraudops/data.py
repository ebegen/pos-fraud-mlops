"""Synthetic point-of-sale (POS) transaction logs for fast-food restaurants.

Each row is one closed ticket at a cash register. Fraud is rare (~1-2%) and follows
patterns cashiers actually use:

- void abuse: items are rung up, the customer pays, then lines are voided and the cash kept
- refund fraud: refunds without a matching sale, often right after a shift change
- sweethearting: large unexplained discounts for friends
- no-sale drawer opens around low-value cash tickets

`drift` shifts the input distribution (new menu prices, more card payments, a promo period)
so the monitoring stack has something real to detect.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "ticket_total",
    "n_items",
    "n_voids",
    "void_amount",
    "discount_pct",
    "refund_amount",
    "is_cash",
    "drawer_opens",
    "minutes_since_shift_start",
    "hour",
    "cashier_tenure_days",
    "cashier_void_rate_7d",
]
TARGET = "is_fraud"


def generate(n: int = 50_000, fraud_rate: float = 0.015, drift: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Generate `n` tickets. `drift` in [0, 1] moves prices, payment mix and discount levels."""
    rng = np.random.default_rng(seed)
    hour = rng.choice(np.arange(6, 24), size=n, p=_hour_profile())
    n_items = rng.poisson(3.2, n) + 1
    price_level = 1.0 + 0.35 * drift  # menu price increase
    ticket_total = np.round(n_items * rng.lognormal(np.log(4.5 * price_level), 0.35, n), 2)
    is_cash = (rng.random(n) < (0.38 - 0.25 * drift)).astype(int)  # card payments grow
    discount_pct = np.clip(rng.exponential(2.0 + 8.0 * drift, n) * (rng.random(n) < 0.25 + 0.4 * drift), 0, 60)
    n_voids = rng.binomial(n_items, 0.03)
    void_amount = np.round(n_voids * rng.uniform(1.5, 6.0, n) * price_level, 2)
    refund_amount = np.where(rng.random(n) < 0.01, np.round(rng.uniform(2, 15, n), 2), 0.0)
    drawer_opens = 1 + (rng.random(n) < 0.02).astype(int)
    minutes_since_shift_start = rng.uniform(0, 480, n)
    cashier_tenure_days = rng.gamma(2.0, 120.0, n)
    cashier_void_rate_7d = np.clip(rng.normal(0.03, 0.01, n), 0, 1)

    # legitimate tickets that look suspicious: manager comps, customers changing orders, real refunds
    comp = rng.random(n) < 0.03
    discount_pct[comp] = rng.uniform(25, 70, comp.sum())
    changed = rng.random(n) < 0.04
    n_voids[changed] = rng.integers(1, 4, changed.sum())
    void_amount[changed] = np.round(n_voids[changed] * rng.uniform(2, 8, changed.sum()) * price_level, 2)
    real_refund = rng.random(n) < 0.015
    refund_amount[real_refund] = np.round(rng.uniform(5, 35, real_refund.sum()) * price_level, 2)
    minutes_since_shift_start[real_refund] = rng.uniform(0, 480, real_refund.sum())
    drawer_opens[rng.random(n) < 0.03] = 2

    y = np.zeros(n, dtype=int)
    fraud_idx = rng.choice(n, size=int(n * fraud_rate), replace=False)
    y[fraud_idx] = 1
    kind = rng.choice(4, size=len(fraud_idx), p=[0.4, 0.25, 0.2, 0.15])
    for i, k in zip(fraud_idx, kind):
        if k == 0:  # void abuse
            n_voids[i] = max(n_voids[i], rng.integers(2, 6))
            void_amount[i] = round(n_voids[i] * rng.uniform(3, 9) * price_level, 2)
            is_cash[i] = 1 if rng.random() < 0.85 else 0
            cashier_void_rate_7d[i] = min(1.0, cashier_void_rate_7d[i] + rng.uniform(0.03, 0.12))
        elif k == 1:  # refund fraud
            refund_amount[i] = round(rng.uniform(10, 45) * price_level, 2)
            minutes_since_shift_start[i] = rng.uniform(0, 40)
            is_cash[i] = 1
        elif k == 2:  # sweethearting
            discount_pct[i] = rng.uniform(30, 90)
            cashier_tenure_days[i] = rng.gamma(1.2, 40.0)
        else:  # no-sale drawer opens
            drawer_opens[i] = rng.integers(2, 5)
            is_cash[i] = 1
            ticket_total[i] = round(rng.uniform(1, 6), 2)
        # a third of fraud is subtle: one void, a mid discount, history looks normal
        if rng.random() < 0.35:
            cashier_void_rate_7d[i] = rng.normal(0.03, 0.01)
            if k == 0:
                n_voids[i] = 1
                void_amount[i] = round(rng.uniform(2, 7) * price_level, 2)
            elif k == 2:
                discount_pct[i] = rng.uniform(20, 45)
            elif k == 1:
                minutes_since_shift_start[i] = rng.uniform(0, 480)

    return pd.DataFrame(
        {
            "ticket_total": ticket_total,
            "n_items": n_items,
            "n_voids": n_voids,
            "void_amount": void_amount,
            "discount_pct": np.round(discount_pct, 1),
            "refund_amount": refund_amount,
            "is_cash": is_cash,
            "drawer_opens": drawer_opens,
            "minutes_since_shift_start": np.round(minutes_since_shift_start, 1),
            "hour": hour,
            "cashier_tenure_days": np.round(cashier_tenure_days, 0),
            "cashier_void_rate_7d": np.round(np.clip(cashier_void_rate_7d, 0, 1), 4),
            TARGET: y,
        }
    )


def _hour_profile() -> np.ndarray:
    w = np.array([2, 4, 5, 4, 3, 6, 9, 8, 5, 4, 4, 6, 9, 9, 7, 5, 3, 2], dtype=float)
    return w / w.sum()


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic POS tickets")
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--drift", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/tickets.parquet"))
    a = ap.parse_args()
    df = generate(a.n, drift=a.drift, seed=a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out) if a.out.suffix == ".parquet" else df.to_csv(a.out, index=False)
    print(f"{len(df)} tickets, fraud rate {df[TARGET].mean():.3%} -> {a.out}")


if __name__ == "__main__":
    main()
