"""Send synthetic traffic to the API. Use --drift to simulate a menu price change / promo period
and watch the drift panels and alerts react in Grafana.

  python scripts/loadgen.py --rps 20 --minutes 5
  python scripts/loadgen.py --rps 20 --minutes 5 --drift 0.8
"""

from __future__ import annotations

import argparse
import time

import httpx

from fraudops.data import FEATURES, generate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--rps", type=float, default=20)
    ap.add_argument("--minutes", type=float, default=5)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--drift", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    n = int(a.rps * a.minutes * 60)
    df = generate(max(n, a.batch), drift=a.drift, seed=a.seed)[FEATURES]
    rows = df.to_dict("records")
    sent = flagged = 0
    t_end = time.time() + a.minutes * 60
    with httpx.Client(base_url=a.url, timeout=10) as c:
        for i in range(0, len(rows), a.batch):
            if time.time() > t_end:
                break
            t0 = time.time()
            r = c.post("/predict", json={"tickets": rows[i : i + a.batch]})
            r.raise_for_status()
            flagged += sum(x["decision"] == "fraud" for x in r.json()["results"])
            sent += a.batch
            time.sleep(max(0.0, a.batch / a.rps - (time.time() - t0)))
            if sent % 1000 == 0:
                print(f"{sent} tickets, {flagged} flagged")
    print(f"done: {sent} tickets, {flagged} flagged ({flagged / max(sent, 1):.2%})")


if __name__ == "__main__":
    main()
