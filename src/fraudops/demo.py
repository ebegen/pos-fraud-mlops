"""Gradio demo: score a ticket through the live API and see the model card.

  API_URL=http://localhost:8000 python -m fraudops.demo
"""

from __future__ import annotations

import os

import gradio as gr
import httpx

API = os.environ.get("API_URL", "http://localhost:8000")

PRESETS = {
    "Normal lunch ticket": [18.4, 4, 0, 0.0, 0.0, 0.0, 0, 1, 210, 12, 400, 0.03],
    "Void abuse (cash, 4 voids)": [9.9, 5, 4, 26.5, 0.0, 0.0, 1, 1, 300, 19, 90, 0.11],
    "Refund right after shift start": [6.2, 1, 0, 0.0, 0.0, 32.0, 1, 1, 12, 7, 200, 0.03],
    "Sweetheart discount": [24.0, 5, 0, 0.0, 75.0, 0.0, 0, 1, 260, 21, 35, 0.03],
}
FIELDS = [
    ("ticket_total", "Ticket total ($)"),
    ("n_items", "Items"),
    ("n_voids", "Voided lines"),
    ("void_amount", "Voided amount ($)"),
    ("discount_pct", "Discount %"),
    ("refund_amount", "Refund ($)"),
    ("is_cash", "Cash (1) / card (0)"),
    ("drawer_opens", "Drawer opens"),
    ("minutes_since_shift_start", "Minutes since shift start"),
    ("hour", "Hour"),
    ("cashier_tenure_days", "Cashier tenure (days)"),
    ("cashier_void_rate_7d", "Cashier void rate, 7 days"),
]


def score(*vals):
    ticket = {k: v for (k, _), v in zip(FIELDS, vals)}
    for k in ("n_items", "n_voids", "is_cash", "drawer_opens", "hour"):
        ticket[k] = int(ticket[k])
    try:
        r = httpx.post(f"{API}/predict", json=ticket, timeout=10).json()
    except httpx.HTTPError as e:
        return f"API not reachable: {e}", ""
    res = r["results"][0]
    label = "🚩 FRAUD" if res["decision"] == "fraud" else "✅ OK"
    return f"## {label} — fraud probability {res['fraud_probability']:.1%}", f"scored by model `{r['model_version']}`"


def card():
    try:
        m = httpx.get(f"{API}/model", timeout=10).json()
    except httpx.HTTPError as e:
        return f"API not reachable: {e}"
    rows = "\n".join(f"| {x['model']} | {x['sampler']} | {x['val_pr_auc']} |" for x in m["leaderboard"])
    t = m["test"]
    return (
        f"**{m['model']} + {m['sampler']}** · version `{m['version']}` · threshold {m['threshold']:.3f}\n\n"
        f"Test: PR-AUC {t['pr_auc']:.3f} · precision {t['precision']:.3f} · recall {t['recall']:.3f} · F1 {t['f1']:.3f}\n\n"
        f"| model | resampling | val PR-AUC |\n|---|---|---|\n{rows}"
    )


with gr.Blocks(title="POS fraud scoring") as ui:
    gr.Markdown("# POS fraud scoring\nScores one cash-register ticket through the deployed model API.")
    with gr.Tab("Score a ticket"):
        with gr.Row():
            preset = gr.Dropdown(list(PRESETS), label="Example", value="Normal lunch ticket", scale=3)
            btn = gr.Button("Score", variant="primary", scale=1)
        out = gr.Markdown()
        ver = gr.Markdown()
        inputs = []
        for i in range(0, len(FIELDS), 4):
            with gr.Row():
                inputs += [gr.Number(label=lbl, value=v) for (_, lbl), v in zip(FIELDS[i : i + 4], PRESETS["Normal lunch ticket"][i : i + 4])]
        preset.change(lambda p: PRESETS[p], preset, inputs)
        btn.click(score, inputs, [out, ver])
    with gr.Tab("Model card"):
        md = gr.Markdown()
        gr.Button("Refresh").click(card, None, md)
        ui.load(card, None, md)

if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
