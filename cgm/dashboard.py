"""Interactive HTML dashboard for design teams (docs/dashboard.html) + static figures for the README.

  python -m cgm.dashboard
"""
from __future__ import annotations

import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from lifelines import KaplanMeierFitter  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from . import analysis as A  # noqa: E402
from .sim import simulate  # noqa: E402

warnings.filterwarnings("ignore")
BLUE, ORANGE, RED, GREEN, GREY = "#2563eb", "#ea580c", "#dc2626", "#16a34a", "#6b7280"


def km_curve(g, cause_mask):
    k = KaplanMeierFitter().fit(g["days_worn"], cause_mask)
    s = k.survival_function_
    return s.index.to_numpy(), s.iloc[:, 0].to_numpy()


def main():
    d = simulate()
    r = json.load(open("results/results.json"))
    p = d["clinical"]["pairs"]
    rw = d["real_world"].merge(d["lots"], on="lot")
    early = rw["end_reason"].isin(["sensor_failure", "detached"])
    L = pd.DataFrame(r["bench_to_body"]["table"])

    # ---------------- static figures for the README
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    days = sorted(p["day"].unique())
    for col, c, lab in (("cgm_v1", BLUE, "firmware v1"), ("cgm_v2", ORANGE, "firmware v2")):
        ax[0].plot(days, [A.mard(col)(p[p["day"] == x]) * 100 for x in days], "o-", color=c, label=lab)
    ax[0].set_title("MARD by day of wear (%)", fontsize=10)
    ax[0].set_xlabel("day")
    ax[0].legend(fontsize=8, frameon=False)
    for name, g, c in (("ADH-7 adhesive", rw[rw["adhesive"] == "ADH-7"], RED), ("other adhesives", rw[rw["adhesive"] != "ADH-7"], BLUE),
                       ("membrane < 9.4 um", rw[rw["membrane_um"] < 9.4], ORANGE)):
        x, y = km_curve(g, early[g.index])
        ax[1].step(x, y, where="post", color=c, label=name)
    ax[1].set_title("Sensors still working (failure or detachment)", fontsize=10)
    ax[1].set_xlabel("days worn")
    ax[1].set_ylim(0, 1.02)
    ax[1].legend(fontsize=8, frameon=False)
    clin = L[L["bias"].notna()]
    for line, c in (("Line-1", BLUE), ("Line-2", GREEN)):
        g = clin[clin["line"] == line]
        ax[2].scatter(g["bench"] * 100, g["bias"] * 100, color=c, s=18, label=line)
    ax[2].axhspan(-5, 5, color=GREY, alpha=0.1)
    ax[2].set_xlabel("bench sensitivity vs. typical lot (%)")
    ax[2].set_ylabel("in-body bias vs. typical lot (%)")
    ax[2].set_title("Bench predicts in-body bias (36 lots)", fontsize=10)
    ax[2].legend(fontsize=8, frameon=False)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("results/overview.png", dpi=150)

    # ---------------- interactive dashboard
    acc, fw, rel, cmp = r["accuracy_v1"], r["firmware"], r["reliability"], r["compression"]
    tiles = [("MARD (v1)", f"{acc['mard'][0]:.1%}", f"95% CI {acc['mard'][1]:.1%}-{acc['mard'][2]:.1%}"),
             ("Within 20/20", f"{acc['within_20_20'][0]:.1%}", "of 172,800 reference pairs"),
             ("Clarke zone A+B", f"{acc['clarke']['A'] + acc['clarke']['B']:.1%}", f"zone D {acc['clarke']['D']:.2%}"),
             ("Full 15-day wear", f"{rel['full_wear_rate']:.1%}", "of 6,000 real-world sessions"),
             ("Compression false alarms removed", f"{cmp['model']['compression_alarms_suppressed']:.0%}",
              f"true lows silenced: {cmp['model']['true_lows_suppressed']} of {cmp['test_true_lows']}")]
    f1 = make_subplots(rows=1, cols=2, subplot_titles=("MARD by day of wear", "Error by glucose range (MAD, mg/dL)"))
    for col, c, lab in (("cgm_v1", BLUE, "v1"), ("cgm_v2", ORANGE, "v2")):
        f1.add_trace(go.Scatter(x=days, y=[A.mard(col)(p[p["day"] == x]) * 100 for x in days], name=f"firmware {lab}", line=dict(color=c)), 1, 1)
        rngs = pd.cut(p["reference"], [0, 70, 180, 1000], labels=["<70", "70-180", ">180"])
        f1.add_trace(go.Bar(x=["<70", "70-180", ">180"], y=[(p.loc[rngs == k, col] - p.loc[rngs == k, "reference"]).abs().mean() for k in ["<70", "70-180", ">180"]],
                            name=f"firmware {lab}", marker_color=c, showlegend=False), 1, 2)
    f1.update_layout(height=360, margin=dict(t=40, b=30), template="plotly_white")
    samp = p.sample(4000, random_state=0)
    z = A.clarke_zone(samp["reference"], samp["cgm_v1"])
    f2 = go.Figure()
    for zone, c in (("A", GREEN), ("B", ORANGE), ("C", RED), ("D", RED), ("E", RED)):
        m = z == zone
        if m.any():
            f2.add_trace(go.Scattergl(x=samp["reference"][m], y=samp["cgm_v1"][m], mode="markers", name=f"zone {zone}",
                                      marker=dict(size=3, color=c, opacity=0.5)))
    f2.update_layout(title="Clarke error grid (4,000 sampled pairs)", xaxis_title="reference (mg/dL)", yaxis_title="CGM (mg/dL)",
                     height=420, template="plotly_white")
    f3 = go.Figure()
    for name, g, c in (("ADH-7 adhesive", rw[rw["adhesive"] == "ADH-7"], RED), ("other adhesives", rw[rw["adhesive"] != "ADH-7"], BLUE),
                       ("membrane < 9.4 um", rw[rw["membrane_um"] < 9.4], ORANGE), ("membrane >= 9.4 um", rw[rw["membrane_um"] >= 9.4], GREEN)):
        x, y = km_curve(g, early[g.index])
        f3.add_trace(go.Scatter(x=x, y=y, mode="lines", line_shape="hv", name=name, line=dict(color=c)))
    f3.update_layout(title="Kaplan-Meier: sensors still working", xaxis_title="days worn", yaxis_title="share", height=380, template="plotly_white")
    f4 = go.Figure()
    for line, c in (("Line-1", BLUE), ("Line-2", GREEN)):
        g = L[L["line"] == line]
        f4.add_trace(go.Scatter(x=g["bench"] * 100, y=g["pred"] * 100, mode="markers", name=f"{line} (predicted)",
                                marker=dict(color=c, size=9, symbol=np.where(g["hold_model"], "x", "circle")),
                                text=[f"{k}: membrane {m} um, true dev {t:+.1%}, P(out)={q:.2f}" for k, m, t, q in zip(g["lot"], g["membrane_um"], g["true_dev"], g["p_out"])]))
    f4.add_hrect(y0=-5, y1=5, fillcolor=GREY, opacity=0.1, line_width=0)
    f4.update_layout(title="Lot release: predicted in-body bias from the bench test (x = held)", xaxis_title="bench vs. typical lot (%)",
                     yaxis_title="predicted in-body bias (%)", height=400, template="plotly_white")
    from plotly.offline import get_plotlyjs
    plotly_js = get_plotlyjs()                                       # inline: the page works offline
    tile_html = "".join(f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{s}</div></div>' for k, v, s in tiles)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CGM Performance Dashboard</title>
<script>{plotly_js}</script>
<style>body{{font-family:system-ui,Segoe UI,sans-serif;margin:24px;color:#111827;background:#fff}}h1{{font-size:20px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:12px 0 20px}}
.tile{{border:1px solid #e5e7eb;border-radius:8px;padding:12px}}.k{{font-size:12px;color:#6b7280}}.v{{font-size:24px;font-weight:600}}
.s{{font-size:11px;color:#6b7280}}.note{{font-size:12px;color:#6b7280}}</style></head><body>
<h1>CGM sensor performance &amp; reliability (simulated program)</h1>
<p class="note">Clinical study: 720 participants, 36 lots, lab reference every 15 min on 5 clinic days. Real world: 6,000 sessions, 48 lots.
All data simulated with known truth; see README.</p><div class="tiles">{tile_html}</div>
{f1.to_html(full_html=False, include_plotlyjs=False)}{f2.to_html(full_html=False, include_plotlyjs=False)}
{f3.to_html(full_html=False, include_plotlyjs=False)}{f4.to_html(full_html=False, include_plotlyjs=False)}</body></html>"""
    open("docs/dashboard.html", "w").write(html)
    print("docs/dashboard.html results/overview.png")


if __name__ == "__main__":
    main()
