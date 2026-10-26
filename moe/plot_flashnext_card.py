#!/usr/bin/env python3
"""Render the Scion-FlashNext model-card figures.

Data lives in moe/results/qwen4exp/flashnext-card-data.json; every number in
that file is copied from the release card or the tensor inventory named in its
"sources" block.  Outputs land in docs/figures/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "moe/results/qwen4exp/flashnext-card-data.json").read_text())
OUT = ROOT / "docs/figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#1f77b4"
LIGHT_BLUE = "#8ecae6"
TEAL = "#17becf"
DARK = "#264653"
GREY = "#8c8c8c"
LIGHT_GREY = "#c7c7c7"
ORANGE = "#ff7f0e"
RED = "#d62728"
GREEN = "#2ca02c"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.12,
    "grid.linewidth": 0.6,
})


def clean(ax, axis="x"):
    ax.grid(axis=axis, color="#000000", alpha=0.10, linewidth=0.6)
    ax.set_axisbelow(True)


def header(fig, title, subtitle, top=0.86):
    fig.suptitle(title, x=0.01, y=0.985, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.01, 0.935, subtitle, ha="left", va="top", fontsize=10, color="#555555")
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.set(rect=(0.0, 0.06, 1.0, top - 0.06))


def source(fig, text):
    fig.text(0.5, 0.005, text, ha="center", va="bottom", fontsize=8, color="#666666")


def save(fig, name):
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(path.relative_to(ROOT))


def fig_at_a_glance():
    comp = DATA["composition"]
    sizes = DATA["sizes"]
    perf = DATA["performance"]
    model = DATA["model"]

    fig = plt.figure(figsize=(14, 6.6), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.18], wspace=0.42)

    # --- composition -------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(comp))
    vals = [c["gb"] for c in comp]
    colors = [c["color"] for c in comp]
    ax.barh(y, vals, color=colors, height=0.62, edgecolor="white", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([c["label"] for c in comp], fontsize=9.5)
    ax.invert_yaxis()
    for i, c in enumerate(comp):
        label = f"{c['gb']:.2f} GB" if c["gb"] >= 0.1 else f"{c['gb']:.3f} GB"
        pct = "<0.1%" if c["pct"] < 0.1 else f"{c['pct']:.1f}%"
        ax.text(c["gb"] + 0.35, i, f"{label}  ({pct})",
                va="center", ha="left", fontsize=9, color="#222222")
    ax.set_xlim(0, 33)
    ax.set_xlabel("GB in the default file")
    ax.set_title("Where the 57.62 GB goes", loc="left", pad=8)
    clean(ax, "x")

    # --- size ladder -------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    labels = [s["label"] for s in sizes]
    vals = [s["gb"] for s in sizes]
    groups = [s["group"] for s in sizes]
    cmap = {
        "fp8": DARK,
        "community": LIGHT_GREY,
        "scion": LIGHT_BLUE,
        "pruned": ORANGE,
        "release": BLUE,
    }
    colors = [cmap[g] for g in groups]
    y = np.arange(len(sizes))
    bars = ax.barh(y, vals, color=colors, height=0.66, edgecolor="white", linewidth=0.7)
    for i, s in enumerate(sizes):
        if s["group"] == "release":
            bars[i].set_edgecolor("#111111")
            bars[i].set_linewidth(1.4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.invert_yaxis()
    for i, s in enumerate(sizes):
        fmt = f"{s['gb']:.1f}" if s["gb"] >= 100 else f"{s['gb']:.2f}"
        ax.text(s["gb"] + 3, i, f"{fmt} GB", va="center", ha="left",
                fontsize=9, color="#222222",
                fontweight="bold" if s["group"] == "release" else "normal")
    ax.set_xlim(0, 215)
    ax.set_xlabel("Published file size (GB)")
    ax.set_title("File size vs published builds", loc="left", pad=8)
    clean(ax, "x")

    header(fig,
           "Scion-FlashNext-176B-A6B at a glance",
           f"{model['total_params_b']:.1f}B params: 125B language model + 51.2B n-gram table  |  "
           f"57.62 GB, 3.2x smaller than FP8  |  {perf['tok_s'][0]:.1f}-{perf['tok_s'][1]:.1f} tok/s on 2x RX 7900 XT",
           top=0.84)
    source(fig, "Sources: model card; tensor inventory (release-header.json). Sizes as published on the Hub, checked 2026-10-14.")
    save(fig, "flashnext-at-a-glance.png")


def fig_performance():
    p = DATA["performance"]
    fig = plt.figure(figsize=(12, 3.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    tiles = [
        (f"{p['tok_s'][0]:.1f}-{p['tok_s'][1]:.1f}", "tok/s", "generation"),
        (f"{p['ms_per_token']:.1f} ms", "", "per generated token"),
        (f"{p['tokens_per_window'][0]:.1f}-{p['tokens_per_window'][1]:.1f}", "", "tokens per pipeline window"),
    ]
    positions = [0.02, 0.35, 0.68]
    for (big, unit, caption), x in zip(tiles, positions):
        ax.add_patch(Rectangle((x, 0.18), 0.30, 0.64, facecolor="#f4f7fa",
                               edgecolor="#d5e1ea", linewidth=1.0, transform=ax.transAxes))
        ax.plot([x, x + 0.30], [0.82, 0.82], color=BLUE, linewidth=4,
                solid_capstyle="butt", transform=ax.transAxes)
        ax.text(x + 0.15, 0.56, big, ha="center", va="center",
                fontsize=25 if big != "80.5-80.8" else 24, fontweight="bold", color="#14324a",
                transform=ax.transAxes)
        if unit:
            ax.text(x + 0.15, 0.41, unit, ha="center", va="center",
                    fontsize=12, color="#14324a", transform=ax.transAxes)
        ax.text(x + 0.15, 0.27, caption, ha="center", va="center",
                fontsize=10, color="#555555", transform=ax.transAxes)

    fig.text(0.5, 0.94, "Measured on 2x RX 7900 XT 20 GB", ha="center", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.865, "32 GB host RAM  |  4,096 context  |  greedy  |  60.17 GB Q4_0-table file  |  Strata",
             ha="center", fontsize=10, color="#555555")
    fig.text(0.5, 0.055, "pipeline windows 2: +62-68% over the same engine without it on this workload",
             ha="center", fontsize=9, color="#666666")
    save(fig, "flashnext-performance.png")


def fig_quality_ab():
    q = DATA["quality_ab"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), layout="constrained")
    labels = ["Scion\n60.17 GB", "ISTA Q2_0\n66.4 GB"]
    x = np.arange(2)

    def panel(ax, vals, errs, ylim, ylabel, title, note, note_color):
        ax.plot(x, vals, color="#cccccc", lw=1.2, zorder=1)
        for i, (v, e) in enumerate(zip(vals, errs)):
            ax.errorbar([i], [v], yerr=[e] if e is not None else None,
                        fmt="o", ms=10, color=[BLUE, GREY][i], capsize=6,
                        elinewidth=1.2, markeredgecolor="white", markeredgewidth=0.8,
                        zorder=3)
            fmt = f"{v:.3f}" if isinstance(v, float) and v < 10 else f"{v:.2f}"
            ax.annotate(fmt, (i, v), textcoords="offset points", xytext=(0, 11),
                        ha="center", va="bottom", fontsize=10)
        ax.set_xlim(-0.45, 1.45)
        ax.set_ylim(*ylim)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.text(0.5, 0.05, note, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=10, color=note_color, fontweight="bold")
        clean(ax, "y")

    panel(axes[0], [q["scion"]["ppl"], q["ista"]["ppl"]],
          [q["scion"]["ppl_err"], q["ista"]["ppl_err"]], (0, 6.0),
          "PPL (lower is better)", "Wikitext-2 PPL",
          f"+{q['ppl_delta_pct']:.1f}% PPL (Scion worse)", RED)

    panel(axes[1], [q["scion"]["hellaswag"], q["ista"]["hellaswag"]],
          [None, None], (0, 110),
          "HellaSwag % (higher is better)", "HellaSwag 400",
          f"+{q['hellaswag_delta']:.2f} pts, paired p = {q['hellaswag_p']:.2f} (tie)", "#555555")

    panel(axes[2], [q["scion"]["winogrande"], q["ista"]["winogrande"]],
          [q["scion"]["winogrande_err"], q["ista"]["winogrande_err"]], (0, 100),
          "Winogrande % (higher is better)", "Winogrande 400",
          f"+{q['winogrande_delta']:.2f} pts, p = {q['winogrande_p']:.2f} (not conclusive)", BLUE)

    header(fig,
           "Scion vs ISTA GSQ-RCO Q2_0: same harness, same 400 tasks",
           "blue = Scion, grey = ISTA  |  whiskers = published CI (HellaSwag has no published CI)  |  axes start at 0",
           top=0.84)
    source(fig, "Source: model card community benchmark (2026-10-06 A100 session). PPL CI ±0.032/±0.033; Winogrande CI ±2.09/±2.18; HS/WG p-values are paired McNemar.")
    save(fig, "flashnext-quality-ab.png")


def fig_kld_gate():
    k = DATA["kld_gate"]
    metrics = ["mean", "p50", "p90", "p99", "p999", "max"]
    names = ["mean", "p50", "p90", "p99", "p99.9", "max"]
    rel = [k["released"][m] for m in metrics]
    ext = [k["step4000"][m] for m in metrics]

    fig = plt.figure(figsize=(14, 5.8), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.12)

    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(metrics))
    h = 0.34
    ax.barh(y - h / 2, rel, height=h, color=BLUE, label="step-3000 (released)")
    ax.barh(y + h / 2, ext, height=h, color=LIGHT_GREY, label="step-4000 (not shipped)")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(0.18, 22)
    for i, (a, b) in enumerate(zip(rel, ext)):
        ax.text(a * 1.06, i - h / 2, f"{a:g}", va="center", ha="left", fontsize=9)
        ax.text(b * 1.06, i + h / 2, f"{b:g}", va="center", ha="left", fontsize=9, color="#555555")
    ax.set_xlabel("KLD vs BF16 teacher (log scale, lower is better)")
    ax.set_title("Quantiles", loc="left")
    ax.legend(loc="upper right", frameon=False)
    clean(ax, "x")

    ax = fig.add_subplot(gs[0, 1])
    split = k["split"]
    vals = [s["pct"] for s in split]
    labels = [f"{s['label']} {s['pct']:.1f}% ({s['value']:.4f})" for s in split]
    wedges, _ = ax.pie(vals, colors=[s["color"] for s in split], startangle=90,
                       counterclock=False, wedgeprops=dict(width=0.36, edgecolor="white", linewidth=1.5))
    ax.text(0, 0, "mean\n0.5800", ha="center", va="center", fontsize=14, fontweight="bold", color="#14324a")
    ax.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.20), ncol=1, frameon=False)
    ax.set_title("Where the KLD mass sits", loc="left")

    header(fig,
           "KLD gate: released step-3000 vs the step-4000 extension",
           "48 layers, 4,088 tokens  |  mean 0.5800 vs 0.5880  |  top-1 agreement 0.7236 vs 0.7231",
           top=0.84)
    source(fig, "Sources: qwen4exp-eval-kld-48l-step3000.json; qwen4exp-eval-kld-48l-step4000.json. 8 wikitext windows x 512 tokens.")
    save(fig, "flashnext-kld-gate.png")


def fig_664():
    t = DATA["task664"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), layout="constrained")

    # headline vs completed
    ax = axes[0]
    groups = ["Gen\ncap 700", "Think-harder\ncap 2048"]
    x = np.arange(2)
    w = 0.34
    headline = [t["gen"]["pass1"], t["think_hard"]["pass1"]]
    finished = [t["gen"]["finished_acc"], t["think_hard"]["finished_acc"]]
    ax.bar(x - w / 2, headline, width=w, color=LIGHT_BLUE, label="headline (floor)")
    ax.bar(x + w / 2, finished, width=w, color=BLUE, label="completed inside the cap")
    for i, (a, b) in enumerate(zip(headline, finished)):
        ax.text(i - w / 2, a + 1.2, f"{a:.1f}%", ha="center", va="bottom", fontsize=9.5)
        ax.text(i + w / 2, b + 1.2, f"{b:.1f}%", ha="center", va="bottom", fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Headline vs records that completed inside the cap", loc="left")
    ax.legend(loc="upper left", frameon=False)
    clean(ax, "y")

    # truncation
    ax = axes[1]
    cap = [t["gen"]["cap_pct"], t["think_hard"]["cap_pct"]]
    noans = [t["gen"]["no_answer_pct"], t["think_hard"]["no_answer_pct"]]
    ax.bar(x - w / 2, cap, width=w, color=GREY, label="hit the token cap")
    ax.bar(x + w / 2, noans, width=w, color=RED, label="no answer at all")
    for i, (a, b) in enumerate(zip(cap, noans)):
        ax.text(i - w / 2, a + 0.7, f"{a:.0f}%", ha="center", va="bottom", fontsize=9.5)
        ax.text(i + w / 2, b + 0.7, f"{b:.0f}%", ha="center", va="bottom", fontsize=9.5)
    ax.text(0, 18.5, "0/324 no answer", ha="center", fontsize=8.5, color="#555555")
    ax.text(1, 29.5, "175/178 no answer", ha="center", fontsize=8.5, color="#555555")
    ax.set_xticks(x)
    ax.set_xticklabels(["Gen\n1,992 records", "Think-harder\n664 tasks"])
    ax.set_ylim(0, 33)
    ax.set_ylabel("% of records")
    ax.set_title("Why the headline is low", loc="left")
    ax.legend(loc="upper left", frameon=False)
    clean(ax, "y")

    header(fig,
           "664-task reasoning/code run: a floor, not a capability number",
           f"gen pass@1 {t['gen']['pass1']:.1f}% / pass@3 {t['gen']['pass3']:.1f}%  |  "
           f"think-harder {t['think_hard']['pass1']:.1f}%  |  majority arm {t['majority']:.1f}%",
           top=0.84)
    source(fig, "Source: model card decision-arm benchmark (2026-10-09). 'Completed' means did not hit the token cap; no-answer records are counted in the headline.")
    save(fig, "flashnext-664-truncation.png")


def fig_q4x160():
    q = DATA["q4x160"]
    fig = plt.figure(figsize=(14, 6.4), layout="constrained")
    gs = fig.add_gridspec(2, 4, height_ratios=[0.75, 1.0], hspace=0.45, wspace=0.32)

    # file sizes
    ax = fig.add_subplot(gs[0, :])
    labels = ["Q4_0 table\n(release)", "Q4X160\n(shipped, -2.55 GB)", "Q3X32\n(fallback, -6.41 GB, not shipped)"]
    vals = [q["q4_0"]["file_gb"], q["q4x160"]["file_gb"], q["q3x32"]["file_gb"]]
    colors = [LIGHT_GREY, BLUE, ORANGE]
    x = np.arange(3)
    bars = ax.bar(x, vals, width=0.52, color=colors, edgecolor="white", linewidth=1.0)
    bars[1].set_edgecolor("#111111")
    bars[1].set_linewidth(1.4)
    for i, v in enumerate(vals):
        ax.text(i, v - 1.3, f"{v:.2f} GB", ha="center", va="top", fontsize=10,
                fontweight="bold" if i == 1 else "normal", color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(50, 62)
    ax.set_ylabel("File size (GB)")
    ax.set_title("PLE table container: file size", loc="left")
    clean(ax, "y")

    # deltas
    panels = [
        ("ppl_pct", "PPL delta (%)\nlower is better", (0, 1.6)),
        ("kld", "KLD mean delta\nlower is better", (-0.002, 0.012)),
        ("hs", "HellaSwag delta\nhigher is better", (-0.4, 0.2)),
        ("wg", "Winogrande delta\nhigher is better", (-0.4, 0.4)),
    ]
    for i, (key, title, ylim) in enumerate(panels):
        ax = fig.add_subplot(gs[1, i])
        vals = [q["q4x160"][key], q["q3x32"][key]]
        x = np.arange(2)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.bar(x, vals, width=0.55, color=[BLUE, ORANGE])
        for j, v in enumerate(vals):
            fmt = f"{v:+.4f}" if key == "kld" else f"{v:+.2f}"
            va = "bottom" if v >= 0 else "top"
            offset = (ylim[1] - ylim[0]) * 0.04
            ax.text(j, v + (offset if v >= 0 else -offset), fmt,
                    ha="center", va=va, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(["Q4X160", "Q3X32"], fontsize=9)
        ax.set_ylim(*ylim)
        ax.set_title(title, loc="left", fontsize=10.5)
        clean(ax, "y")

    header(fig,
           "Q4X160 keeps the table small without paying for it",
           "experts untouched; only the 51.2B-parameter table changes  |  blue = Q4X160 (shipped); orange = Q3X32 (fallback, not shipped)",
           top=0.84)
    source(fig, "Sources: Q4X160 gate (2026-10-12, paired, 48 layers); deltas are against the Q4_0-table release.")
    save(fig, "flashnext-q4x160-trade.png")


def fig_capability():
    c = DATA["capability"]
    fig, ax = plt.subplots(figsize=(12, 4.8), layout="constrained")
    cats = [x["label"] for x in c["categories"]]
    x = np.arange(len(cats))
    w = 0.34
    q6 = c["q6k"]
    f16 = c["f16"]
    ax.bar(x - w / 2, q6, width=w, color=BLUE, label=f"Q6_K body ({c['q6k_total']})")
    ax.bar(x + w / 2, f16, width=w, color=LIGHT_GREY, label=f"F16 body ({c['f16_total']})")
    for i, (a, b) in enumerate(zip(q6, f16)):
        maxv = c["categories"][i]["max"]
        ax.text(i - w / 2, a + 0.08, f"{a}/{maxv}", ha="center", va="bottom", fontsize=9.5)
        ax.text(i + w / 2, b + 0.08, f"{b}/{maxv}", ha="center", va="bottom", fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylim(0, 5.8)
    ax.set_ylabel("Tasks passed")
    ax.legend(loc="upper right", frameon=False)
    clean(ax, "y")
    header(fig,
           "Capability screen: 20 tasks, 0 harness errors",
           "Q6_K default vs F16 fidelity body  |  code and logic are the weak categories at this checkpoint",
           top=0.82)
    source(fig, "Source: model card capability table.")
    save(fig, "flashnext-capability.png")


def main():
    fig_at_a_glance()
    fig_performance()
    fig_quality_ab()
    fig_kld_gate()
    fig_664()
    fig_q4x160()
    fig_capability()


if __name__ == "__main__":
    main()
