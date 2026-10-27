#!/usr/bin/env python3
"""Render the Scion-35B-A3B model-card figures.

Data lives in moe/results/qwen35-retention/35b-card-data.json (the retention
values are copied from retention-grid.json; everything else from the release
card).  Outputs land in docs/figures/.
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
DATA = json.loads((ROOT / "moe/results/qwen35-retention/35b-card-data.json").read_text())
OUT = ROOT / "docs/figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#1f77b4"
LIGHT_BLUE = "#8ecae6"
TEAL = "#17becf"
PURPLE = "#9467bd"
RED = "#d62728"
GREEN = "#2ca02c"
GREY = "#8c8c8c"
LIGHT_GREY = "#c7c7c7"
ORANGE = "#ff7f0e"

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
    m = DATA["model"]
    perf = DATA["performance"]
    sizes = DATA["sizes"]
    colors = {
        "bf16": PURPLE,
        "q4": RED,
        "iq2": GREEN,
        "community": LIGHT_GREY,
        "release": BLUE,
    }

    fig = plt.figure(figsize=(14, 5.8), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.18)

    ax = fig.add_subplot(gs[0, 0])
    labels = [s["label"] for s in sizes]
    vals = [s["gb"] for s in sizes]
    y = np.arange(len(sizes))
    bars = ax.barh(y, vals, color=[colors[s["group"]] for s in sizes],
                   height=0.64, edgecolor="white", linewidth=0.8)
    for i, s in enumerate(sizes):
        if s["group"] == "release":
            bars[i].set_edgecolor("#111111")
            bars[i].set_linewidth(1.4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    for i, s in enumerate(sizes):
        fmt = f"{s['gb']:.2f}" if s["gb"] < 20 else f"{s['gb']:.1f}"
        ax.text(s["gb"] + 1.2, i, f"{fmt} GB", va="center", ha="left",
                fontsize=9.5, fontweight="bold" if s["group"] == "release" else "normal")
    ax.set_xlim(0, 82)
    ax.set_xlabel("File size (GB)")
    ax.set_title("File size vs reference builds", loc="left", pad=8)
    clean(ax, "x")

    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.axis("off")
    tiles = [
        (0.01, 1.01, f"{m['file_gb']:.2f} GB", f"one file, {m['bpw']:.2f} bpw"),
        (1.00, 1.01, "6.3x", "smaller than BF16"),
        (0.01, 0.01, f"{perf['tok_s'][0]:.0f}", "tok/s, one RX 7900 XT"),
        (1.00, 0.01, "88.7%", "GSM8K pass@1, thinking"),
    ]
    for x, y, big, cap in tiles:
        ax.add_patch(Rectangle((x + 0.03, y + 0.03), 0.90, 0.88,
                               facecolor="#f4f7fa", edgecolor="#d5e1ea", linewidth=1.0))
        ax.plot([x + 0.03, x + 0.93], [y + 0.91, y + 0.91], color=BLUE, linewidth=4,
                solid_capstyle="butt")
        ax.text(x + 0.48, y + 0.58, big, ha="center", va="center",
                fontsize=22 if big != "88.7%" else 24, fontweight="bold", color="#14324a")
        ax.text(x + 0.48, y + 0.29, cap, ha="center", va="center",
                fontsize=9.5, color="#555555")

    header(fig,
           "Scion-35B-A3B at a glance",
           "34.9B total, ~3B active  |  ternary expert banks + trained corrections  |  one 11.34 GB file",
           top=0.84)
    source(fig, "Sources: release card; retention grid. Sizes are the published on-disk files of the same model.")
    save(fig, "35b-at-a-glance.png")


def fig_task_retention():
    r = DATA["retention"]
    groups = ["BF16", "Q4_K_M", "IQ2_M", "Q2_K", "Scion"]
    keys = ["bf16", "q4", "iq2", "q2k", "release"]
    hs = [r[k]["hellaswag"] - r["bf16"]["hellaswag"] for k in keys]
    wg = [r[k]["winogrande"] - r["bf16"]["winogrande"] for k in keys]
    x = np.arange(len(groups))
    w = 0.34

    fig, ax = plt.subplots(figsize=(11, 5.2), layout="constrained")
    ax.axhspan(-2, 2, color=GREEN, alpha=0.08, zorder=0)
    ax.axhline(0, color="#333333", linewidth=0.8, zorder=1)
    ax.bar(x - w / 2, hs, width=w, color=BLUE, label="HellaSwag 400", zorder=2)
    ax.bar(x + w / 2, wg, width=w, color=TEAL, label="Winogrande 400", zorder=2)
    for i, (a, b) in enumerate(zip(hs, wg)):
        if i == 0:
            continue
        for j, v in enumerate((a, b)):
            va = "bottom" if v >= 0 else "top"
            off = 0.12 if v >= 0 else -0.12
            ax.text(i + (-w / 2 if j == 0 else w / 2), v + off, f"{v:+.2f}",
                    ha="center", va=va, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["BF16\n71.07 GB", "Q4_K_M\n21.71 GB", "IQ2_M\n12.56 GB",
                        "Q2_K\n13.84 GB", "Scion\n11.34 GB"], fontsize=9.5)
    ax.set_ylim(-5.6, 2.4)
    ax.set_ylabel("Delta vs BF16 (percentage points)")
    ax.set_title("Delta vs BF16", loc="left")
    ax.legend(loc="lower left", frameon=False)
    ax.text(0.99, 0.06, "shaded band = ±2 points", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color="#555555")
    clean(ax, "y")
    header(fig,
           "Task retention: HellaSwag and Winogrande vs BF16",
           "same 400-task protocol for every row; negative is worse  |  Scion: HellaSwag -2.25, Winogrande +0.25",
           top=0.84)
    source(fig, "Source: retention-grid.json / model card. Error bars omitted; the card's 400-task CI is about ±2 points.")
    save(fig, "35b-task-retention.png")


def fig_reasoning_code():
    data = DATA["reasoning"]
    labels = [d["label"] for d in data]
    x = np.arange(len(data))
    w = 0.26
    think = [d["thinking"] for d in data]
    think_err = [d["thinking_err"] for d in data]
    nothink = [d["no_thinking"] for d in data]
    nothink_err = [d["no_thinking_err"] for d in data]
    pass3 = [d["pass3"] for d in data]
    pass3_err = [d["pass3_err"] for d in data]

    fig, ax = plt.subplots(figsize=(12, 5.2), layout="constrained")
    ax.bar(x - w, think, width=w, color=BLUE, label="pass@1, thinking",
           yerr=think_err, capsize=4, error_kw=dict(ecolor="#333333", lw=1.0))
    ax.bar(x, nothink, width=w, color=LIGHT_BLUE, label="pass@1, no thinking",
           yerr=nothink_err, capsize=4, error_kw=dict(ecolor="#333333", lw=1.0))
    ax.bar(x + w, pass3, width=w, color=TEAL, label="pass@3, no thinking",
           yerr=pass3_err, capsize=4, error_kw=dict(ecolor="#333333", lw=1.0))
    for i, (vals, errs) in enumerate(zip(zip(think, nothink, pass3),
                                         zip(think_err, nothink_err, pass3_err))):
        for j, (v, e) in enumerate(zip(vals, errs)):
            ax.text(i + (j - 1) * w, v + e + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(40, 100)
    ax.set_ylabel("Score (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)
    clean(ax, "y")
    header(fig,
           "Reasoning and code: GSM8K, MATH and HumanEval",
           "thinking uses a 2,048-token cap; no-thinking uses 700  |  pass@3 is a different metric; sampled sets, not official full-set rows",
           top=0.80)
    source(fig, "Source: model card, 2026-10-07 run on one RX 7900 XT. Error bars are 1σ binomial.")
    save(fig, "35b-reasoning-code.png")


def fig_additional_tasks():
    data = sorted(DATA["additional"], key=lambda d: d["score"], reverse=True)
    labels = [f"{d['label']} ({d['n']})" for d in data]
    vals = [d["score"] for d in data]
    errs = [d["err"] for d in data]
    y = np.arange(len(data))

    fig, ax = plt.subplots(figsize=(11, 4.8), layout="constrained")
    ax.barh(y, vals, color=BLUE, height=0.58, xerr=errs, capsize=5,
            error_kw=dict(ecolor="#333333", lw=1.0))
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    for i, (v, e) in enumerate(zip(vals, errs)):
        ax.text(v + e + 1.8, i, f"{v:.2f}%", va="center", ha="left", fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Score (%)")
    clean(ax, "x")
    header(fig,
           "Additional tasks: ARC, MMLU and TruthfulQA",
           "same release and fork build, re-measured 2026-09-30 on one RX 7900 XT  |  error bars are 1σ binomial",
           top=0.82)
    source(fig, "Source: release card additional-tasks table. Sample sizes are in the labels.")
    save(fig, "35b-additional-tasks.png")


def fig_drafter():
    d = DATA["drafter"]
    work = d["workloads"]
    labels = [w["label"] for w in work]
    plain = [w["plain"] for w in work]
    draft = [w["draft"] for w in work]
    y = np.arange(len(work))

    fig = plt.figure(figsize=(14, 6.0), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.24)

    ax = fig.add_subplot(gs[0, 0])
    ax.barh(y - 0.18, plain, height=0.34, color=LIGHT_GREY, label="plain")
    ax.barh(y + 0.18, draft, height=0.34, color=BLUE, label="+ MTP drafter")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.invert_yaxis()
    for i, w in enumerate(work):
        ax.text(w["draft"] + 2.5, i + 0.18, f"{w['draft']:.1f} tok/s ({w['gain']}, {w['accept']})",
                va="center", ha="left", fontsize=8.8)
    ax.set_xlim(0, 190)
    ax.set_xlabel("Generation speed (tok/s)")
    ax.set_title("Speed by workload", loc="left")
    clean(ax, "x")

    ax = fig.add_subplot(gs[0, 1])
    pts = d["accept_by_length"]
    x = [p["tokens"] for p in pts]
    yv = [p["accept"] for p in pts]
    ax.plot(x, yv, marker="o", markersize=7, color=BLUE, linewidth=2.0)
    for xx, yy in zip(x, yv):
        ax.annotate(f"{yy}%", (xx, yy), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9.5)
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    ax.minorticks_off()
    ax.set_ylim(60, 90)
    ax.set_xlabel("Generation length (tokens, log scale)")
    ax.set_ylabel("Draft acceptance (%)")
    ax.set_title("Acceptance by generation length", loc="left")
    clean(ax, "y")

    header(fig,
           "MTP drafter: +7-33% depending on the workload",
           "Strata, one RX 7900 XT, greedy  |  grey = plain, blue = + MTP drafter  |  acceptance in the labels; speed option, not bit-exact",
           top=0.84)
    source(fig, "Source: release card Performance section (same card, same day, same protocol).")
    save(fig, "35b-drafter.png")


def fig_decision_head():
    h = DATA["decision_head"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained",
                             gridspec_kw={"width_ratios": [1.0, 1.15]})

    ax = axes[0]
    vals = [h["auc_head"], h["auc_self"]]
    ax.bar([0, 1], vals, width=0.55, color=[BLUE, LIGHT_GREY])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["decision head", "prompted self-judge"])
    ax.set_ylim(0.8, 1.02)
    ax.set_ylabel("AUC")
    ax.set_title("Judge AUC", loc="left")
    clean(ax, "y")

    ax = axes[1]
    caught = h["caught"]
    missed = h["wrong_total"] - caught
    ax.barh([0], [caught], color=BLUE, height=0.45)
    ax.barh([0], [missed], left=[caught], color=LIGHT_GREY, height=0.45)
    ax.text(caught / 2, 0, f"{caught} caught", ha="center", va="center",
            color="white", fontsize=11, fontweight="bold")
    ax.text(caught + missed / 2, 0, f"{missed} missed", ha="center", va="center",
            color="#333333", fontsize=10)
    ax.set_xlim(0, h["wrong_total"])
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.set_xlabel("This model's wrong answers (of 28)")
    ax.set_title("Wrong answers caught", loc="left")
    clean(ax, "x")

    header(fig,
           "Decision head: a 66 KB correctness readout",
           f"ridge + Platt on the model's own last-prompt-token hidden  |  {h['false_rejects']} false rejects on 28 wrong answers; +{h['tasks_over_majority']} tasks over majority",
           top=0.82)
    source(fig, "Source: release card / SkyIsNotGreen/decision-heads. Best-of-3 at about half the pool cost (gating); +7 tasks over majority (veto).")
    save(fig, "35b-decision-head.png")


def main():
    fig_at_a_glance()
    fig_task_retention()
    fig_reasoning_code()
    fig_additional_tasks()
    fig_drafter()
    fig_decision_head()


if __name__ == "__main__":
    main()
