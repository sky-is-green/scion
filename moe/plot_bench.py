#!/usr/bin/env python3
"""Render the 35B quant-retention grid in the community (Unsloth-style) layout.

x = file size (GB), y = metric with "(lower/higher is better)" in the axis label.
Every point is labelled with a leader line; the Pareto frontier is drawn.
Ours = blue triangle; Q4_K_M = red diamond; IQ2_M = green circle;
BF16 = purple square; other quants = grey circles.

The three "lower is better" panels (PPL, KLD mean, KLD 99.9%) are drawn with
an inverted y-axis so that "up = better" in every panel (set ORIENTATION=standard
for the conventional orientation).

Labels are placed deterministically from hand-tuned anchors (no adjustText):
each anchor keeps its label clear of every marker, the Pareto line, the other
labels and the axes frame, and matplotlib's annotate() draws the leader line,
so both ends of every pointer stay attached.  audit_panel() re-checks the
result and prints a warning for any collision (it should stay silent).
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_bench import ORDER, SIZES_GB  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results" / "qwen35-retention"
data = json.loads((RESULTS / "retention-grid.json").read_text())

panels = [
    ("ppl",       "PPL — wikitext-2, c512, 580 chunks", "PPL (lower is better)", False, False),
    ("kld_mean",  "KLD mean vs BF16 — 50 chunks",        "KLD mean (lower is better)", True, False),
    ("kld_999",   "KLD 99.9% vs BF16 — 50 chunks",       "KLD 99.9% (lower is better)", True, False),
    ("hellaswag", "HellaSwag 400 — zero-shot acc_norm",  "HellaSwag % (higher is better)", False, True),
]

blue, red, green, purple, grey = "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#999999"
STYLE = {
    "release": ("^", 160, blue),
    "Q4_K_M":  ("D", 90,  red),
    "IQ2_M":   ("o", 80,  green),
    "BF16":    ("s", 90,  purple),
}
DEFAULT = ("o", 45, grey)

# orientation of the three "lower is better" panels (ppl / kld_mean / kld_999):
#   "up" (default) — y-axis inverted, so up = better in every panel
#   "standard"     — conventional orientation, lower values at the bottom
ORIENTATION = os.environ.get("ORIENTATION", "up")
FLIPPED = ORIENTATION != "standard"

# ---------------------------------------------------------------------------
# label anchors: panel -> tag -> (x, y, ha, va) in data coordinates.
# Hand-tuned on the 2026-09-27 data so that no label sits on a marker, the
# Pareto line, another label or the axes frame.
# ---------------------------------------------------------------------------
ANCHORS = {
    "ppl": {
        "release": (13.6, 8.25, "left", "center"),
        "IQ2_M":   (14.2, 8.40, "left", "center"),
        "Q2_K":    (15.0, 8.60, "left", "center"),
        "IQ3_M":   (14.5, 7.52, "right", "center"),
        "Q3_K_M":  (19.5, 7.56, "left", "center"),
        "IQ4_XS":  (21.6, 7.13, "left", "center"),
        "Q4_K_M":  (20.6, 7.15, "right", "center"),
        "Q5_K_M":  (27.2, 7.44, "left", "center"),
        "Q6_K":    (31.0, 7.15, "left", "center"),
        "Q8_0":    (39.8, 7.16, "left", "center"),
        "BF16":    (69.0, 7.16, "right", "center"),
    },
    "kld_mean": {
        "release": (13.0, 0.350, "left", "center"),
        "IQ2_M":   (14.9, 0.218, "left", "center"),
        "Q2_K":    (16.4, 0.130, "left", "center"),
        "IQ3_M":   (14.9, 0.0567, "right", "center"),
        "Q3_K_M":  (19.8, 0.070, "left", "center"),
        "IQ4_XS":  (17.9, 0.020, "right", "center"),
        "Q4_K_M":  (23.6, 0.0314, "left", "center"),
        "Q5_K_M":  (27.3, 0.0185, "left", "center"),
        "Q6_K":    (31.0, 0.0082, "left", "center"),
        "Q8_0":    (39.6, 0.0043, "left", "center"),
    },
    "kld_999": {
        "release": (13.2, 7.25, "left", "center"),
        "IQ2_M":   (13.8, 4.70, "left", "center"),
        "Q2_K":    (15.2, 2.95, "left", "center"),
        "IQ3_M":   (14.9, 1.225, "right", "center"),
        "Q3_K_M":  (19.6, 1.78, "left", "center"),
        "IQ4_XS":  (17.9, 0.600, "right", "center"),
        "Q4_K_M":  (23.6, 1.141, "left", "center"),
        "Q5_K_M":  (27.3, 0.660, "left", "center"),
        "Q6_K":    (31.4, 0.400, "left", "center"),
        "Q8_0":    (39.6, 0.210, "left", "center"),
    },
    "hellaswag": {
        "release": (13.6, 78.02, "left", "center"),
        "IQ2_M":   (13.6, 78.62, "left", "center"),
        "Q2_K":    (15.9, 76.40, "left", "center"),
        "IQ3_M":   (14.9, 80.35, "right", "center"),
        "Q3_K_M":  (19.4, 80.25, "left", "center"),
        "IQ4_XS":  (19.6, 81.35, "center", "center"),
        "Q4_K_M":  (21.713, 79.42, "center", "center"),
        "Q5_K_M":  (27.3, 80.15, "left", "center"),
        "Q6_K":    (31.0, 80.45, "left", "center"),
        "Q8_0":    (39.7, 80.25, "left", "center"),
        "BF16":    (71.067, 80.50, "center", "center"),
    },
}

# small explanatory callouts that get their own leader line to a marker
CALLOUTS = {
    "ppl": [dict(tag="release", text="best PPL of the 2-bit class",
                 xytext=(13.8, 8.06))],
}


def pareto_front(points, lower_better):
    """Best-so-far frontier: increasing size, strictly improving metric."""
    front = []
    best = None
    for x, y, _tag in sorted(points, key=lambda p: p[0]):
        better = (best is None) or (y < best if lower_better else y > best)
        if better:
            front.append((x, y))
            best = y
    return front


def marker_radius_pt(tag: str) -> float:
    return math.sqrt(STYLE.get(tag, DEFAULT)[1] / math.pi)


def _seg_crosses_rect(p, q, rect, pad=1.5):
    """True when segment p-q intersects rect (padded); everything in display px."""
    x0, y0, x1, y1 = rect[0] - pad, rect[1] - pad, rect[2] + pad, rect[3] + pad
    if max(p[0], q[0]) < x0 or min(p[0], q[0]) > x1:
        return False
    if max(p[1], q[1]) < y0 or min(p[1], q[1]) > y1:
        return False
    dx, dy = q[0] - p[0], q[1] - p[1]
    t0, t1 = 0.0, 1.0
    for pp, qq in ((-dx, p[0] - x0), (dx, x1 - p[0]),
                   (-dy, p[1] - y0), (dy, y1 - p[1])):
        if pp == 0:
            if qq < 0:
                return False
        else:
            r = qq / pp
            if pp < 0:
                if r > t1:
                    return False
                t0 = max(t0, r)
            else:
                if r < t0:
                    return False
                t1 = min(t1, r)
    return t0 <= t1


def text_bbox(ann, renderer):
    """Display bbox of the annotation's text only (leader line excluded)."""
    ann.get_window_extent(renderer)            # resolves the text position
    return Text.get_window_extent(ann, renderer)


def audit_panel(fig, ax, annotations, points, frontier, legend, name):
    """Warn about any label collision; returns the list of problems."""
    renderer = fig.canvas.get_renderer()
    pad = 1.5
    frame = ax.bbox
    problems = []
    boxes = [(text_bbox(ann, renderer), ann.get_text()) for ann in annotations]
    for bb, txt in boxes:
        if (bb.x0 < frame.x0 + pad or bb.x1 > frame.x1 - pad
                or bb.y0 < frame.y0 + pad or bb.y1 > frame.y1 - pad):
            problems.append(f"{name}: '{txt}' reaches outside the axes")
    for i, (a, atxt) in enumerate(boxes):
        for b, btxt in boxes[i + 1:]:
            if (a.x0 - pad < b.x1 and b.x0 - pad < a.x1
                    and a.y0 - pad < b.y1 and b.y0 - pad < a.y1):
                problems.append(f"{name}: '{atxt}' overlaps '{btxt}'")
    fpts = [ax.transData.transform(p) for p in frontier]
    fsegs = list(zip(fpts[:-1], fpts[1:]))
    for bb, txt in boxes:
        rect = (bb.x0, bb.y0, bb.x1, bb.y1)
        for x, y, tag in points:
            px, py = ax.transData.transform((x, y))
            r = marker_radius_pt(tag) * fig.dpi / 72.0
            if (bb.x0 - pad < px + r and px - r < bb.x1 + pad
                    and bb.y0 - pad < py + r and py - r < bb.y1 + pad):
                problems.append(f"{name}: '{txt}' touches the {tag} marker")
        for a, b in fsegs:
            if _seg_crosses_rect(a, b, rect, pad):
                problems.append(f"{name}: '{txt}' crosses the Pareto line")
        if legend is not None:
            lb = legend.get_window_extent(renderer)
            if (bb.x0 - pad < lb.x1 and lb.x0 - pad < bb.x1
                    and bb.y0 - pad < lb.y1 and lb.y0 - pad < bb.y1):
                problems.append(f"{name}: '{txt}' overlaps the legend")
    for p in problems:
        print("LAYOUT WARNING:", p)
    return problems


fig, axes = plt.subplots(2, 2, figsize=(15, 13.5), dpi=150)
fig.subplots_adjust(left=0.075, right=0.975, top=0.90, bottom=0.175, hspace=0.42, wspace=0.26)
fig.suptitle("Qwen3.8-35B-A3B-Distill — quantization retention vs file size",
             fontsize=16, fontweight="bold", y=0.978)
fig.text(0.5, 0.947,
         "same protocol for every row · our method is the blue triangle — the smallest point on every panel · K-quant / IQ quants use imatrix calibration",
         ha="center", fontsize=10.5, color="#444444")

n_problems = 0
for ax, (key, title, ylab, logy, higher_better) in zip(axes.ravel(), panels):
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlabel("File size (GB)", fontsize=11, fontweight="bold")
    ax.set_ylabel(ylab, fontsize=11, fontweight="bold")
    ax.set_xlim(8, 88)
    ax.grid(True, which="both", alpha=0.3, linewidth=0.6)
    if logy:
        ax.set_yscale("log")

    points = []
    for tag in ORDER:
        if tag not in data or key not in data[tag]:
            continue
        points.append((SIZES_GB[tag], data[tag][key], tag))

    # Pareto frontier
    front = pareto_front(points, lower_better=not higher_better)
    ax.plot([p[0] for p in front], [p[1] for p in front],
            color="#777777", lw=1.1, alpha=0.8, zorder=3,
            label="Pareto frontier (smaller + better)")

    # points
    for x, y, tag in points:
        marker, size, color = STYLE.get(tag, DEFAULT)
        ax.scatter([x], [y], marker=marker, s=size, c=color, edgecolors="black",
                   linewidths=0.7, zorder=5 if tag in STYLE else 4)

    # fix the axes limits before measuring/placing the labels
    ax.margins(x=0.05, y=0.22)
    if FLIPPED and not higher_better:
        ax.invert_yaxis()

    legend = None
    if key == "ppl":
        legend = ax.legend(loc="upper right", fontsize=9, frameon=True)

    # labels: explicit anchor + leader line (annotate keeps both ends attached)
    fig.canvas.draw()
    annotations = []
    for x, y, tag in points:
        if tag == "release":
            lab = f"ours {y:.3g}"
        elif tag in ("Q4_K_M", "IQ2_M", "BF16"):
            lab = f"{tag} {y:.3g}"
        else:
            lab = tag
        color = STYLE[tag][2] if tag in STYLE else "#444444"
        tx, ty, ha, va = ANCHORS[key][tag]
        annotations.append(ax.annotate(
            lab, xy=(x, y), xytext=(tx, ty), textcoords="data",
            ha=ha, va=va, fontsize=8.5, color=color, zorder=8,
            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6,
                            alpha=0.9, shrinkA=2, shrinkB=3)))

    for co in CALLOUTS.get(key, []):
        x0, y0 = next((x, y) for x, y, tag in points if tag == co["tag"])
        cx, cy = co["xytext"]
        annotations.append(ax.annotate(
            co["text"], xy=(x0, y0), xytext=(cx, cy), textcoords="data",
            ha="left", va="center", fontsize=8.5, color="#444444", style="italic",
            zorder=8, arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6,
                                      alpha=0.9, shrinkA=2, shrinkB=3)))

    fig.canvas.draw()
    n_problems += len(audit_panel(fig, ax, annotations, points, front, legend, key))

handles = [
    Line2D([], [], marker="^", color="w", markerfacecolor=blue, markeredgecolor="black",
           markersize=13, label="ours: ternary PQ2_0 + trained corrections (2.61 bpw)"),
    Line2D([], [], marker="D", color="w", markerfacecolor=red, markeredgecolor="black",
           markersize=8, label="Q4_K_M (competitor)"),
    Line2D([], [], marker="o", color="w", markerfacecolor=green, markeredgecolor="black",
           markersize=7, label="IQ2_M (imatrix 2-bit peer)"),
    Line2D([], [], marker="o", color="w", markerfacecolor=grey, markeredgecolor="black",
           markersize=6, label="other K-quant / IQ quant (imatrix)"),
    Line2D([], [], marker="s", color="w", markerfacecolor=purple, markeredgecolor="black",
           markersize=8, label="BF16 reference"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10,
           bbox_to_anchor=(0.5, 0.005))

fig.text(0.5, 0.125,
         "What the release buys:  11.3 GB — smallest on the chart   ·   97% of BF16 task accuracy, tied with Q4_K_M   ·   best PPL of the 2-bit class",
         ha="center", fontsize=10.5, color="#222222", fontweight="bold")
fig.text(0.5, 0.098,
         "Known gap: KLD tail — Q4-class distributional fidelity is still ahead (no imatrix calibration); that is the next lever.",
         ha="center", fontsize=10, color="#444444", style="italic")

stem = "retention-grid" if FLIPPED else "retention-grid-standard"
out = RESULTS / f"{stem}.png"
fig.savefig(out, dpi=150)
fig.savefig(RESULTS / f"{stem}.svg")
print(f"wrote {out} and {stem}.svg ({n_problems} layout warnings)")
