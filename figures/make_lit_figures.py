#!/usr/bin/env python3
"""Draw the literature figures from lit_data.py:  ~/venvs/defines/bin/python figures/make_lit_figures.py
Writes out/lit1_capability.*, out/lit2_cycles.*, out/lit3_depthwise.* (pdf for the thesis, png for reading)."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.patches import Rectangle                     # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lit_data as D                                         # noqa: E402
OUT = HERE / "out"
HOP, GREY, INK, INK2, MUTED, GRID, AXIS, BAND = "#2a78d6", "#898781", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#e8f1fb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "text.color": INK, "axes.labelcolor": INK2, "xtick.labelcolor": INK2,
                     "ytick.labelcolor": INK, "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "pdf.fonttype": 42, "svg.hashsalt": "hopcim"})


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03, metadata={"Creator": None, "Producer": None, "CreationDate": None})
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.08, metadata={"Software": None})
    plt.close(fig)
    print("  out/" + name)


def fig1():
    rows = [(n, v) for n, v in D.WORKS] + [("HopCIM (this work)", D.HOPCIM)]
    nc, nr = len(D.COLS), len(rows)
    cw, rh, left, top = 0.64, 0.32, 1.75, 0.98
    fig = plt.figure(figsize=(left + nc * cw + 0.05, top + nr * rh + 0.30))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, left + nc * cw + 0.05); ax.set_ylim(top + nr * rh + 0.30, 0); ax.axis("off")
    for j, (lab, _) in enumerate(D.COLS):
        ax.text(left + (j + .5) * cw, top - 0.08, lab, ha="left", va="bottom", rotation=38, fontsize=6.6, color=INK2, linespacing=1.0)
    for i, (n, v) in enumerate(rows):
        y = top + i * rh; hop = i == nr - 1
        if hop:
            ax.add_patch(Rectangle((0, y), left + nc * cw, rh, color=BAND, lw=0))
        if i:
            ax.plot([0, left + nc * cw], [y, y], color=GRID if not hop else AXIS, lw=0.6 if not hop else 1.0)
        ax.text(0.06, y + rh / 2, n, va="center", ha="left", fontsize=7.6, fontweight="bold" if hop else "normal", color=INK)
        for j, c in enumerate(v):
            x, yy, col = left + (j + .5) * cw, y + rh / 2, HOP if hop else INK2
            if c == "y":
                ax.plot(x, yy, "o", ms=7.2, color=col, mec=col)
            elif c == "p":
                ax.plot(x, yy, marker="o", ms=7.2, fillstyle="left", mfc=col, mec=col, mew=0.9, color=col, markerfacecoloralt="white")
            elif c == "n":
                ax.plot(x, yy, "o", ms=7.2, mfc="white", mec=MUTED, mew=0.9)
            else:
                ax.text(x, yy, "–", ha="center", va="center", fontsize=9, color=MUTED)
    y = top + nr * rh + 0.15
    for k, (sym, t) in enumerate([("y", "yes"), ("p", "partly"), ("n", "no"), ("-", "not stated")]):
        x = left + 0.3 + k * 1.15
        if sym == "y":
            ax.plot(x, y, "o", ms=6, color=INK2)
        elif sym == "p":
            ax.plot(x, y, marker="o", ms=6, fillstyle="left", mfc=INK2, mec=INK2, color=INK2, markerfacecoloralt="white", mew=0.9)
        elif sym == "n":
            ax.plot(x, y, "o", ms=6, mfc="white", mec=MUTED, mew=0.9)
        else:
            ax.text(x, y, "–", ha="center", va="center", fontsize=9, color=MUTED)
        ax.text(x + 0.10, y, t, va="center", ha="left", fontsize=6.8, color=INK2)
    save(fig, "lit1_capability")


def num(v):
    return f"{v / 1e6:.3g}M" if v >= 1e6 else f"{v / 1e3:.3g}K"


def fig2():
    rows = D.CYCLES
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    for i, (lab, lo, hi, kind, _) in enumerate(rows):
        c = HOP if kind == "hopcim" else GREY
        if hi > lo:
            ax.plot([lo, hi], [i, i], color=c, lw=2.2, solid_capstyle="round", zorder=2)
        for v in {lo, hi}:
            ax.plot(v, i, "o", ms=8, color=c, mec="white", mew=1.4, zorder=3)
        t = num(lo) if hi == lo else f"{num(lo)}–{num(hi)}"
        ax.text(hi * 1.28, i, t, va="center", ha="left", fontsize=7.2, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows]); ax.invert_yaxis()
    for t, r in zip(ax.get_yticklabels(), rows):
        if r[3] == "hopcim":
            t.set_fontweight("bold")
    ax.set_xscale("log"); ax.set_xlim(2e4, 5e8)
    ax.xaxis.set_major_locator(FixedLocator([1e5, 1e6, 1e7, 1e8])); ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:.0f}M" if v >= 1e6 else f"{v / 1e3:.0f}K"))
    ax.set_xlabel("clock cycles per inference, KWS DS-CNN (log scale; fewer is better)")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", length=2, color=AXIS)
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    save(fig, "lit2_cycles")


def fig3():
    rows = D.DW
    fig, ax = plt.subplots(figsize=(5.6, 2.1))
    for i, (lab, v, kind, _) in enumerate(rows):
        ax.barh(i, v, height=0.58, color=HOP if kind == "hopcim" else GREY, linewidth=0)
        ax.text(v + 0.5, i, f"{v:.2f}", va="center", fontsize=7.4, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows]); ax.invert_yaxis()
    for t, r in zip(ax.get_yticklabels(), rows):
        if r[2] == "hopcim":
            t.set_fontweight("bold")
    ax.set_xlim(0, 52); ax.set_xlabel("estimated power at mode U, µW [E] (28-nm model)")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", length=2, color=AXIS)
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    save(fig, "lit3_depthwise")


fig1(); fig2(); fig3()
