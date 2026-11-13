#!/usr/bin/env python3
"""Draw the paper figures from the frozen evidence.

  ~/venvs/defines/bin/python figures/make_figures.py

Rebuilds figures/figures_data.json with figdata.py (which reads every number from the frozen evidence), then
writes figures/out/<name>.pdf (for the paper: no title, fonts embedded) and <name>.png (with a title, for reading).
Needs matplotlib; the venv ~/venvs/defines has it. The PDFs carry no creation date, so a rebuild from the same
evidence with the same matplotlib gives the same files.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator   # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figdata                                      # noqa: E402

OUT = HERE / "out"
WIDTH = 3.5                                          # inches, one IEEE column

# palette: the reference instance of the data-viz method (slots 1-4 in their validated order) and its chart ink
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INSIDE_INK = ["#ffffff", "#ffffff", "#ffffff", "#0b0b0b"]   # label ink inside each fill
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#ffffff", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": AXIS, "ytick.color": AXIS,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.hashsalt": "hopcim",
})


def tick(v, _=None):
    """Axis ticks at powers of ten: 100, 1,000, 10K, 100K, 1M, 10M, 100M."""
    if v < 10_000:
        return f"{v:,.0f}"
    return f"{v / 1e6:.0f}M" if v >= 1e6 else f"{v / 1e3:.0f}K"


def value(v):
    """A bar's value: in full below 10,000, otherwise three significant figures with K or M."""
    if v < 10_000:
        return f"{v:,.0f}"
    d, s = (1e6, "M") if v >= 1e6 else (1e3, "K")
    return f"{v / d:#.3g}{s}"


def quiet_axes(ax, grid_axis="x"):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.tick_params(axis="x", length=2, width=0.6)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.5, linestyle="-")
    ax.set_axisbelow(True)


def save(fig, name, title):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02,
                metadata={"Creator": None, "Producer": None, "CreationDate": None})
    bb = fig.get_tightbbox(fig.canvas.get_renderer())       # inches: place the title above everything, flush left
    t = fig.text(bb.x0 / fig.get_figwidth(), bb.y1 / fig.get_figheight() + 0.03, title, ha="left", va="bottom",
                 fontsize=8, fontweight="bold", color=INK)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.05, metadata={"Software": None})
    t.remove()
    plt.close(fig)
    print(f"  figures/out/{name}.pdf, .png")


def log_bars(rows, labels, values, name, title, xlabel, height):
    fig, ax = plt.subplots(figsize=(WIDTH, height))
    y = range(len(rows))
    ax.barh(y, values, height=0.55, color=SERIES[0], linewidth=0)
    ax.set_xscale("log")
    k0, k1 = len(str(min(values))) - 1, len(str(max(values)))      # decades: 10**k0 below the smallest, 10**k1 above the largest
    ax.set_xlim(10 ** k0, max(values) * 9)                          # room for the largest bar's label
    ax.xaxis.set_major_locator(FixedLocator([10 ** k for k in range(k0, k1 + 1) if 10 ** k <= max(values) * 9]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(tick))
    for yi, v in zip(y, values):
        ax.text(v * 1.25, yi, value(v), va="center", ha="left", fontsize=6.5, color=INK)
    ax.set_yticks(list(y), labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    quiet_axes(ax)
    save(fig, name, title)


def fig1(d):
    rows = d["rows"]
    labels = [f"{r['id']}  {r['label']}" + ("*" if r["id"] == "C" else "") for r in rows]
    log_bars(rows, labels, [r["total"] for r in rows], "fig1_cpu_active_per_hop", d["title"],
             "CPU-active cycles per 20-ms hop (log scale)", 1.55)


def fig2(d):
    rows = d["rows"]
    log_bars(rows, [r["label"] for r in rows], [r["value"] for r in rows], "fig2_network_cycles", d["title"],
             "cycles per inference (log scale)", 1.55)


def fig3(d):
    rows = d["rows"]
    fig, ax = plt.subplots(figsize=(WIDTH, 1.75))
    y = [0, 1, 2, 3.35]                               # the model-only bar sits apart
    for yi, r in zip(y, rows):
        left = 0.0
        for k, g in enumerate(r["groups"]):
            ax.barh(yi, g["nJ"], left=left, height=0.6, color=SERIES[k], edgecolor=SURFACE, linewidth=1.0,
                    label=g["name"] if yi == 0 else None)
            if g["nJ"] >= 120:                          # wide enough for its label
                ax.text(left + g["nJ"] / 2, yi, f"{g['nJ']:.0f}", ha="center", va="center", fontsize=6, color=INSIDE_INK[k])
            left += g["nJ"]
        ax.text(left + 18, yi, f"{r['total_nJ']:,.0f}", va="center", ha="left", fontsize=6.5, color=INK)
    ax.set_yticks(y, [r["label"] if r["basis"] != "model only, not built" else f"{r['label']}\n(model, not built)" for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1550)
    ax.set_xticks([0, 250, 500, 750, 1000, 1250, 1500])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("nJ per inference [E], 28-nm estimate")
    quiet_axes(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False, handlelength=0.9, handleheight=0.9,
              columnspacing=1.2, borderaxespad=0, labelcolor=INK)
    save(fig, "fig3_energy_breakdown", d["title"])


def fig4(d):
    rows = d["rows"]

    def fmt(m, v):
        if "[E], nJ" in m:
            return f"{v:,.1f}"
        if "µW" in m:
            return f"{v:.1f}"
        return f"{v:,.0f}"

    fig, ax = plt.subplots(figsize=(WIDTH, 2.45))
    y = range(len(rows))
    ax.axvline(1.0, color=AXIS, linewidth=0.8, zorder=1)
    ax.scatter([r["ratio"] for r in rows], list(y), s=26, color=SERIES[0], edgecolor=SURFACE, linewidth=1.0, zorder=3)
    for yi, r in zip(y, rows):
        ax.text(r["ratio"] + 0.012, yi, f"{r['ratio']:.3f}", va="center", ha="left", fontsize=6.5, color=INK)
    ax.set_yticks(list(y), [f"{r['metric']}\nB1 {fmt(r['metric'], r['b1']['value'])}  ·  S {fmt(r['metric'], r['s']['value'])}"
                            for r in rows], fontsize=6.3, linespacing=1.25)
    ax.invert_yaxis()
    ax.set_xlim(0.74, 1.12)
    ax.set_xticks([0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1])
    ax.set_xlabel("B1 / S  (below 1: B1 is lower)")
    quiet_axes(ax)
    save(fig, "fig4_b1_vs_s", d["title"])


def main():
    data = figdata.build()
    figdata.OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print("figures/figures_data.json from the frozen evidence")
    f = data["figures"]
    fig1(f["fig1_cpu_active_per_hop"])
    fig2(f["fig2_network_cycles"])
    fig3(f["fig3_energy_breakdown"])
    fig4(f["fig4_b1_vs_s"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
