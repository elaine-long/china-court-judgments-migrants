"""Project figure style (AGENTS.md §4).

- scripts generate every figure; no manual edits;
- paper figures: vector PDF; diagnostics may be PNG;
- no titles inside figures (captions live in LaTeX);
- groups are distinguishable in black and white: line style + marker, greys only.
"""
import logging

import matplotlib

matplotlib.use("Agg")
logging.getLogger("fontTools").setLevel(logging.WARNING)  # PDF font subsetting is chatty
import matplotlib.pyplot as plt  # noqa: E402

RC = {
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.5,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
}
# (label, colour, linestyle, marker): four groups, readable in greyscale
GROUPS = {
    "pilot_local": ("Pilot, local", "black", "-", "o"),
    "pilot_migrant": ("Pilot, migrant", "0.45", "-", "s"),
    "nonpilot_local": ("Non-pilot, local", "black", "--", "^"),
    "nonpilot_migrant": ("Non-pilot, migrant", "0.45", "--", "v"),
}
SERIES = [("black", "-", "o"), ("0.45", "--", "s"), ("0.2", ":", "^"), ("0.6", "-.", "v")]
SIZE_1COL = (3.4, 2.4)
SIZE_2COL = (6.8, 3.0)


def use():
    plt.rcParams.update(RC)


def vlines(ax, dates, color="0.35"):
    for d in dates:
        ax.axvline(d, color=color, lw=0.6, ls=":")


def save(fig, path):
    """Save as given; paper figures should use a .pdf path."""
    fig.savefig(path)
    plt.close(fig)
