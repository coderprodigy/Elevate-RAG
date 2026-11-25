"""Figure style: validated categorical palette (fixed order), recessive chrome, thin marks.

Palette validated with the dataviz validator (light surface #fcfcfb): all checks pass; three hues
are below 3:1 contrast, so every figure carries a legend and a companion markdown table.
Colour follows the *entity* (e.g. V4 is always blue), never its rank in a figure.
"""

from __future__ import annotations

from typing import Any

CATEGORICAL = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Fixed entity → slot maps (one per figure family; at most 8 entities each).
PIPELINE_ORDER = {
    "baselines": ["v4", "v0", "v1", "v2", "v3"],
    "modes": ["v4", "lr_naive", "lr_local", "lr_global", "lr_hybrid", "lr_mix"],
}


def color_for(entity: str, family: list[str]) -> str:
    """Slot by fixed position in ``family``; entities outside it share the muted grey."""
    return CATEGORICAL[family.index(entity)] if entity in family else MUTED


def apply_style() -> None:
    import matplotlib as mpl

    style: dict[str, Any] = {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "font.size": 9,
    }
    mpl.rcParams.update(style)  # type: ignore[arg-type]  # rc keys are checked at runtime
