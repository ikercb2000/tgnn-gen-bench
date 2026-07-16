"""Figure conventions, in one place.

Mirrors the plotting style guide. Scripts import from here rather than
re-declaring constants, so a change lands once instead of drifting per script.

Target medium: NeurIPS text block 5.5 x 9 in, body 10 pt. Author every figure at
the width it will occupy on the page and place it with width=f\\linewidth, so
the render scale is 1 and fonts land at the size set here.
"""

# import packages

import matplotlib.pyplot as plt

# constants

TEXT_W = 5.5          # in — full text width; author widths are fractions of this

DPI = 300             # PNG export. PDF is vector; DPI applies only to rasterized layers.
DRAFT_DPI = 150       # iteration

# Author width = placed width. Pick one, then use width=f\linewidth to match.
FULL_W = 5.50         # f = 1.0
TWO3_W = 3.67         # f = 0.67
HALF_W = 2.65         # f = 0.48, two side by side
THIRD_W = 1.76        # f = 0.32

# Heights are chosen by aspect. Panel width = the author width above / n_cols.
ROW_H = 1.80          # panel height in a single-row figure
SINGLE_H = 2.20       # panel height for a standalone HALF_W figure

# Fonts are printed sizes, because scale is 1. Floor is 7 pt.
FONT_TITLE = 9
FONT_LABEL = 9
FONT_TICK = 8
FONT_LEGEND = 8

# Tol Bright, colourblind-safe. Roles are positional, not nominal.
PRIMARY = "#4477AA"
SECONDARY = "#EE6677"
TERTIARY = "#228833"
FOURTH = "#CCBB44"
FIFTH = "#66CCEE"
SIXTH = "#AA3377"
DARK_GREY = "#333333"   # reference marks
LIGHT_GREY = "#BBBBBB"  # reference lines, bands

# Marker sizes are DIAMETERS in points. plot() takes a diameter; scatter() takes
# an area, so pass the square there.
MARKER_SIZE = 7          # sparse plots on panels wider than ~2.5 in
MARKER_SIZE_SMALL = 4    # dense plots, or panels under ~2 in
MARKER_SIZE_DENSE = 3    # alpha-blended clouds
MARKER_EDGE_COLOR = "white"
MARKER_EDGE_WIDTH = 0.8

# Lines
LINE_LW = 1.2
REF_LW = 1.0

ERROR_BAND_ALPHA = 0.2
GRID_ALPHA = 0.3
GRID_LINESTYLE = "--"
GRID_LW = 0.5
CAP_SIZE = 4
ERRORBAR_LW = 1.5

BASE_RC = {
    "axes.labelsize": FONT_LABEL,
    "axes.titlesize": FONT_TITLE,
    "xtick.labelsize": FONT_TICK,
    "ytick.labelsize": FONT_TICK,
    "legend.fontsize": FONT_LEGEND,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.dpi": DPI,
}

USETEX_RC = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    # amssymb is not in matplotlib's default preamble.
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
}


def apply_style(usetex: bool = True, dpi: int = DPI) -> bool:
    """Apply the conventions. Returns whether usetex is actually in effect.

    shutil.which('latex') only proves latex is installed; a render can still
    fail on a missing package and matplotlib then falls back silently, giving
    sans-serif figures that look styled but are not. So probe with a real
    render and catch the exception.
    """
    plt.rcParams.update(BASE_RC)
    plt.rcParams["figure.dpi"] = dpi
    if not usetex:
        return False
    plt.rcParams.update(USETEX_RC)
    try:
        fig = plt.figure()
        fig.text(0.5, 0.5, r"$\lesssim$")
        fig.canvas.draw()
        plt.close(fig)
        return True
    except Exception:
        plt.close("all")
        plt.rcParams.update({"text.usetex": False, "font.family": "sans-serif"})
        return False


def save(fig, out, dpi: int = DPI) -> None:
    """Save PNG and PDF from the same figure.

    Never bbox_inches='tight': it re-crops after layout, so the saved figure is
    not the width it was authored at and the placed scale is no longer 1.
    """
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=dpi)


__all__ = [
    "BASE_RC", "CAP_SIZE", "DARK_GREY", "DPI", "DRAFT_DPI", "ERRORBAR_LW",
    "ERROR_BAND_ALPHA", "FIFTH", "FONT_LABEL", "FONT_LEGEND", "FONT_TICK",
    "FONT_TITLE", "FOURTH", "FULL_W", "GRID_ALPHA", "GRID_LINESTYLE", "GRID_LW",
    "HALF_W", "LIGHT_GREY", "LINE_LW", "MARKER_EDGE_COLOR", "MARKER_EDGE_WIDTH",
    "MARKER_SIZE", "MARKER_SIZE_DENSE", "MARKER_SIZE_SMALL", "PRIMARY", "REF_LW",
    "ROW_H", "SECONDARY", "SINGLE_H", "SIXTH", "TERTIARY", "TEXT_W", "THIRD_W",
    "TWO3_W", "USETEX_RC", "apply_style", "save",
]
