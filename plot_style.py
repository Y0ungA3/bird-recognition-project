"""Shared visual language for report-ready scientific figures."""
from pathlib import Path

import matplotlib as mpl
import numpy as np

BG = "#F7F5EF"
PANEL = "#FFFEFA"
TEXT = "#263238"
MUTED = "#68737A"
GRID = "#D8DDD9"
NAVY = "#355C7D"
TEAL = "#3B8B87"
GOLD = "#D49A3A"
CORAL = "#C45B4B"
PURPLE = "#7B668F"
PALE_NAVY = "#BDD1DF"
PALE_TEAL = "#BFDCD7"
PALE_GOLD = "#F0D8AA"


def apply_report_style():
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "medium",
            "axes.labelsize": 10,
            "axes.labelcolor": TEXT,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "savefig.facecolor": BG,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.14,
            "figure.dpi": 150,
        }
    )


def style_axis(axis, y_grid=True):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(GRID)
    axis.spines["bottom"].set_color(GRID)
    axis.tick_params(length=0, pad=6)
    if y_grid:
        axis.yaxis.grid(True, color=GRID, linewidth=0.7, alpha=0.75)
    axis.set_axisbelow(True)


def solid_bar(axis, x, values, errors, color, width=0.62, alpha=1.0):
    return axis.bar(
        x,
        values,
        width,
        yerr=errors,
        color=color,
        edgecolor="none",
        alpha=alpha,
        error_kw={
            "ecolor": TEXT,
            "elinewidth": 1.2,
            "capsize": 3,
            "capthick": 1.2,
        },
        zorder=3,
    )


def direct_labels(axis, x, values, errors=0, fmt="{:.2f}", color=TEXT):
    errors = np.broadcast_to(errors, (len(values),))
    for x_value, value, error in zip(x, values, errors):
        axis.text(
            x_value,
            value + error + 1.3,
            fmt.format(value),
            ha="center",
            va="bottom",
            color=color,
            fontsize=9,
            fontweight="medium",
        )


def panel_letter(axis, letter, title, subtitle=None):
    axis.text(
        0.0,
        1.10,
        letter,
        transform=axis.transAxes,
        color=GOLD,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    axis.text(
        0.055,
        1.10,
        title,
        transform=axis.transAxes,
        color=TEXT,
        fontsize=11,
        fontweight="medium",
        ha="left",
        va="bottom",
    )
    if subtitle:
        axis.text(
            0.055,
            1.035,
            subtitle,
            transform=axis.transAxes,
            color=MUTED,
            fontsize=9,
            ha="left",
            va="bottom",
        )


def page_header(figure, title, subtitle):
    figure.patches.append(
        mpl.patches.Rectangle(
            (0.035, 0.965),
            0.035,
            0.008,
            transform=figure.transFigure,
            facecolor=GOLD,
            edgecolor="none",
            zorder=10,
        )
    )
    figure.text(
        0.085,
        0.972,
        title,
        ha="left",
        va="top",
        color=TEXT,
        fontsize=18,
        fontweight="medium",
    )
    figure.text(
        0.085,
        0.916,
        subtitle,
        ha="left",
        va="top",
        color=MUTED,
        fontsize=10,
    )


def save(figure, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
