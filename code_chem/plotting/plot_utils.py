"""Shared matplotlib utilities for abundance plots.

Provides importable helpers that can be called after lines are plotted:

    apply_abundance_axis_limits(ax)
        Inspects all Line2D objects on *ax* and sets:
          - x-axis  : [min_r, max_r]  (initial → final radius)
          - y-axis  : [upper / 1e10, upper]  where *upper* is the next full
                      decade above the plotted maximum abundance

    apply_abundance_axis_limits_shared(axes)
        Same as above but collects data across ALL provided axes first, then
        applies a single consistent limit to every axis.  Use this whenever
        axes share a y-axis (sharey=True) so that limits are computed from
        all species together (e.g. CO in panel 0 won't be clipped by the
        lower maximum of the daughters panel 1).

    add_log_ticks(ax)
        Adds logarithmic minor ticks (10 sub-divisions per decade) on every
        axis that is already in log scale, and shows ticks on all four edges
        directed outward.  Also applies a dashed grid and tick label size 16.
        Safe to call on linear-scale axes too.

Both limit functions are no-ops when the axes contain no valid data or the
relevant scale is not 'log'.
"""

from __future__ import annotations

import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter
from matplotlib import pyplot as plt
import re

# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _collect_xy(axes_list) -> tuple[list[float], list[float]]:
    all_x: list[float] = []
    all_y: list[float] = []
    for ax in axes_list:
        for line in ax.get_lines():
            xd = np.asarray(line.get_xdata(), dtype=float)
            yd = np.asarray(line.get_ydata(), dtype=float)
            valid = np.isfinite(xd) & np.isfinite(yd) & (xd > 0) & (yd > 0)  # ← added y check
            all_x.extend(xd[valid].tolist())
            all_y.extend(yd[valid].tolist())
    return all_x, all_y


# ---------------------------------------------------------------------------
# Abundance axis limits
# ---------------------------------------------------------------------------

def apply_abundance_axis_limits(ax) -> None:
    """Set smart axis limits for an abundance plot by inspecting plotted lines.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes that already contain the abundance traces.

    Notes
    -----
    X-axis
        Spanning the full radial range of all plotted data
        (from the smallest ``r`` to the largest ``r``).

    Y-axis
        Upper limit = next full order of magnitude above the plotted maximum
        (e.g. max = 3×10⁻⁵ → upper = 10⁻⁴).
        Lower limit = upper × 10⁻¹⁰  (10 decades below upper).
    """
    all_x, all_y = _collect_xy([ax])

    if all_x:
        ax.set_xlim(min(all_x), max(all_x))

    if all_y:
        max_ab = max(all_y)
        upper_exp = np.ceil(np.log10(max_ab))
        # Guard: if max_ab is exactly a power of 10, step one decade higher so
        # the top line is not right at the axis edge.
        if np.isclose(max_ab, 10.0 ** upper_exp):
            upper_exp += 1.0
        upper = 10.0 ** upper_exp
        lower = upper * 1e-10
        ax.set_ylim(lower, upper)


def apply_abundance_axis_limits_shared(axes) -> None:
    """Set consistent axis limits across a group of axes sharing x or y.

    Collects all plotted data from every axis in *axes*, computes a single
    set of limits, then applies them uniformly.  This is the correct function
    to use when ``sharey=True`` or ``sharex=True``, because calling
    :func:`apply_abundance_axis_limits` axis-by-axis causes later calls to
    overwrite limits set by earlier ones.

    Parameters
    ----------
    axes : sequence of matplotlib.axes.Axes
    """
    axes_list = list(np.asarray(axes).flat)
    all_x, all_y = _collect_xy(axes_list)

    x_lim = (min(all_x), max(all_x)) if all_x else None
    if all_y:
        max_ab = max(all_y)
        upper_exp = np.ceil(np.log10(max_ab))
        if np.isclose(max_ab, 10.0 ** upper_exp):
            upper_exp += 1.0
        upper = 10.0 ** upper_exp
        lower = upper * 1e-10
        y_lim = (lower, upper)
    else:
        y_lim = None

    for ax in axes_list:
        if x_lim is not None:
            ax.set_xlim(*x_lim)
        if y_lim is not None:
            ax.set_ylim(*y_lim)


# ---------------------------------------------------------------------------
# Log minor ticks on all four edges
# ---------------------------------------------------------------------------

def add_log_ticks(ax) -> None:
    """Add logarithmic minor ticks on all four edges of *ax*.

    For each axis dimension (x, y) that is in log scale the minor tick
    locator is set to ``LogLocator(subs='auto')`` which produces the familiar
    increasing-spacing sub-ticks within each decade.  Ticks are placed on
    all four sides and directed outward.  Also applies a dashed grid style
    and sets the tick-label font size to 16.

    Safe to call on linear-scale axes: those dimensions are skipped.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    """
    if ax.get_xscale() == "log":
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs="auto", numticks=100))
        ax.xaxis.set_minor_formatter(NullFormatter())

    if ax.get_yscale() == "log":
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs="auto", numticks=100))
        ax.yaxis.set_minor_formatter(NullFormatter())

    # Ticks on all four sides, pointing outward, font size 14
    ax.tick_params(which="both", top=True, right=True, bottom=True, left=True,
                   direction="out", labelsize=14)
    ax.tick_params(which="minor", length=3, width=0.6)
    ax.tick_params(which="major", length=6, width=0.9)

    # Dashed grid
    ax.grid(True, linestyle="--", alpha=0.8)

def set_plot_style(dark_mode: bool = False) -> None:
    if dark_mode:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Times New Roman",
            "font.sans-serif": "helvetica",

            "figure.facecolor": "black",
            "axes.facecolor": "black",
            "savefig.facecolor": "black",

            "text.color": "white",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "axes.edgecolor": "white",
        })
    else:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Times New Roman",
            "font.sans-serif": "helvetica",

            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",

            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.edgecolor": "black",
        })

# ============================================================================
# Molecule / species LaTeX formatting
# ============================================================================

def format_species_label(species: str) -> str:
    """
    Convert a chemical species string into publication-quality LaTeX.

    Examples
    --------
    CH4      -> $\mathrm{CH}_4$
    H2O      -> $\mathrm{H}_2\mathrm{O}$
    HCO+     -> $\mathrm{HCO}^+$
    N2H+     -> $\mathrm{N}_2\mathrm{H}^+$
    C18O     -> $\mathrm{C}^{18}\mathrm{O}$
    13CO     -> $^{13}\mathrm{CO}$
    H13CO+   -> $\mathrm{H}^{13}\mathrm{CO}^+$
    """

    s = species.strip()

    # ----------------------------------------------------------------------
    # Extract charge
    # ----------------------------------------------------------------------
    charge = ""
    m = re.search(r"([+-]+)$", s)
    if m:
        charge = m.group(1)
        s = s[:-len(charge)]

    # ----------------------------------------------------------------------
    # Tokenize:
    #   isotope prefixes
    #   element symbols
    #   numeric subscripts
    # ----------------------------------------------------------------------
    tokens = re.findall(r"\d+|[A-Z][a-z]?", s)

    out = []

    i = 0
    while i < len(tokens):
        tok = tokens[i]

        # --------------------------------------------------------------
        # Leading isotope number
        # Example: 13CO
        # --------------------------------------------------------------
        if tok.isdigit():
            if i + 1 < len(tokens):
                nxt = tokens[i + 1]
                out.append(rf"^{{{tok}}}\mathrm{{{nxt}}}")
                i += 2
                continue

        # --------------------------------------------------------------
        # Element symbol
        # --------------------------------------------------------------
        # Element symbol
        elem = r"\mathrm{{" + tok + r"}}"

        # Following number becomes subscript
        if i + 1 < len(tokens) and tokens[i + 1].isdigit():
            num = tokens[i + 1]
            elem += rf"_{{{num}}}"

            i += 1

        out.append(elem)
        i += 1

    body = "".join(out)

    if charge:
        body += rf"^{{{charge}}}"

    return rf"${body}$"