"""Shared figure style."""
import glob
import os

import numpy as np

INK, MUTED, LINK, RULE = "#2E2350", "#625A7C", "#493F67", "#7D7692"
HAIR, WASH, PANEL, EOSIN = "#E6E5EA", "#D9D7E0", "#FFFFFF", "#CA6B8B"


def mix(a, b, pct):
    """Colour `a` moved toward colour `b` by `pct` percent."""
    A = [int(a.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    B = [int(b.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    return "#%02X%02X%02X" % tuple(round(x + (y - x) * pct / 100) for x, y in zip(A, B))


HEMATOXYLIN = mix(INK, "#5B4B9A", 40)

GROUP_LABEL = {"AKI": "AKI", "CKD": "CKD", "DM-R": "DM-R", "Healthy_Reference": "Reference"}
GROUP_COLOR = {"AKI": "#B83A4B", "CKD": "#1B7F79", "DM-R": "#D9A21B", "Healthy_Reference": "#8C8799"}

MODEL_LABEL = {"zero_shot": "DeepSpot-M (zero-shot)", "kidney": "DeepSpot-M Kidney",
               "group_specific": "DeepSpot-M Kidney (group-specific)", "measured": "10x Visium (measured)",
               "stpath": "STPath", "squall": "SQUALL",
               "stpath_ft": "STPath fine-tuned", "squall_ft": "SQUALL fine-tuned"}
DSM_HUE, STPATH_HUE, SQUALL_HUE = "#2D6BB4", "#3FA65F", "#A5263A"
MODEL_COLOR = {"zero_shot": mix(DSM_HUE, PANEL, 50), "kidney": DSM_HUE, "group_specific": "#C8611F",
               "measured": INK, "stpath": mix(STPATH_HUE, PANEL, 50), "squall": mix(SQUALL_HUE, PANEL, 50),
               "stpath_ft": STPATH_HUE, "squall_ft": SQUALL_HUE}
MODEL_MARKER = {"measured": "D", "zero_shot": "o", "kidney": "s", "group_specific": "^", "stpath": "v",
                "squall": "X", "stpath_ft": "v", "squall_ft": "X"}
ADAPTED = {"kidney", "group_specific", "stpath_ft", "squall_ft"}
EXPR_CMAP = "magma"

MM = 1 / 25.4
FS_HEADER, FS_LAB, FS_TICK, FS_SMALL = 8, 6, 5, 5
FS_LETTER = 11
HEADER = dict(fontsize=FS_HEADER, fontweight="normal", ha="left", va="baseline", color=INK)
LETTER = dict(fontsize=FS_LETTER, fontweight="bold", ha="left", va="baseline", color=INK)
FONT = "Liberation Sans"
FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fonts")
FONT_FALLBACK = "DejaVu Sans"
LW = 1.0


def font_family():
    """FONT from the bundled files in FONT_DIR, then FONT_FALLBACK."""
    from matplotlib import font_manager
    for f in sorted(glob.glob(os.path.join(FONT_DIR, "*.ttf"))):
        font_manager.fontManager.addfont(f)
    return [FONT, FONT_FALLBACK]


def fmt_p(p):
    """P value as 'P < 0.001', 'P = 0.003' or 'P = 0.03'."""
    P = r"$\mathit{P}$"
    if p < 0.001:
        return f"{P} < 0.001"
    return f"{P} = {p:.3f}" if p < 0.01 else f"{P} = {p:.2f}"


def thou(n):
    """Integer with thin-space thousands separators: 17 119."""
    return f"{int(round(float(n))):,}".replace(",", "\u2009")


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fam = font_family()
    plt.rcParams.update({
        "font.family": fam, "mathtext.fontset": "custom", "mathtext.rm": fam[0], "mathtext.sf": fam[0],
        "mathtext.it": f"{fam[0]}:italic", "mathtext.bf": f"{fam[0]}:bold", "mathtext.cal": fam[0],
        "font.size": FS_LAB, "text.color": INK,
        "axes.edgecolor": RULE, "axes.labelcolor": INK, "axes.linewidth": LW,
        "axes.labelsize": FS_LAB, "axes.titlesize": FS_LAB, "axes.titleweight": "normal",
        "axes.titlepad": 3, "axes.spines.top": False, "axes.spines.right": False,
        "axes.facecolor": PANEL, "figure.facecolor": PANEL,
        "xtick.color": RULE, "ytick.color": RULE, "xtick.labelcolor": MUTED,
        "ytick.labelcolor": MUTED, "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "xtick.major.width": LW, "ytick.major.width": LW, "xtick.minor.width": LW, "ytick.minor.width": LW,
        "xtick.major.size": 2, "ytick.major.size": 2, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
        "lines.linewidth": LW, "lines.markeredgewidth": LW, "patch.linewidth": LW, "grid.linewidth": LW,
        "hatch.linewidth": LW, "contour.linewidth": LW,
        **{f"boxplot.{p}.linewidth": LW for p in ("boxprops", "whiskerprops", "capprops", "medianprops",
                                                   "meanprops", "flierprops")},
        "boxplot.flierprops.markeredgewidth": LW,
        "legend.frameon": False, "legend.fontsize": FS_SMALL,
        "legend.handlelength": 1.0, "legend.handletextpad": 0.4, "legend.columnspacing": 1.2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
    })


def save(fig, path):
    """Write `path`.pdf and `path`.png."""
    for fmt in ("pdf", "png"):
        fig.savefig(f"{path}.{fmt}", dpi=600, bbox_inches="tight", pad_inches=0.03, facecolor=PANEL)


def header(fig, x, y, letter, text=None):
    """Panel letter and title at figure coordinates (x, y)."""
    from matplotlib.transforms import offset_copy
    t = fig.text(x, y, letter.lower(), **LETTER)
    if text:
        fig.text(x, y, text, transform=offset_copy(fig.transFigure, fig=fig, x=FS_LETTER * 1.05,
                                                   units="points"), **HEADER)
    return t


def split_dots(ax, x, vals, color, w=0.6, ms=1.6, mean=None, sd=None, line_color=INK, z=2):
    """One dot per value spread over width `w` at `x`, with the mean as a horizontal and the SD as a vertical line."""
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return np.nan, np.nan
    mean = float(v.mean()) if mean is None else float(mean)
    sd = (float(v.std(ddof=1)) if len(v) > 1 else np.nan) if sd is None else float(sd)
    off = np.linspace(-w / 2, w / 2, len(v)) if len(v) > 1 else np.zeros(1)
    ax.scatter(x + off, v, s=ms ** 2, color=color, lw=0, zorder=z)
    ax.plot([x - w / 2, x + w / 2], [mean, mean], color=line_color, lw=LW, solid_capstyle="butt", zorder=z + 1)
    if np.isfinite(sd) and sd > 0:
        ax.plot([x, x], [mean - sd, mean + sd], color=line_color, lw=LW, solid_capstyle="butt", zorder=z + 1)
    return mean, sd

