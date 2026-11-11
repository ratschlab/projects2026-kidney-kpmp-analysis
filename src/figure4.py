"""Figure 4: the KPMP virtual spatial transcriptomics atlas, compared with measured Visium and Xenium data."""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr, wilcoxon

import style
from style import EOSIN, GROUP_COLOR, GROUP_LABEL, HAIR, HEMATOXYLIN, INK, MUTED, PANEL, RULE, mix

W_MM = 183
LETTER = 2
GAP = 6.0

PROG_LABEL = {"proximal_tubule": "Proximal tubule", "loop_tal": "Loop of Henle · TAL",
              "distal_cd": "Distal · collecting duct", "podocyte": "Podocyte", "injury_repair": "Injury · repair",
              "fibroblast": "Fibroblast", "vascular_smc": "Vessel · smooth muscle", "immune": "Immune · myeloid"}
D_BLOCKS = [("Epithelium", ["proximal_tubule", "loop_tal", "distal_cd", "podocyte"]),
            ("Injury", ["injury_repair"]), ("Stroma", ["fibroblast", "vascular_smc"]), ("Immune", ["immune"])]
D_PROGS = [p for _, ps in D_BLOCKS for p in ps]
D_COLUMNS = [("function", [("egfr", "eGFR"), ("albuminuria", "Albuminuria"), ("proteinuria", "Proteinuria"),
                           ("hba1c", "HbA1c")]),
             ("pathologist", [("C_if_pct", "Fibrosis (IF) %"), ("C_ta_common_type_pct", "Atrophy (TA) %"),
                              ("C_tubular_injury_pct", "Tubular injury %"),
                              ("C_mononuclear_wbc_pct", "Infiltrate %")]),
             ("", [("dmr_vs_ckd", "DM-R vs CKD")])]
D_VARS = [v for _, cs in D_COLUMNS for v, _ in cs]
R_MAX, R_CMAP = 0.5, "RdBu_r"
D_RING_S = 9.0
NA_FILL = mix(HAIR, PANEL, 40)

NICHE_COLOR = {"N1": "#1baf7a", "N2": "#5b3a8e", "N3": "#9e1b32", "N4": "#eb6834", "N5": "#e87ba4",
               "N6": "#0d6e6e", "N9": "#2a78d6", "N8": "#A9A6B4"}
QUALITY = {"N8"}
NICHE_IDS = sorted(NICHE_COLOR, key=lambda n: int(n[1:]))
C_COLS = [["N4", "N1", "N5", "N3"], ["N9", "N2", "N6", None]]
BAND_LABEL = {"low": "Low fibrosis", "mid": "Moderate fibrosis", "high": "Severe fibrosis"}
UMAP_LABEL_MM = 20.0
LATTICE_UM = 56

ORDER = ["CKD", "DM-R", "Healthy_Reference", "AKI"]
STAIN_COLOR = {"FFPE": HEMATOXYLIN, "frozen": mix(HEMATOXYLIN, PANEL, 62)}

B_X, R2, LEG_X = 106, 67, 144
BX_AVAIL, BX_GAP, BX_HMAX, BX_MINW = LEG_X - 6, 3.6, 15.0, 31.0
C_LW, C_GUTTER, C_GAP, C_ROW_GAP = 19.5, 3.5, 0.4, 1.1
C_BAR_UM = 20
ROW_DE_H, ROW_FG_H = 52.0, 41.0

E_MIN_HITS, E_MAX = 5, 16

F_SETS = [("atlas_PT_niches_pct", "xenium_pct_PT", "Proximal tubule", ["N4", "N5", "N3"], "N4"),
          ("atlas_N1_pct", "xenium_pct_TAL", "Thick ascending limb", ["N1"], "N1")]
F_X, F_TOP, F_W, F_H = (12.0, 50.0), 8.5, 26.0, 22.0
F_LIM = (86, 44)
G_X = 85.0
G_GENE = "SLC13A1"
BEFORE, AFTER, LINK = mix(RULE, PANEL, 45), HEMATOXYLIN, mix(HAIR, RULE, 25)
XEN_PANEL = {"hKidney_480g": ("Xenium hKidney panel", "#5B4B9A"), "hMulti_480g": ("Xenium hMulti panel", EOSIN)}
MAP_W, MAP_GAP, BAR_UM, CELL_UM, MAP_PAD = 34.0, 1.4, 500, 12.0, 20.0
FS_AX, FS_TK, FS_NOTE = 5.2, 4.6, 4.6
DESC_LABEL = {"CM_arteriolar_hyalinosis": "Arteriolar hyalinosis", "C_cortex_pct": "Cortex %",
              "C_if_focality": "Fibrosis focality", "C_if_pct": "Fibrosis (IF) %",
              "C_interstitial_eosinophils": "Eosinophils", "C_interstitial_eosinophils_focality": "Eosinophil focality",
              "C_mononuclear_wbc_focality": "Infiltrate focality", "C_mononuclear_wbc_lymphocytes": "Lymphocytes",
              "C_mononuclear_wbc_pct": "Infiltrate %", "C_ta_common_type_pct": "Atrophy (TA) %",
              "C_tubular_injury_swelling": "Tubular swelling", "C_tubular_injury_tubulitis": "Tubulitis",
              "C_tubular_injury_tubulitis_lymphocytic": "Lymphocytic tubulitis",
              "C_tubular_injury_vacuolization": "Vacuolization", "M_medulla_pct": "Medulla %",
              "M_mononuclear_wbc_plasma_cells": "Plasma cells (medulla)", "C_tubular_injury_pct": "Tubular injury %",
              "CM_arteriosclerosis": "Arteriosclerosis", "C_intratubular_hyaline": "Hyaline casts",
              "C_mononuclear_wbc_plasma_cells": "Plasma cells", "M_mononuclear_wbc_pct": "Infiltrate % (medulla)",
              "M_intratubular_acellular_casts_pct": "Acellular casts % (medulla)",
              "C_intratubular_acellular_casts_pct": "Acellular casts %",
              "C_intratubular_acellular_casts_nos": "Acellular casts",
              "M_mononuclear_wbc_focality": "Infiltrate focality (medulla)"}


def descriptor_label(d):
    if d in DESC_LABEL:
        return DESC_LABEL[d]
    return d.split("_", 1)[1].replace("_", " ").capitalize() + (" (medulla)" if d.startswith("M_") else "")


def bh(p):
    """Benjamini-Hochberg adjusted P values (NaN stays NaN)."""
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    o = ok[np.argsort(p[ok])]
    q = p[o] * len(o) / np.arange(1, len(o) + 1)
    out[o] = np.minimum(np.minimum.accumulate(q[::-1])[::-1], 1)
    return out


def c_markers(markers):
    return markers.split()[:2]


def quality_items(niches):
    q = [n for n in niches if n in QUALITY]
    return [(n, r) for n in q for r in range(8 // len(q))]


def load_spots(d):
    return dict(np.load(d / "umap_spots.npz"))


def load_lattice(d):
    return pd.read_csv(d / "example_lattice.csv.gz")


def xenium_example(d):
    """Cells of the drawn window, predictions as stored upstream (float16), with their map coordinates (window
    turned horizontal, starting at 0)."""
    t = pd.read_csv(d / "xenium_example.csv.gz").astype({"unadapted": np.float16, "adapted": np.float16})
    xs, ys = horizontal(t.x_um.to_numpy(), t.y_um.to_numpy())
    return t.assign(gene=G_GENE, x_map_um=xs, y_map_um=ys)


def tables(data_dir):
    d = Path(data_dir)
    cohort = pd.read_csv(d / "cohort.csv").replace({"diagnosis": {"DM resilient": "no lesions"}})
    stats = {k: int(v) if float(v).is_integer() else float(v)
             for k, v in pd.read_csv(d / "atlas_stats.csv").itertuples(index=False)}
    slides = pd.read_csv(d / "gt450_slides.csv")
    spots = load_spots(d)
    niches = niche_table(d, slides, spots["niche"])
    examples = example_table(d, slides)
    selection = pd.read_csv(d / "tile_selection.csv")
    tiles = tile_table(d, niches, selection)
    programs = program_table(d)
    cells = pd.read_csv(d / "visium_cells.csv").dropna(subset=["r", "r_visium"]).reset_index(drop=True)
    per_descriptor = descriptor_table(cells)
    shown = shown_descriptors(per_descriptor)
    rho, hit, same = visium_agreement(cells)

    spots_total = int(cohort.spots.sum())
    a = [("participants", int(cohort.participants.sum())),
         ("slides", int(cohort.ffpe_slides.sum() + cohort.frozen_slides.sum())),
         ("ffpe_slides", int(cohort.ffpe_slides.sum())), ("frozen_slides", int(cohort.frozen_slides.sum())),
         ("spots", spots_total), ("spots_ffpe", stats["spots_ffpe"]), ("spots_frozen", stats["spots_frozen"]),
         ("genes", stats["genes"]), ("expression_values", spots_total * stats["genes"])]
    a += [(k, stats[k]) for k in ("tissue_mm2", "tissue_mm2_ffpe", "sections", "participants_with_baseline_egfr",
                                  "participants_with_fibrosis_score", "participants_with_descriptors",
                                  "xenium_participants", "xenium_slides", "participants_ffpe",
                                  "participants_frozen")]
    b = [("gt450_slides", len(slides)), ("gt450_participants", int(slides.participant.nunique())),
         ("gt450_spots", int(slides.spots.sum())), ("umap_spots", len(spots["niche"]))]
    b += [(f"{r.niche} {r.name} (% of GT450 spots)", f"{r.share * 100:.1f}") for r in niches.itertuples()]
    b += [(f"example {r.band}", f"{r.participant} {r.slide_id} eGFR {r.egfr:.0f} IF {r.if_pct:.0f} % unit "
                                f"{r.scanner_unit}") for r in examples.itertuples()]
    c = [(f"{r.niche} tiles", f"candidates {r.candidates} ({r.candidate_participants} participants), decile "
                              f"{r.decile} ({r.decile_participants} participants), shown {r.shown}; rule: {r.rule}")
         for r in selection.itertuples()]
    w = programs.pivot_table(index=["program", "variable"], columns="stain",
                             values=["effect_adjusted", "fdr_shown"])
    both = w[(w[("fdr_shown", "FFPE")] < 0.05) & (w[("fdr_shown", "frozen")] < 0.05)
             & (np.sign(w[("effect_adjusted", "FFPE")]) == np.sign(w[("effect_adjusted", "frozen")]))]
    n = programs.groupby(["stain", "variable"]).n.first()
    dd = [("cells", int(w[("effect_adjusted", "FFPE")].notna().sum())),
          ("ffpe_cells_fdr_below_0.05", int((w[("fdr_shown", "FFPE")] < 0.05).sum())),
          ("frozen_cells_fdr_below_0.05", int((w[("fdr_shown", "frozen")] < 0.05).sum())),
          ("cells_both_stains_same_sign", len(both)),
          ("participants_egfr", str((int(n["FFPE", "egfr"]), int(n["frozen", "egfr"])))),
          ("participants_fibrosis", str((int(n["FFPE", "C_if_pct"]), int(n["frozen", "C_if_pct"]))))]
    e = [("cells", len(cells)), ("spearman", rho), ("atlas_cells_fdr_below_0.05", int(hit.sum())),
         ("same_sign", same), ("visium_participants_median", float(cells.n_visium.median())),
         ("descriptors", len(per_descriptor)), ("median_spearman_per_descriptor",
                                                 float(per_descriptor.spearman.median())),
         ("positive_descriptors", int((per_descriptor.spearman > 0).sum())), ("shown_descriptors", len(shown))]
    xn = pd.read_csv(d / "xenium_niches.csv")
    f = []
    for xcol, ycol, title, chips, _ in F_SETS:
        r, p = spearmanr(xn[xcol], xn[ycol])
        f += [(f"{title} x", xcol), (f"{title} y", ycol), (f"{title} niches", str(chips)),
              (f"{title} rho", float(r)), (f"{title} p", float(p)), (f"{title} n", len(xn))]
    xs = pd.read_csv(d / "xenium_sections.csv")
    sections = xs[["section", "unadapted", "adapted", "panel"]].assign(genes="heldout_chromosome_split",
                                                                       metric="pearson_knn5")
    example = xenium_example(d)
    ex = xs.set_index("section").loc[example.section.iloc[0]]
    pan = sections.groupby("panel")[["unadapted", "adapted"]].median()
    g = [("heldout_genes", stats["xenium_heldout_genes"]), ("sections", len(sections)),
         ("median_r_before", float(sections.unadapted.median())), ("median_r_after", float(sections.adapted.median())),
         ("sections_improved", int((sections.adapted > sections.unadapted).sum())),
         ("wilcoxon_p", float(wilcoxon(sections.adapted, sections.unadapted).pvalue))]
    g += [(f"median_r_{lab}_{q}", float(pan.at[q, k])) for q in pan.index
          for k, lab in (("unadapted", "before"), ("adapted", "after"))]
    g += [(f"sections_{q}", int((sections.panel == q).sum())) for q in pan.index]
    g += [("example_gene", G_GENE), ("example_section", ex.name), ("example_panel", ex.panel),
          ("example_r_before", float(ex[f"unadapted_{G_GENE}"])), ("example_r_after", float(ex[f"adapted_{G_GENE}"])),
          ("example_cells", len(example)),
          ("example_gene_median_r_before", float(xs[f"unadapted_{G_GENE}"].median())),
          ("example_gene_median_r_after", float(xs[f"adapted_{G_GENE}"].median()))]
    summary = pd.DataFrame([(p, k, v) for p, rows in zip("ABCDEFG", (a, b, c, dd, e, f, g)) for k, v in rows],
                           columns=["panel", "quantity", "value"])
    return dict(summary=summary, cohort=cohort, niches=niches, example_biopsies=examples, niche_tiles=tiles,
                program_associations=programs, visium_cells=cells, descriptor_concordance=shown,
                xenium_niches=xn, xenium_sections=sections, xenium_example=example)


def niche_table(d, slides, umap_niche):
    nt = pd.read_csv(d / "niches.csv").set_index("niche").loc[list(NICHE_COLOR)]
    share = slides[list(NICHE_COLOR)].sum() / slides.spots.sum()
    n_umap = pd.Series(umap_niche).value_counts()
    return pd.DataFrame({"order": range(len(nt)), "niche": nt.index, "name": nt["name"].to_numpy(),
                         "on_umap": [n not in QUALITY for n in nt.index], "color": list(NICHE_COLOR.values()),
                         "quality": [n in QUALITY for n in nt.index], "share": share.to_numpy(),
                         "markers": nt.markers.to_numpy(),
                         "umap_spots": [int(n_umap.get(int(n[1:]), 0)) for n in nt.index]})


def example_table(d, slides):
    ex = pd.read_csv(d / "example_biopsies.csv")
    lat = load_lattice(d)
    sl = slides.set_index("slide_id").loc[ex.slide_id]
    out = ex.assign(scanner_unit=sl.scanner_unit.to_numpy(), spots=sl.spots.to_numpy(),
                    piece_spots=ex.participant.map(lat.participant.value_counts()).to_numpy())
    for n in NICHE_IDS:
        out[f"share_{n}"] = sl[n].to_numpy() / sl.spots.to_numpy()
    for n in NICHE_IDS:
        out[f"piece_share_{n}"] = [float((lat.niche[lat.participant == p] == int(n[1:])).mean())
                                   for p in ex.participant]
    return out


def tile_table(d, niches, selection):
    t = pd.read_csv(d / "niche_tiles.csv").set_index(["niche", "rank"])
    rule = selection.set_index("niche").rule
    meta = niches.set_index("niche")
    rows = []
    for c, col in enumerate(C_COLS):
        for k, nid in enumerate(col):
            items = [(nid, r) for r in range(8)] if nid else quality_items(niches.niche)
            for j, (n, r) in enumerate(items):
                rows.append(dict(niche=n, name=meta.loc[n, "name"], column=c, row=k, position=j, rank=r,
                                 **t.loc[(n, r)].to_dict(), rule=rule[n],
                                 markers="" if n in QUALITY else " ".join(c_markers(meta.loc[n, "markers"])),
                                 share=meta.loc[n, "share"]))
    return pd.DataFrame(rows).infer_objects()


def program_table(d):
    a = pd.read_csv(d / "program_associations.csv")
    a["fdr_marginal"] = a.groupby("stain").p_marginal.transform(bh)
    a["fdr_all_programs"] = a.groupby("stain").p_adjusted.transform(bh)
    s = a[a.program.isin(D_PROGS) & a.variable.isin(D_VARS)].copy()
    s["fdr_shown"] = s.groupby("stain").p_adjusted.transform(bh)
    return s[["stain", "program", "variable", "label", "group", "n", "groups", "effect_marginal", "p_marginal",
              "fdr_marginal", "effect_adjusted", "p_adjusted", "fdr_all_programs", "fdr_shown"]].reset_index(
        drop=True)


def visium_agreement(cells):
    hit = (cells.fdr < 0.05).to_numpy()
    same = int((np.sign(cells.r[hit]) == np.sign(cells.r_visium[hit])).sum())
    return float(spearmanr(cells.r, cells.r_visium).statistic), hit, same


def descriptor_table(cells):
    rows = []
    for desc, c in cells.groupby("descriptor"):
        hit = (c.fdr < 0.05).to_numpy()
        rows.append(dict(descriptor=desc, cells=len(c), spearman=float(spearmanr(c.r, c.r_visium).statistic),
                         atlas_hits=int(hit.sum()), visium_hits=int((c.fdr_visium < 0.05).sum()),
                         sign_agree_hits=int((np.sign(c.r[hit]) == np.sign(c.r_visium[hit])).sum())))
    return pd.DataFrame(rows)


def shown_descriptors(per_descriptor):
    d = per_descriptor[per_descriptor.atlas_hits >= E_MIN_HITS]
    d = d.sort_values(["atlas_hits", "spearman"], ascending=False).head(E_MAX)
    d = d.sort_values("spearman", ascending=False).reset_index(drop=True)
    return d.assign(label=d.descriptor.map(descriptor_label))


def text_mm(s, size):
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    return TextPath((0, 0), s, prop=FontProperties(size=size)).get_extents().width * 25.4 / 72


def wrap_mm(s, size, width, first=None):
    """Fewest lines of `s` within `width` mm (first line: `first` mm), preferring breaks after '·', ':' or ','."""
    w, first = s.split(), width if first is None else first
    for k in range(len(w)):
        fits = []
        for cut in itertools.combinations(range(1, len(w)), k):
            lines = [" ".join(w[a:b]) for a, b in zip((0,) + cut, cut + (len(w),))]
            ws = [text_mm(t, size) for t in lines]
            if ws[0] <= first and all(x <= width for x in ws[1:]):
                fits.append(((sum(not w[c - 1].endswith(("·", ":", ",")) for c in cut), max(ws)), lines))
        if fits:
            return min(fits)[1]
    return w


class Page:
    """A figure laid out in millimetres from its top-left corner."""

    def __init__(self, height):
        import matplotlib.pyplot as plt
        self.h = height
        self.fig = plt.figure(figsize=(W_MM * style.MM, height * style.MM))

    def x(self, mm):
        return mm / W_MM

    def y(self, mm):
        return 1 - mm / self.h

    def axes(self, x, y, w, h):
        return self.fig.add_axes([x / W_MM, 1 - (y + h) / self.h, w / W_MM, h / self.h])

    def text(self, x, y, s, **kw):
        return self.fig.text(self.x(x), self.y(y), s, **kw)

    def rect(self, x, y, w, h, color, rounding=None):
        from matplotlib.patches import FancyBboxPatch, Rectangle
        if rounding is None:
            p = Rectangle((self.x(x), self.y(y + h)), w / W_MM, h / self.h, fc=color, ec="none")
        else:
            p = FancyBboxPatch((self.x(x), self.y(y + h)), w / W_MM, h / self.h, fc=color, ec="none",
                               boxstyle=f"round,pad=0,rounding_size={rounding}")
        p.set_transform(self.fig.transFigure)
        self.fig.add_artist(p)

    def hline(self, x0, x1, y, **kw):
        from matplotlib.lines import Line2D
        self.fig.add_artist(Line2D([self.x(x0), self.x(x1)], [self.y(y)] * 2, **kw))

    def header(self, x, y, letter, title):
        style.header(self.fig, self.x(x), self.y(y), letter, title)


def clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def wedge_angles(sizes, start=90.0):
    frac = np.asarray(sizes, float) / np.sum(sizes)
    ends = start - 360 * np.cumsum(frac)
    return [(e, b) for b, e in zip(np.r_[start, ends[:-1]], ends)]


def draw_donut(ax, df):
    """Outer ring diagnosis, middle ring enrollment group, inner ring stain (FFPE | frozen slides)."""
    from matplotlib.patches import Wedge
    ang = wedge_angles(np.ones(len(df)))
    tint, dark = {}, {}
    for g, sub in df.groupby("group", sort=False):
        for k, i in enumerate(sub.index):
            tint[i] = mix(GROUP_COLOR[g], PANEL, [0, 28, 48, 64][k])
            dark[i] = k == 0 and g in ("CKD", "AKI")
    R_OUT, W_OUT, R_MID, W_MID, R_IN, W_IN = 1.0, 0.24, 0.74, 0.10, 0.62, 0.07
    for i, (t1, t2) in enumerate(ang):
        row = df.loc[i]
        ax.add_patch(Wedge((0, 0), R_OUT, t1, t2, width=W_OUT, fc=tint[i], ec=PANEL, lw=1.0))
        tm = t2 - row.ffpe_slides / max(row.ffpe_slides + row.frozen_slides, 1) * (t2 - t1)
        ax.add_patch(Wedge((0, 0), R_IN, tm, t2, width=W_IN, fc=STAIN_COLOR["FFPE"], ec=PANEL, lw=1.0))
        if row.frozen_slides:
            ax.add_patch(Wedge((0, 0), R_IN, t1, tm, width=W_IN, fc=STAIN_COLOR["frozen"], ec=PANEL, lw=1.0))
        a = np.deg2rad((t1 + t2) / 2)
        rot = np.rad2deg(a) - 90 + (180 if np.sin(a) < 0 else 0)
        ax.text(0.88 * np.cos(a), 0.88 * np.sin(a), f"{row.spots / 1e6:.2f}M\nspots", ha="center", va="center",
                rotation=rot, rotation_mode="anchor", fontsize=3.9, linespacing=0.95,
                color=PANEL if dark[i] else INK)
    for g, sub in df.groupby("group", sort=False):
        t1, t2 = ang[sub.index[-1]][0], ang[sub.index[0]][1]
        ax.add_patch(Wedge((0, 0), R_MID, t1, t2, width=W_MID, fc=GROUP_COLOR[g], ec=PANEL, lw=1.0))
    mids = [np.deg2rad((t1 + t2) / 2) for t1, t2 in ang]
    LX = 1.24
    for side in (1, -1):
        mem = sorted([i for i, a in enumerate(mids) if (np.cos(a) >= 0) == (side == 1)],
                     key=lambda i: np.sin(mids[i]), reverse=True)
        ys = np.linspace(0.92, -0.92, len(mem)) if len(mem) > 1 else [np.sin(mids[mem[0]])]
        for slot, i in enumerate(mem):
            xw, yw = np.cos(mids[i]), np.sin(mids[i])
            ax.plot([xw, 1.07 * xw, LX * side], [yw, 1.07 * yw, ys[slot]], color=RULE, lw=1.0)
            ax.text(LX * side + 0.04 * side, ys[slot], f"{df.diagnosis[i]} ({df.participants[i]})",
                    ha="left" if side == 1 else "right", va="center", fontsize=5, color=INK)
    slides = int(df.ffpe_slides.sum() + df.frozen_slides.sum())
    ax.text(0, 0.17, f"{int(df.participants.sum())}", ha="center", va="center", fontsize=7.5, fontweight="bold",
            color=INK)
    ax.text(0, 0.0, "participants", ha="center", va="center", fontsize=4.6, color=MUTED)
    ax.text(0, -0.2, f"{style.thou(slides)} slides\n{df.spots.sum() / 1e6:.2f} M spots", ha="center", va="center",
            fontsize=4.6, color=INK, linespacing=1.25)
    ax.set_aspect("equal")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-1.06, 1.06)
    ax.axis("off")


def donut_legend(page, x, y, df):
    from matplotlib.patches import Patch
    hs = ([Patch(fc=GROUP_COLOR[g], ec="none") for g in ORDER] + [Patch(fc="none", ec="none")]
          + [Patch(fc=STAIN_COLOR[k], ec="none") for k in ("FFPE", "frozen")])
    lab = ([f"{GROUP_LABEL[g]} {style.thou(df[df.group == g].participants.sum())}" for g in ORDER] + [""]
           + [f"FFPE {style.thou(df.ffpe_slides.sum())}", f"frozen {style.thou(df.frozen_slides.sum())} slides"])
    page.fig.legend(hs, lab, loc="upper center", bbox_to_anchor=(page.x(x), page.y(y)), ncol=len(hs), fontsize=5,
                    frameon=False, handlelength=0.8, handleheight=0.8, columnspacing=0.8, handletextpad=0.3,
                    borderaxespad=0, borderpad=0)


def stat_tiles(page, x, y, w, h, stats, gap):
    th = (h - gap * (len(stats) - 1)) / len(stats)
    for k, (val, cap) in enumerate(stats):
        top = y + k * (th + gap)
        page.rect(x, top, w, th, mix(HAIR, PANEL, 35), rounding=0.004)
        page.text(x + 1.6, top + 1.3, val, fontsize=6.5, fontweight="bold", color=INK, ha="left", va="top")
        page.text(x + 1.6, top + th - 1.0, cap, fontsize=4.8, color=MUTED, ha="left", va="bottom",
                  linespacing=1.1)


def panel_a(page, cohort, summary):
    st = summary[summary.panel == "A"].set_index("quantity").value.astype(float)
    draw_donut(page.axes(0, 8, 78, 52), cohort)
    donut_legend(page, 39, 62.2, cohort)
    thou = style.thou
    stats = [(f"{thou(st['tissue_mm2'])} mm²", f"H&E tissue in\n{thou(st['sections'])} sections"),
             (thou(st["genes"]), "genes predicted\nper virtual spot"),
             (f"{st['expression_values'] / 1e11:.2f} × 10¹¹", "virtual expression\nvalues"),
             (thou(st["participants_with_baseline_egfr"]), "non-AKI participants\nwith baseline eGFR"),
             (thou(st["participants_with_fibrosis_score"]), "with pathologist\ndescriptors")]
    stat_tiles(page, 78.5, 8.5, 25, 55, stats, gap=(55 - 5 * 50 / 6) / 4)


def umap_arrows(ax, xy):
    x0, y0 = xy[:, 0].min(), xy[:, 1].min()
    dx, dy = np.ptp(xy[:, 0]) * 0.14, np.ptp(xy[:, 1]) * 0.14
    kw = dict(arrowprops=dict(arrowstyle="-|>,head_length=0.25,head_width=0.12", lw=1.0, color=MUTED))
    ax.annotate("", (x0 + dx, y0), (x0, y0), **kw)
    ax.annotate("", (x0, y0 + dy), (x0, y0), **kw)
    ax.text(x0 + dx / 2, y0 - np.ptp(xy[:, 1]) * 0.02, "UMAP 1", ha="center", va="top", fontsize=4.5, color=MUTED)
    ax.text(x0 - np.ptp(xy[:, 0]) * 0.02, y0 + dy / 2, "UMAP 2", ha="right", va="center", rotation=90,
            fontsize=4.5, color=MUTED)


def umap_limits(xy):
    return ((xy[:, 0].min() - np.ptp(xy[:, 0]) * 0.05, xy[:, 0].max() + np.ptp(xy[:, 0]) * 0.02),
            (xy[:, 1].min() - np.ptp(xy[:, 1]) * 0.07, xy[:, 1].max() + np.ptp(xy[:, 1]) * 0.02))


def label_spots(xy, lab, ids, bins=40, min_dy=0.055):
    """Label position per niche: the UMAP bin where it is most dominant, close labels pushed apart vertically."""
    ex_ = np.linspace(xy[:, 0].min(), xy[:, 0].max(), bins + 1)
    ey_ = np.linspace(xy[:, 1].min(), xy[:, 1].max(), bins + 1)
    tot = np.histogram2d(xy[:, 0], xy[:, 1], [ex_, ey_])[0]
    pos = []
    for nid in ids:
        k = lab == nid
        h = np.histogram2d(xy[k, 0], xy[k, 1], [ex_, ey_])[0]
        score = np.where(tot > 0, h / np.maximum(tot, 1), 0) * np.sqrt(h)
        i, j = np.unravel_index(np.argmax(score), score.shape)
        pos.append([(ex_[i] + ex_[i + 1]) / 2, (ey_[j] + ey_[j + 1]) / 2])
    pos = np.array(pos)
    yr, xr = np.ptp(xy[:, 1]), np.ptp(xy[:, 0])
    for _ in range(50):
        moved = False
        for a in range(len(pos)):
            for b in range(a + 1, len(pos)):
                if abs(pos[a, 0] - pos[b, 0]) < 0.22 * xr and abs(pos[a, 1] - pos[b, 1]) < min_dy * yr:
                    s = np.sign(pos[a, 1] - pos[b, 1]) or 1.0
                    pos[a, 1] += s * 0.01 * yr
                    pos[b, 1] -= s * 0.01 * yr
                    moved = True
        if not moved:
            break
    return pos


def panel_b_umap(page, nt, spots):
    import matplotlib.patheffects as pe
    xy = spots["umap"].astype(float)
    niche = spots["niche"].astype(int)
    col_of = dict(zip(nt.niche.str[1:].astype(int), nt.color))
    o = np.random.default_rng(0).permutation(len(xy))
    ax = page.axes(B_X + 3, 11, W_MM - B_X - 4, 52)
    ax.scatter(xy[o, 0], xy[o, 1], s=0.12, c=pd.Series(niche).map(col_of).to_numpy()[o], lw=0, rasterized=True)
    lab = nt[nt.on_umap]
    xl, yl = umap_limits(xy)
    x_per_mm = np.ptp(xl) / (W_MM - B_X - 4)
    for (cx, cy), r in zip(label_spots(xy, niche, lab.niche.str[1:].astype(int)), lab.itertuples()):
        lines = wrap_mm(r.name, 5.2, UMAP_LABEL_MM)
        half = (max(text_mm(s, 5.2) for s in lines) / 2 + 0.5) * x_per_mm
        ax.text(np.clip(cx, xl[0] + half, xl[1] - half), cy, "\n".join(lines), ha="center", va="center", fontsize=5.2,
                color=INK, linespacing=1.0, path_effects=[pe.withStroke(linewidth=1.6, foreground=PANEL)])
    clean(ax)
    umap_arrows(ax, xy)
    ax.set_xlim(xl)
    ax.set_ylim(yl)


def niche_grid(t, index):
    """Niche index grid of one tissue piece (-1 = no tissue), rotated to put its long axis horizontal."""
    xy = np.c_[t.col, t.row].astype(float)
    v = np.linalg.eigh(np.cov((xy - xy.mean(0)).T))[1][:, -1]
    deg = (float(np.degrees(np.arctan2(v[1], v[0]))) + 90) % 180 - 90
    r0, c0 = t.row.min(), t.col.min()
    g = np.zeros((t.row.max() - r0 + 1, t.col.max() - c0 + 1), np.float32)
    g[t.row.to_numpy() - r0, t.col.to_numpy() - c0] = t.niche.map(index).to_numpy(float) + 1
    g = np.asarray(Image.fromarray(g).rotate(deg, resample=Image.Resampling.NEAREST, expand=True, fillcolor=0.0))
    rr, cc = np.where(g > 0)
    return g[rr.min():rr.max() + 1, cc.min():cc.max() + 1].astype(int) - 1


def biopsy_views(d, ex, nt):
    lat = load_lattice(d)
    index = {int(n[1:]): k for k, n in enumerate(nt.niche)}
    return [(np.asarray(Image.open(d / f"biopsy_{r.band}.png").convert("RGBA")),
             niche_grid(lat[lat.participant == r.participant], index)) for r in ex.itertuples()]


def biopsy_layout(views):
    """Shared scale (mm per lattice step) and column widths of the example biopsies."""
    widths = np.array([g.shape[1] for _, g in views], float)
    scale = min(BX_HMAX / max(g.shape[0] for _, g in views), (BX_AVAIL - 2 * BX_GAP) / widths.sum())
    while np.maximum(widths * scale, BX_MINW).sum() + 2 * BX_GAP > BX_AVAIL:
        scale *= 0.98
    return scale, np.maximum(widths * scale, BX_MINW)


def panel_b_examples(page, nt, ex, views, scale, cols):
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(nt.color.tolist())
    for k, ((he, g), r) in enumerate(zip(views, ex.itertuples())):
        x0 = 2 + cols[:k].sum() + k * BX_GAP
        page.text(x0, R2 + 2.2, f"{BAND_LABEL[r.band]} (IF {r.if_pct:.0f} %)", fontsize=5.4, color=INK,
                  ha="left", va="baseline")
        page.text(x0, R2 + 4.9, f"{r.participant} · {r.diagnosis} · eGFR {r.egfr:.0f}", fontsize=4.8,
                  color=MUTED, ha="left", va="baseline")
        hpx, wpx = g.shape
        w, h = wpx * scale, hpx * scale
        a1 = page.axes(x0, R2 + 6.2, w, h)
        a1.imshow(he, interpolation="lanczos", extent=(0, wpx, hpx, 0), aspect="auto")
        a2 = page.axes(x0, R2 + 6.2 + h + 1.0, w, h)
        a2.imshow(np.ma.masked_less(g, 0), cmap=cmap, vmin=-0.5, vmax=len(nt) - 0.5, interpolation="nearest",
                  aspect="auto")
        for a in (a1, a2):
            clean(a)
            a.set_facecolor(PANEL)
    yb = R2 + 6.2 + 2 * views[0][1].shape[0] * scale + 1.0 + 2.8
    bar = 1000 / LATTICE_UM * scale
    page.hline(2, 2 + bar, yb, color=INK, lw=1.0)
    page.text(2 + bar + 1.0, yb, "1 mm", ha="left", va="center", fontsize=4.5, color=INK)
    page.text(LEG_X, R2 + 2.2, "niche", fontsize=5.0, color=INK, ha="left", va="baseline")
    page.text(W_MM - 2, R2 + 2.2, "% of spots", fontsize=4.5, color=MUTED, ha="right", va="baseline")
    for k, r in enumerate(nt.itertuples()):
        yk = R2 + 5.4 + 2.55 * k + (0.9 if r.quality else 0)
        page.rect(LEG_X, yk - 1.45, 1.8, 1.8, r.color)
        page.text(LEG_X + 2.6, yk, r.name, fontsize=4.8, color=MUTED if r.quality else INK, ha="left",
                  va="baseline")
        page.text(W_MM - 2, yk, f"{r.share * 100:.0f}", fontsize=4.8, color=MUTED, ha="right", va="baseline")
    nq = int(nt.quality.sum())
    page.text(LEG_X, R2 + 5.4 + 2.55 * len(nt) + 1.4, "gray: image-quality niche" + ("s" if nq > 1 else ""),
              fontsize=4.5, color=MUTED, ha="left", va="baseline")


def c_tile_size():
    col_w = (W_MM - 4 - C_GUTTER) / 2
    return col_w, (col_w - C_LW - 7 * C_GAP) / 8


def c_height():
    rows = max(len(c) for c in C_COLS)
    return rows * c_tile_size()[1] + (rows - 1) * C_ROW_GAP


def panel_c(page, nt, tiles, d, top):
    import matplotlib.patheffects as pe
    mosaic = np.asarray(Image.open(d / "niche_tiles.png").convert("RGB"))
    px = mosaic.shape[1] // 8
    meta = nt.set_index("niche")
    col_w, T = c_tile_size()

    def pct(n):
        return f"{meta.loc[n, 'share'] * 100:.0f} %"

    def name_lines(n, size, x_text):
        return wrap_mm(meta.loc[n, "name"], size, C_LW - 0.7 - x_text,
                       first=C_LW - 0.7 - text_mm(pct(n), 4.4) - 0.6 - x_text)

    def strip(x, y, rows):
        for r in rows.itertuples():
            ax = page.axes(x + C_LW + r.position * (T + C_GAP), y, T, T)
            i = NICHE_IDS.index(r.niche)
            img = mosaic[i * px:(i + 1) * px, r.rank * px:(r.rank + 1) * px]
            ax.imshow(img, interpolation="none", aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_color(meta.loc[r.niche, "color"])
                s.set_linewidth(1.0)
            if r.position == len(rows) - 1:
                h_, w_ = img.shape[:2]
                L = C_BAR_UM / (r.crop_px * r.mpp / w_)
                ax.plot([w_ * 0.93 - L, w_ * 0.93], [h_ * 0.9] * 2, color=INK, lw=1.2, solid_capstyle="butt",
                        path_effects=[pe.withStroke(linewidth=2.6, foreground=PANEL)], zorder=5)

    lh = 2.05
    for c, col in enumerate(C_COLS):
        x = 2 + c * (col_w + C_GUTTER)
        for k, nid in enumerate(col):
            y = top + k * (T + C_ROW_GAP)
            rows = tiles[(tiles.column == c) & (tiles.row == k)]
            if nid is None:
                q = [n for n in nt.niche if n in QUALITY]
                names = [name_lines(n, 4.4, 2.0) for n in q]
                lh_ = min(lh, (T - 0.3) / (1 + sum(map(len, names))))
                yy = y + (T - (1 + sum(map(len, names))) * lh_) / 2 + 1.6
                page.text(x, yy, "Image-quality niche" + ("s" if len(q) > 1 else ""), fontsize=4.6, color=MUTED,
                          ha="left", va="baseline")
                for n, lines in zip(q, names):
                    yy += lh_
                    page.rect(x, yy - 1.25, 1.3, 1.3, meta.loc[n, "color"])
                    page.text(x + C_LW - 0.7, yy, pct(n), fontsize=4.4, color=MUTED, ha="right", va="baseline")
                    for i, line in enumerate(lines):
                        page.text(x + 2.0, yy + i * lh_, line, fontsize=4.4, color=MUTED, ha="left",
                                  va="baseline")
                    yy += (len(lines) - 1) * lh_
                strip(x, y, rows)
                continue
            page.rect(x, y, 1.1, T, meta.loc[nid, "color"])
            name = name_lines(nid, 5.0, 2.1)
            own = text_mm(name[0], 5.0) > C_LW - 0.7 - text_mm(pct(nid), 4.4) - 0.6 - 2.1
            n_lines = len(name) + 1 + own
            lh_ = min(lh, (T - 0.3) / n_lines)
            y0 = y + (T - n_lines * lh_) / 2 + 1.55
            for i, line in enumerate(name):
                page.text(x + 2.1, y0 + i * lh_, line, fontsize=5.0, color=INK, ha="left", va="baseline")
            page.text(x + (2.1 if own else C_LW - 0.7), y0 + len(name) * lh_ if own else y0, pct(nid),
                      fontsize=4.4, color=MUTED, ha="left" if own else "right", va="baseline")
            page.text(x + 2.1, y0 + (len(name) + own) * lh_, " ".join(c_markers(meta.loc[nid, "markers"])),
                      fontsize=4.3, fontstyle="italic", color=INK, ha="left", va="baseline")
            strip(x, y, rows)


def split_heatmap(ax, g):
    """Cells split FFPE (left) | frozen (right), coloured by partial r; ring = FDR < 0.05."""
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle
    import matplotlib.pyplot as plt
    cmap, norm = plt.get_cmap(R_CMAP), Normalize(-R_MAX, R_MAX)
    E, Q = {}, {}
    for s in ("FFPE", "frozen"):
        d = g[g.stain == s]
        E[s] = d.pivot(index="program", columns="variable", values="effect_adjusted").reindex(
            index=D_PROGS, columns=D_VARS).to_numpy(float)
        Q[s] = d.pivot(index="program", columns="variable", values="fdr_shown").reindex(
            index=D_PROGS, columns=D_VARS).to_numpy(float)
    rects, fcs, ring = [], [], []
    for i in range(len(D_PROGS)):
        for j in range(len(D_VARS)):
            for h, s in enumerate(("FFPE", "frozen")):
                v, q = E[s][i, j], Q[s][i, j]
                rects.append(Rectangle((j + 0.5 * h, i), 0.5, 1))
                fcs.append(cmap(norm(v)) if np.isfinite(v) else NA_FILL)
                if np.isfinite(q) and q < 0.05:
                    ring.append((j + 0.5 * h + 0.25, i + 0.5, abs(v) > 0.6 * R_MAX))
    ax.add_collection(PatchCollection(rects, facecolors=fcs, edgecolors=PANEL, linewidths=0))
    for j in range(len(D_VARS) + 1):
        ax.axvline(j, color=PANEL, lw=1.0)
    for i in range(len(D_PROGS) + 1):
        ax.axhline(i, color=PANEL, lw=1.0)
    if ring:
        rx, ry, dk = map(np.array, zip(*ring))
        for m, c in ((dk, PANEL), (~dk, INK)):
            ax.scatter(rx[m], ry[m], s=D_RING_S, facecolors="none", edgecolors=c, linewidths=1.0, zorder=4)
    b = 0
    for _, cs in D_COLUMNS[:-1]:
        b += len(cs)
        ax.axvline(b, color=PANEL, lw=2.2)
    ax.set_xlim(0, len(D_VARS))
    ax.set_ylim(len(D_PROGS), 0)
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(len(D_VARS)) + 0.5)
    ax.set_xticklabels([lab for _, cs in D_COLUMNS for _, lab in cs], rotation=55, ha="left", fontsize=4.8,
                       rotation_mode="anchor", color=INK)
    ax.set_yticks(np.arange(len(D_PROGS)) + 0.5)
    ax.set_yticklabels([PROG_LABEL[p] for p in D_PROGS], fontsize=4.8, color=INK)
    ax.tick_params(length=0, pad=1.2)
    for s in ax.spines.values():
        s.set_visible(False)
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap)


def d_key(page, x, top, yb, sm):
    from matplotlib.patches import Rectangle
    ka = page.axes(x, top - 9.2, 3.6, 2.8)
    for h in (0, 0.5):
        ka.add_patch(Rectangle((h, 0), 0.5, 1, fc=mix(INK, PANEL, 80 if h == 0 else 62), ec="none", lw=0))
    ka.scatter([0.25], [0.5], s=D_RING_S, facecolors="none", edgecolors=INK, linewidths=1.0)
    ka.set_xlim(0, 1)
    ka.set_ylim(0, 1)
    ka.axis("off")
    page.text(x + 4.4, top - 7.8, "FFPE | frozen\n○ FDR < 0.05", fontsize=4.6, color=INK, ha="left", va="center",
              linespacing=1.3)
    page.text(x, top - 3.6, "frozen = same participants,", fontsize=4.4, color=MUTED, ha="left", va="baseline")
    page.text(x, top - 1.7, "other tissue preparation", fontsize=4.4, color=MUTED, ha="left", va="baseline")
    cb = page.fig.colorbar(sm, cax=page.axes(x + 6.5, yb + 1.5, 12, 1.3), orientation="horizontal")
    cb.set_ticks([-R_MAX, 0, R_MAX])
    cb.set_ticklabels([f"−{R_MAX:g}", "0", f"+{R_MAX:g}"])
    cb.ax.tick_params(labelsize=4.4, length=1, pad=0.8)
    cb.outline.set_visible(False)
    page.text(x + 5.8, yb + 2.7, "partial r", fontsize=4.6, color=INK, ha="right", va="baseline")


def panel_d(page, g, top, X0=24, CW=5.4, RH=3.6):
    ax = page.axes(X0, top, CW * len(D_VARS), RH * len(D_PROGS))
    sm = split_heatmap(ax, g)
    k = 0
    for _, ps in D_BLOCKS[:-1]:
        k += len(ps)
        ax.axhline(k, color=PANEL, lw=2.2)
    x, yb = X0, top + RH * len(D_PROGS)
    for name, cs in D_COLUMNS:
        w = CW * len(cs)
        if name:
            page.hline(x + 0.4, x + w - 0.4, yb + 1.0, color=RULE, lw=1.0)
            page.text(x + w / 2, yb + 3.2, name, ha="center", va="baseline", fontsize=4.8, color=MUTED)
        x += w
    d_key(page, 3, top, yb, sm)


def panel_e(page, cells, shown, top, X0, W, LX, DW, DH):
    from matplotlib.lines import Line2D
    rho, hit, same = visium_agreement(cells)
    ax = page.axes(X0, top, W, W)
    lim = float(np.ceil(np.nanmax(np.abs(cells[["r", "r_visium"]].to_numpy())) * 10) / 10)
    ax.plot([-lim, lim], [-lim, lim], color=HAIR, lw=1.0, zorder=1)
    ax.axhline(0, color=HAIR, lw=1.0, zorder=1)
    ax.axvline(0, color=HAIR, lw=1.0, zorder=1)
    other = mix(RULE, PANEL, 50)
    ax.scatter(cells.r[~hit], cells.r_visium[~hit], s=1.4, c=other, lw=0, rasterized=True, zorder=2)
    ax.scatter(cells.r[hit], cells.r_visium[hit], s=2.0, c=HEMATOXYLIN, lw=0, rasterized=True, zorder=3)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Atlas partial r (H&E, FFPE)", fontsize=5.2, labelpad=1.5)
    ax.set_ylabel("Measured partial r (10x Visium)", fontsize=5.2, labelpad=1.0)
    ax.tick_params(labelsize=4.6)
    ax.text(0.03, 0.97, f"{len(cells)} descriptor ×\nprogram cells\nρ = {rho:.2f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=4.6, color=INK, linespacing=1.2)
    ax.legend([Line2D([], [], marker="o", ls="", ms=2.0, mfc=HEMATOXYLIN, mec="none"),
               Line2D([], [], marker="o", ls="", ms=2.0, mfc=other, mec="none")],
              [f"atlas FDR < 0.05 ({hit.sum()};\n{same} same sign)", "other"], loc="lower right", fontsize=4.3,
              handletextpad=0.1, borderaxespad=0.2, labelspacing=0.3)
    bx = page.axes(LX, top, DW, DH)
    y = np.arange(len(shown))
    bx.axvline(0, color=HAIR, lw=1.0, zorder=1)
    bx.hlines(y, 0, shown.spearman, color=mix(RULE, PANEL, 55), lw=1.0, zorder=2)
    bx.scatter(shown.spearman, y, s=4, c=HEMATOXYLIN, lw=0, zorder=3)
    bx.set_ylim(len(shown) - 0.5, -0.5)
    bx.set_xlim(-0.2, 1.0)
    bx.set_xticks([0, 0.5, 1])
    bx.set_xticklabels(["0", "0.5", "1"])
    bx.set_yticks(y)
    bx.set_yticklabels(shown.label, fontsize=4.4, color=INK)
    bx.tick_params(axis="y", length=0, pad=1.2)
    bx.tick_params(axis="x", labelsize=4.6)
    bx.spines["left"].set_visible(False)
    bx.set_xlabel("ρ per descriptor", fontsize=5.0, labelpad=1.5)
    page.text(LX + DW, top + DH + 8.6, f"{len(shown)} descriptors with the most atlas hits (≥ {E_MIN_HITS}) shown",
              fontsize=4.4, color=MUTED, ha="right", va="top", linespacing=1.2)
    page.text(X0 - 6, top - 2.4, "pathologist descriptor × program partial r; Visium biopsies from other "
              f"participants (median n {cells.n_visium.median():.0f})", fontsize=4.5, color=MUTED, ha="left",
              va="baseline")


def chip_title(page, x, y, niches, text):
    for j, n in enumerate(niches):
        page.rect(x + j * 2.0, y - 0.8, 1.6, 1.6, NICHE_COLOR[n])
    page.text(x + len(niches) * 2.0 + 0.5, y, text, fontsize=5.4, color=INK, ha="left", va="center")


def panel_f(page, xn, top):
    for k, (xcol, ycol, title, chips, col) in enumerate(F_SETS):
        x0 = F_X[k]
        chip_title(page, x0, top + F_TOP - 3.4, [col], title)
        ax = page.axes(x0, top + F_TOP, F_W, F_H)
        ax.scatter(xn[xcol], xn[ycol], s=6, color=NICHE_COLOR[col], lw=0, zorder=2)
        b, a = np.polyfit(xn[xcol], xn[ycol], 1)
        xl = np.array([xn[xcol].min(), xn[xcol].max()])
        ax.plot(xl, a + b * xl, color=NICHE_COLOR[col], lw=0.9, zorder=1.5, solid_capstyle="butt")
        ax.set_xlim(0, F_LIM[0])
        ax.set_ylim(0, F_LIM[1])
        ax.set_xticks([0, 40, 80])
        ax.set_yticks([0, 20, 40])
        ax.tick_params(labelsize=FS_TK)
        ax.set_xlabel("Atlas PT niches (% of spots)" if len(chips) > 1 else "Atlas niche (% of spots)",
                      fontsize=FS_AX, labelpad=1.5)
        ax.set_ylabel("Xenium (% of cells)", fontsize=FS_AX, labelpad=1.5)
        r = spearmanr(xn[xcol], xn[ycol]).statistic
        ax.text(0.04, 0.98, f"ρ = {r:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=FS_NOTE, color=INK)
    page.text(F_X[1] + F_W, top + F_TOP + F_H + 7.4, f"{len(xn)} participants", fontsize=FS_NOTE, color=MUTED,
              ha="right", va="top")


def horizontal(x, y):
    """Cell coordinates turned by 90 degrees when the window is taller than wide, shifted to start at 0."""
    if np.ptp(y) > np.ptp(x):
        x, y = y, x.max() - x
    return x - x.min(), y - y.min()


def cell_map(ax, x, y, v, w_mm):
    """Cells as CELL_UM dots at their centroids on a 2-98 % colour scale, the highest on top."""
    lo, hi = np.nanpercentile(v, [2, 98])
    ppu = w_mm / 25.4 * 72 / (np.ptp(x) + 2 * MAP_PAD)
    o = np.argsort(v, kind="stable")
    ax.scatter(x[o], y[o], c=v[o], s=(CELL_UM * ppu) ** 2, marker="o", cmap=style.EXPR_CMAP, vmin=lo,
               vmax=hi if hi > lo else lo + 1e-6, linewidths=0, rasterized=True)
    ax.set_xlim(-MAP_PAD, np.ptp(x) + MAP_PAD)
    ax.set_ylim(np.ptp(y) + MAP_PAD, -MAP_PAD)
    ax.axis("off")


def panel_g(page, s, t, summary, top):
    from matplotlib.lines import Line2D
    p = float(wilcoxon(s.adapted, s.unadapted).pvalue)
    x0 = G_X
    page.text(x0 + 22.0, top + 5.0, "Genes not used in adaptation", fontsize=5.4, color=INK, ha="center",
              va="center")
    ax = page.axes(x0 + 12.0, top + 8.5, 20.0, 22.0)
    for u, a in zip(s.unadapted, s.adapted):
        ax.plot([0, 1], [u, a], color=LINK, lw=0.5, zorder=1)
    for xx, k, c in ((0, "unadapted", BEFORE), (1, "adapted", AFTER)):
        ax.boxplot([s[k]], positions=[xx], widths=0.36, whis=1.5, showfliers=False, patch_artist=True, zorder=1.6,
                   boxprops=dict(facecolor=mix(c, PANEL, 78), edgecolor=INK, lw=0.6),
                   medianprops=dict(color=INK, lw=1.2, solid_capstyle="butt"),
                   whiskerprops=dict(color=INK, lw=0.6), capprops=dict(color=INK, lw=0.6))
        ax.scatter(np.full(len(s), xx), s[k], s=2.0 ** 2, color=[XEN_PANEL[q][1] for q in s.panel], lw=0, zorder=2)
        m = float(s[k].median())
        ax.text(xx + (-0.26 if xx == 0 else 0.26), m, f"{m:.2f}", ha="right" if xx == 0 else "left", va="center",
                fontsize=FS_NOTE, color=INK)
    ax.legend([Line2D([], [], marker="o", ls="", ms=2.0, mfc=c, mec="none") for _, c in XEN_PANEL.values()],
              [lab for lab, _ in XEN_PANEL.values()], loc="upper left", fontsize=4.3, handletextpad=0.1,
              borderaxespad=0.2, labelspacing=0.3, frameon=False)
    ax.set_xlim(-0.75, 1.75)
    ax.set_ylim(min(0.0, float(s[["unadapted", "adapted"]].min().min()) - 0.03), 0.35)
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.set_xticks([0, 1], ["Before", "After"])
    ax.tick_params(axis="x", length=0, labelsize=FS_AX, labelcolor=INK, pad=2)
    ax.tick_params(axis="y", labelsize=FS_TK)
    ax.set_ylabel("Pearson correlation", fontsize=FS_AX, labelpad=1.5)
    ax.text(0.5, -0.14, "adaptation", transform=ax.transAxes, ha="center", va="top", fontsize=FS_AX, color=INK)
    ax.text(0.5, 1.0, style.fmt_p(p), transform=ax.transAxes, ha="center", va="bottom", fontsize=FS_NOTE, color=INK)
    page.text(x0 + 22.0, top + 8.5 + 22.0 + 7.4, f"{len(s)} participants not in training", fontsize=FS_NOTE,
              color=MUTED, ha="center", va="top")

    st = summary[summary.panel == "G"].set_index("quantity").value
    xs, ys = t.x_map_um.to_numpy(), t.y_map_um.to_numpy()
    mw, mh = MAP_W, MAP_W / ((np.ptp(xs) + 2 * MAP_PAD) / (np.ptp(ys) + 2 * MAP_PAD))
    mx = x0 + 52.5
    page.text(mx, top + 5.0, f"$\\mathit{{{G_GENE}}}$, not used in adaptation", fontsize=5.4, color=INK, ha="left",
              va="center")
    y = top + 8.5
    for k, lab, r in (("measured", "Measured", None), ("unadapted", "Before", st["example_r_before"]),
                      ("adapted", "After", st["example_r_after"])):
        ax = page.axes(mx, y, mw, mh)
        cell_map(ax, xs, ys, t[k].to_numpy(), mw)
        page.text(mx - 1.2, y + mh / 2, lab, fontsize=FS_AX, color=INK, ha="right", va="center")
        if r is not None:
            page.text(mx + mw + 1.0, y + mh / 2, f"r = {float(r):.2f}".replace("-", "−"), fontsize=FS_NOTE,
                      color=INK, ha="left", va="center")
        y += mh + MAP_GAP
    ppu = mw / (np.ptp(xs) + 2 * MAP_PAD)
    yb = y + 1.4
    page.hline(mx, mx + BAR_UM * ppu, yb, color=INK, lw=1.0)
    page.text(mx + BAR_UM * ppu + 1.0, yb, f"{BAR_UM} µm", ha="left", va="center", fontsize=4.5, color=INK)
    kx = mx + mw - 14.0
    cax = page.axes(kx, yb - 0.65, 10.0, 1.3)
    cax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=style.EXPR_CMAP)
    clean(cax)
    page.text(kx - 0.8, yb, "low", ha="right", va="center", fontsize=4.5, color=MUTED)
    page.text(kx + 10.8, yb, "high", ha="left", va="center", fontsize=4.5, color=MUTED)


def draw(tables, data_dir):
    d = Path(data_dir)
    nt = tables["niches"]
    ex = tables["example_biopsies"]
    views = biopsy_views(d, ex, nt)
    scale, cols = biopsy_layout(views)
    hmax = max(g.shape[0] for _, g in views) * scale
    rc = max(R2 + 6.2 + 2 * hmax + 1.0 + GAP + 1.5, R2 + 6.2 + 2 * views[0][1].shape[0] * scale + 1.0 + 2.8 + 6.5,
             R2 + 5.4 + 2.55 * len(nt) + 6.0)
    r4 = rc + c_height() + GAP
    r5 = r4 + ROW_DE_H
    page = Page(r5 + ROW_FG_H)
    panel_a(page, tables["cohort"], tables["summary"])
    page.header(LETTER, 5, "A", "The KPMP virtual spatial transcriptomics atlas")
    panel_b_umap(page, nt, load_spots(d))
    page.header(B_X, 5, "B", "Expression-derived tissue niches")
    panel_b_examples(page, nt, ex, views, scale, cols)
    panel_c(page, nt, tables["niche_tiles"], d, rc + 0.6)
    page.header(LETTER, rc - 1.5, "C", "Representative histology of tissue niches")
    panel_d(page, tables["program_associations"], r4 + 12.5)
    page.header(LETTER, r4, "D", "Molecular programs and kidney phenotype")
    panel_e(page, tables["visium_cells"], tables["descriptor_concordance"], r4 + 6.0, X0=92, W=34, LX=151, DW=26,
            DH=29)
    page.header(85, r4, "E", "Concordance with measured Visium associations")
    panel_f(page, tables["xenium_niches"], r5)
    page.header(LETTER, r5, "F", "Niche proportions against measured Xenium cells")
    panel_g(page, tables["xenium_sections"], tables["xenium_example"], tables["summary"], r5)
    page.header(G_X, r5, "G", "Virtual extension of Xenium panels")
    return page.fig
