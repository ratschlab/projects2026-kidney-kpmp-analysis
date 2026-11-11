"""Figure 2: accuracy of virtual spatial transcriptomics on held-out KPMP biopsies."""
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from scipy.spatial import cKDTree

import style

TOP_K = [100, 500, 1000, 5000, 10000, 19338]
TOP_K_SHOWN = [100, 1000, 5000, 19338]
TOP_K_LABEL = {100: "100", 500: "500", 1000: "1K", 5000: "5K", 10000: "10K", 19338: "all"}
MODELS_A = ["zero_shot", "squall", "stpath", "squall_ft", "stpath_ft", "kidney"]
SCORED_A = MODELS_A + ["group_specific"]
EXPOSURE = ["No KPMP data", "KPMP specimens in pretraining", "on the KPMP training biopsies"]
TRAINING = [("As released", 0, 2), ("Fine-tuned", 2, 3)]
FAMILY = [("DeepSpot-M", "kidney", {0: "zero_shot", 2: "kidney"}),
          ("STPath", "stpath_ft", {1: "stpath", 2: "stpath_ft"}),
          ("SQUALL", "squall_ft", {1: "squall", 2: "squall_ft"})]
KEY_COL_MM, KEY_ROW_MM = (17, 33, 33, 36), 2.6
CHANCE_LABEL = "Chance (spot permutation)"
CHANCE_COLOR = "#4c4f69"
CHANCE_DASH = (0, (1, 1.4))
RELEASED_DASH = (0, (3, 1.6))

R_CUTOFF = 0.3
MODELS_B = ["zero_shot", "kidney", "group_specific"]
MODELS_B_SHOWN = ["zero_shot", "kidney"]
COMPARTMENTS = [
    ("Epithelium", [("Proximal tubule", ["proximal_tubule", "healthy_pt"]),
                    ("PT gluconeogenesis", ["pt_gluconeogenesis"]),
                    ("Podocyte", ["podocyte"]),
                    ("Loop of Henle · TAL", ["loop_tal"]),
                    ("Distal · collecting duct", ["distal_cd"])]),
    ("Stroma", [("Vessel · smooth muscle", ["vascular_smc"]),
                ("Fibroblast", ["fibroblast"]),
                ("Myofibroblast", ["myofibroblast"])]),
    ("Injury", [("Injury · repair", ["injury_repair", "injury_other"])]),
    ("Immune", [("Myeloid · IFN-γ · T cell", ["immune", "ifng_ain"]),
                ("B cell · TLS", ["b_tls"])]),
]

MAP_ROWS = ["he", "measured", "squall_ft", "stpath_ft", "kidney"]
MAP_MODELS = ["squall_ft", "stpath_ft", "kidney"]
MAP_LABEL = {"he": "H&E image", "measured": "10x Visium\n(measured)", "squall_ft": "SQUALL\nfine-tuned",
             "stpath_ft": "STPath\nfine-tuned", "kidney": "DeepSpot-M\nfine-tuned"}
SCALEBAR_UM = 250
HEAD_IN, ROW_GAP_IN, FOOT_IN = 0.17, 0.13, 0.09
LABEL_PAD_IN, SWATCH_COL_IN, MAP_GAP_IN, MAP_MAX_H_IN = 0.06, 0.11, 0.1, 0.42

CONTRASTS = [("AKI", "DM-R"), ("CKD", "DM-R")]
Z_GROUPS = ["AKI", "CKD", "DM-R"]
A_GROUPS = ["Healthy_Reference", "CKD", "DM-R", "AKI"]
PROGRAMS = [
    ("Injury / maladaptive repair", [("VCAM1", {"AKI": 1}), ("HAVCR1", {"AKI": 1}),
                                     ("MMP7", {"AKI": 1, "CKD": 1}), ("CDH6", {"AKI": 1, "CKD": 1}),
                                     ("CDKN1A", {"AKI": 1, "CKD": 1})]),
    ("PT gluconeogenesis", [("PC", {"CKD": -1}), ("PCK1", {"CKD": -1}), ("FBP1", {"CKD": -1}),
                            ("G6PC1", {"CKD": -1})]),
    ("Fibroblast content", [("Fibroblast score", {"CKD": 1})]),
]
MARKERS = [m for _, ms in PROGRAMS for m, _ in ms]
SOURCES_D = ["measured", "zero_shot", "kidney", "group_specific"]
SOURCES_D_SHOWN = ["measured", "kidney"]
LABEL = {**style.MODEL_LABEL, "kidney": "DeepSpot-M fine-tuned"}

WIDTH_MM, LEFT_MM, RIGHT_MM, LETTER_MM, SPLIT_MM = 183, 11, 181, 2, 121
ROW1_MM, ROW3_MM, HEAD_MM, GAP_MM, KEY_MM = 47, 56, 7, 5, 13


def read_f32(path, columns):
    return pd.read_csv(path, dtype={c: np.float32 for c in columns}, float_precision="round_trip")


def topk_by_split(data_dir):
    acc = read_f32(data_dir / "gene_accuracy_by_split.csv.gz", SCORED_A)
    rows = []
    for (split, group), sub in acc.groupby(["split", "group"]):
        for m in SCORED_A:
            v = np.sort(sub[m].dropna().to_numpy(float))[::-1]
            cum = np.cumsum(v) / np.arange(1, len(v) + 1)
            rows += [{"split": split, "group": group, "model": m, "top_k": k, "r": cum[min(k, len(cum)) - 1]}
                     for k in TOP_K]
    return pd.DataFrame(rows)


def chance_by_split(data_dir):
    chance = pd.read_csv(data_dir / "chance_by_split.csv", float_precision="round_trip")
    return chance.assign(model="chance")[["split", "group", "model", "top_k", "r"]]


def program_accuracy(data_dir):
    acc = pd.read_csv(data_dir / "gene_accuracy.csv.gz", index_col="gene", float_precision="round_trip")
    sets = pd.read_csv(data_dir / "gene_sets.csv")
    n_above_cutoff = (acc >= R_CUTOFF).sum()
    rows = []
    for compartment, programs in COMPARTMENTS:
        for program, names in programs:
            genes = [g for g in sets[sets.set.isin(names)].gene if g in acc.index]
            rows += [{"compartment": compartment, "program": program, "model": m, "n_genes": len(genes),
                      "r": float(acc.loc[genes, m].mean()), "n_genes_r_ge_0_3": int(n_above_cutoff[m])}
                     for m in MODELS_B]
    return pd.DataFrame(rows)


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def moran(v, xy, k=6):
    """Moran's I of v on the k-nearest-spot graph."""
    from scipy.sparse import csr_matrix
    nb = cKDTree(xy).query(xy, k=k + 1)[1][:, 1:]
    w = csr_matrix((np.ones(nb.size), (np.repeat(np.arange(len(xy)), k), nb.ravel())), shape=(len(xy),) * 2)
    z = v - v.mean()
    d = (z ** 2).sum()
    return float(len(v) / w.sum() * (z @ (w @ z)) / d) if d > 0 else np.nan


def read_spots(data_dir):
    cols = ["x", "y"] + [f"{m}{s}" for m in ["measured"] + MAP_MODELS for s in ("", "_smoothed")]
    return read_f32(data_dir / "shown_biopsy_spots.csv", cols)


def shown_biopsies(data_dir):
    shown = pd.read_csv(data_dir / "shown_biopsies.csv")
    spots = read_spots(data_dir)
    rows = []
    for _, b in shown.iterrows():
        d = spots[spots.biopsy == b.biopsy]
        gt = np.ascontiguousarray(d.measured_smoothed.to_numpy(np.float32))
        r = {}
        for m in MAP_MODELS:
            pred = np.ascontiguousarray(d[f"{m}_smoothed"].to_numpy(np.float32))
            ok = np.isfinite(pred)
            r[f"r_{m}"] = pearson(gt[ok], pred[ok])
        v = d.measured.to_numpy(float)
        rows.append({"group": b.group, "gene": b.gene, "split": b.split, "biopsy": b.biopsy, **r,
                     "moran_i_measured": moran(v, np.c_[d.x.values, d.y.values].astype(float)),
                     "detected_fraction": float((v > 0).mean()), "n_spots": len(d),
                     "n_candidates": b.n_candidates})
    return pd.DataFrame(rows)


def marker_means(data_dir):
    held = pd.read_csv(data_dir / "heldout_biopsies.csv")
    means = pd.read_csv(data_dir / "biopsy_marker_means.csv", float_precision="round_trip")
    return (means.melt(id_vars=["split", "biopsy", "marker"], value_vars=SOURCES_D, var_name="source")
                 .merge(held[["split", "biopsy", "group"]], on=["split", "biopsy"]))


def cliffs_delta(a, b):
    a, b = np.asarray(a)[:, None], np.asarray(b)[None, :]
    return float(((a > b).sum() - (a < b).sum()) / (a.size * b.size)) if a.size and b.size else np.nan


def group_differences(means):
    delta, zdiff = [], []
    for (split, source, marker), sub in means.groupby(["split", "source", "marker"]):
        z_sub = sub[sub.group.isin(Z_GROUPS)]
        v = z_sub.value
        z = (v - v.mean()) / v.std() if v.std() > 0 else v * np.nan
        for a, b in CONTRASTS:
            key = {"split": split, "source": source, "marker": marker, "contrast": f"{a} vs {b}"}
            delta.append({**key, "value": cliffs_delta(sub[sub.group == a].value.dropna(),
                                                       sub[sub.group == b].value.dropna())})
            zdiff.append({**key, "value": z[z_sub.group == a].mean() - z[z_sub.group == b].mean()})
    agg = lambda rows: (pd.DataFrame(rows).groupby(["contrast", "source", "marker"])["value"]
                          .agg(mean="mean", sd="std").reset_index())
    return agg(delta), agg(zdiff)


def tables(data_dir):
    per_split = pd.concat([topk_by_split(data_dir), chance_by_split(data_dir)], ignore_index=True)
    topk = (per_split.groupby(["group", "model", "top_k"])["r"]
                     .agg(mean="mean", sd="std", n_splits="size").reset_index())
    delta, zdiff = group_differences(marker_means(data_dir))
    return {"topk": topk, "topk_by_split": per_split, "program_accuracy": program_accuracy(data_dir),
            "shown_biopsies": shown_biopsies(data_dir), "group_difference_delta": delta,
            "group_difference_z": zdiff}


def trend_line(m):
    """Line colour and dash of model m in panel a: solid if adapted on KPMP training biopsies, dashed if not."""
    adapted = m in style.ADAPTED
    return dict(color=style.mix(style.MODEL_COLOR[m], style.PANEL, 55 if adapted else 30),
                ls="-" if adapted else RELEASED_DASH)


def test_biopsy_counts(data_dir):
    held = pd.read_csv(data_dir / "heldout_biopsies.csv")
    held = held[held.in_stpath_squall_training == 0]
    return held.groupby(["split", "group"]).biopsy.nunique().groupby("group").median()


def draw_topk(fig, rect, topk, per_split, n_test):
    x0, y0, w, h = rect
    gap = 0.012
    fw = (w - gap * (len(A_GROUPS) - 1)) / len(A_GROUPS)
    xl = 0.16 * h
    width = 0.9 / len(MODELS_A)
    xk = np.arange(len(TOP_K_SHOWN))
    axes = []
    for i, g in enumerate(A_GROUPS):
        ax = fig.add_axes([x0 + i * (fw + gap), y0 + xl, fw, h - xl - 0.06 * h], sharey=axes[0] if axes else None)
        for j, m in enumerate(MODELS_A):
            sub = topk[(topk.group == g) & (topk.model == m)].set_index("top_k")
            pts = per_split[(per_split.group == g) & (per_split.model == m)]
            xs = [(x, k) for x, k in zip(xk + (j - (len(MODELS_A) - 1) / 2) * width, TOP_K_SHOWN) if k in sub.index]
            ax.plot([x for x, _ in xs], [sub.loc[k, "mean"] for _, k in xs], lw=style.LW, zorder=1,
                    solid_capstyle="round", **trend_line(m))
            for x, k in xs:
                style.split_dots(ax, x, pts[pts.top_k == k].sort_values("split").r,
                                 style.mix(style.MODEL_COLOR[m], style.PANEL, 35), w=0.8 * width, ms=1.4,
                                 mean=sub.loc[k, "mean"], sd=sub.loc[k, "sd"], line_color=style.MODEL_COLOR[m])
        chance = topk[(topk.group == g) & (topk.model == "chance")].set_index("top_k")["mean"].reindex(TOP_K_SHOWN)
        ax.plot(xk, chance.to_numpy(), ls=CHANCE_DASH, color=CHANCE_COLOR, lw=style.LW, zorder=5)
        ax.set_xticks(xk)
        ax.set_xticklabels([TOP_K_LABEL[k] for k in TOP_K_SHOWN])
        ax.set_xlim(-0.55, len(TOP_K_SHOWN) - 0.45)
        ax.set_title(f"{style.GROUP_LABEL[g]}  ·  {int(n_test.get(g, 0))} biopsies", loc="left", color=style.INK)
        if i:
            ax.tick_params(labelleft=False)
        ax.yaxis.grid(True, color=style.HAIR, lw=style.LW)
        ax.set_axisbelow(True)
        axes.append(ax)
    axes[0].set_ylabel("Pearson correlation")
    axes[0].set_ylim(bottom=0)
    fig.text(x0 + w / 2, y0, "Best-predicted genes (top-K)", ha="center", va="bottom", fontsize=style.FS_LAB,
             color=style.INK)


def draw_model_handle(fig, x, y, m):
    from matplotlib.lines import Line2D
    half = 2.4 * style.MM / fig.get_figwidth()
    c = style.MODEL_COLOR[m]
    fig.add_artist(Line2D([x - half, x + half], [y, y], transform=fig.transFigure, lw=style.LW, **trend_line(m)))
    fig.add_artist(Line2D([x], [y], transform=fig.transFigure, marker="o", ms=2.2, lw=0, color=c, mew=0))


def draw_exposure_key(fig, x, y):
    """Key of panel a as a table: model family by exposure to KPMP data, top-left at (x, y); chance line to the right."""
    from matplotlib.lines import Line2D
    W, H = fig.get_figwidth() / style.MM, fig.get_figheight() / style.MM
    xs = [x + sum(KEY_COL_MM[:i]) / W for i in range(len(KEY_COL_MM) + 1)]
    dy, mm = KEY_ROW_MM / H, 1 / W
    for t, a, b in TRAINING:
        fig.text(xs[a + 1], y, t, ha="left", va="top", fontsize=style.FS_SMALL, color=style.INK)
        fig.add_artist(Line2D([xs[a + 1], xs[b + 1] - 3 * mm], [y - 0.95 * dy] * 2, transform=fig.transFigure,
                              color=style.MUTED, lw=0.4))
    for j, t in enumerate(EXPOSURE):
        fig.text(xs[j + 1], y - 1.1 * dy, t, ha="left", va="top", fontsize=style.FS_SMALL, color=style.MUTED)
    for i, (name, hue, cells) in enumerate(FAMILY):
        yy = y - (i + 2.55) * dy
        fig.text(xs[0], yy, name, ha="left", va="center", fontsize=style.FS_SMALL, color=style.MODEL_COLOR[hue])
        for j in range(len(EXPOSURE)):
            if j in cells:
                draw_model_handle(fig, xs[j + 1] + 2.4 * mm, yy, cells[j])
            else:
                fig.text(xs[j + 1] + 2.4 * mm, yy, "–", ha="center", va="center", fontsize=style.FS_SMALL,
                         color=style.MUTED)
    yy = y - (len(FAMILY) + 1.55) * dy
    x0 = xs[-1] + 2 * mm
    fig.add_artist(Line2D([x0, x0 + 4.8 * mm], [yy, yy], transform=fig.transFigure, color=CHANCE_COLOR,
                          ls=CHANCE_DASH, lw=style.LW))
    fig.text(x0 + 5.8 * mm, yy, CHANCE_LABEL, ha="left", va="center", fontsize=style.FS_SMALL, color=style.INK)


def stacked_rows(sections, gap):
    ys, heads, y = {}, [], 0.0
    for name, rows in sections:
        heads.append((name, y))
        y += 1.0
        for row in rows:
            ys[row[0]] = y
            y += 1.0
        y += gap
    return ys, heads, y - gap


def draw_programs(fig, rect, programs):
    ys, heads, ymax = stacked_rows(COMPARTMENTS, 0.3)
    x0, y0, w, h = rect
    lab_w, xl, top = 0.43 * w, 0.17 * h, 0.13 * h
    ax = fig.add_axes([x0 + lab_w, y0 + xl, w - lab_w, h - xl - top])
    wide = programs.pivot_table(index="program", columns="model", values="r")
    rows = list(ys)
    yy = np.array([ys[r] for r in rows])
    for r in rows:
        ax.plot([wide.loc[r, "zero_shot"], wide.loc[r, "kidney"]], [ys[r]] * 2, color=style.WASH, lw=1.2,
                zorder=1, solid_capstyle="round")
    for m in MODELS_B_SHOWN:
        ax.plot(wide.reindex(rows)[m], yy, style.MODEL_MARKER[m], ms=2.8, color=style.MODEL_COLOR[m],
                mec=style.PANEL, mew=0, lw=0, zorder=3 if m == "kidney" else 2)
    for _, yh in heads[1:]:
        ax.axhline(yh - 0.65, color=style.HAIR, lw=style.LW, zorder=0)
    ax.axvline(0, color=style.RULE, lw=style.LW, zorder=0)
    ax.axvline(R_CUTOFF, color=style.HAIR, lw=style.LW, ls=(0, (2, 2)), zorder=0)
    ax.set_xlim(min(-0.1, np.floor(np.nanmin(wide.values) * 10) / 10),
                max(0.7, np.ceil(np.nanmax(wide.values) * 10) / 10))
    ax.set_ylim(ymax + 0.6, -0.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Pearson correlation", labelpad=1.5)
    for r in rows:
        ax.text(-0.03, ys[r], r, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=style.FS_SMALL, color=style.INK)
    for name, yh in heads:
        ax.text(-lab_w / (w - lab_w), yh, name.upper(), transform=ax.get_yaxis_transform(), ha="left",
                va="center", fontsize=style.FS_SMALL, color=style.MUTED)


def horizontal(x, y):
    """Spot coordinates rotated so the biopsy's long axis is horizontal, shifted to start at 0."""
    xy = np.c_[x, y].astype(float)
    xy -= xy.mean(0)
    v = np.linalg.eigh(np.cov(xy.T))[1]
    a = np.arctan2(v[1, -1], v[0, -1])
    c, s = np.cos(-a), np.sin(-a)
    u, w = xy[:, 0] * c - xy[:, 1] * s, xy[:, 0] * s + xy[:, 1] * c
    return u - u.min(), w - w.min()


def text_width_in(text, **font):
    return TextPath((0, 0), text, size=style.FS_SMALL, prop=FontProperties(**font)).get_extents().width / 72


def plan_maps(shown, data_dir, width_in):
    spots = read_spots(data_dir)
    blocks = []
    for _, b in shown.iterrows():
        d = spots[spots.biopsy == b.biopsy].reset_index(drop=True)
        x, y = horizontal(d.x.values, d.y.values)
        pad = 0.03 * np.ptp(x)
        blocks.append(dict(b=b, d=d, x=x, y=y, pad=pad, asp=(np.ptp(y) + 2 * pad) / (np.ptp(x) + 2 * pad)))
    label_in = SWATCH_COL_IN + max(text_width_in(s) for m in MAP_ROWS for s in MAP_LABEL[m].split("\n")) + LABEL_PAD_IN
    cw = (width_in - label_in - MAP_GAP_IN * (len(blocks) - 1)) / len(blocks)
    for bl in blocks:
        bl["mw"] = min(cw, MAP_MAX_H_IN / bl["asp"])
        bl["mh"] = bl["mw"] * bl["asp"]
    rh = max(bl["mh"] for bl in blocks)
    height = HEAD_IN + len(MAP_ROWS) * rh + (len(MAP_ROWS) - 1) * ROW_GAP_IN + FOOT_IN
    return dict(blocks=blocks, cw=cw, rh=rh, label_in=label_in, height_in=height)


def r_text(v):
    t = f"{v:.2f}"
    return "0.00" if t == "-0.00" else t.replace("-", "−")


def draw_maps(fig, left_in, top_in, lay, data_dir):
    from matplotlib.patches import Rectangle
    from PIL import Image
    fw, fh = fig.get_size_inches()
    fxi = lambda v: v / fw
    fyi = lambda v: 1 - v / fh
    rh = lay["rh"]
    row_y = [top_in + HEAD_IN + i * (rh + ROW_GAP_IN) for i in range(len(MAP_ROWS))]
    for m, yt in zip(MAP_ROWS, row_y):
        yc = yt + rh / 2
        if m in MAP_MODELS:
            fig.patches.append(Rectangle((fxi(left_in), fyi(yc + 0.035)), fxi(0.07), 0.07 / fh,
                                         transform=fig.transFigure, facecolor=style.MODEL_COLOR[m],
                                         edgecolor="none"))
        fig.text(fxi(left_in + SWATCH_COL_IN), fyi(yc), MAP_LABEL[m], ha="left", va="center",
                 fontsize=style.FS_SMALL, color=style.INK, linespacing=1.1)
    for j, bl in enumerate(lay["blocks"]):
        b, d, x, y, pad, mw, mh = bl["b"], bl["d"], bl["x"], bl["y"], bl["pad"], bl["mw"], bl["mh"]
        xc = left_in + lay["label_in"] + j * (lay["cw"] + MAP_GAP_IN)
        y_head = fyi(top_in + HEAD_IN - 0.05)
        fig.text(fxi(xc), y_head, b.gene, ha="left", va="baseline", fontsize=style.FS_LAB, color=style.INK,
                 style="italic")
        gw = TextPath((0, 0), b.gene, size=style.FS_LAB, prop=FontProperties(style="italic")).get_extents().width
        fig.text(fxi(xc + (gw + 3.5) / 72), y_head, f"{style.GROUP_LABEL[b.group]} · {b.biopsy}", ha="left",
                 va="baseline", fontsize=style.FS_SMALL, color=style.MUTED)
        spacing = np.median(cKDTree(np.c_[x, y]).query(np.c_[x, y], k=2)[0][:, 1])
        ext = (-pad, np.ptp(x) + pad, -pad, np.ptp(y) + pad)
        for m, yt in zip(MAP_ROWS, row_y):
            top = yt + (rh - mh) / 2
            ax = fig.add_axes([fxi(xc), fyi(top + mh), fxi(mw), mh / fh])
            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[3], ext[2])
            ax.axis("off")
            if m == "he":
                he = np.asarray(Image.open(data_dir / f"he_{b.biopsy}.png").convert("RGB"))
                ax.imshow(he, aspect="auto", extent=(ext[0], ext[1], ext[3], ext[2]), interpolation="lanczos")
                bar = SCALEBAR_UM / b.pixel_size_um
                if bar < 0.9 * np.ptp(x):
                    ax.plot([0, bar], [np.ptp(y) + pad + 0.35 * ROW_GAP_IN / mh * np.ptp(y)] * 2, color=style.INK,
                            lw=style.LW, solid_capstyle="butt", clip_on=False)
                continue
            v = d[m].to_numpy()
            lo, hi = np.nanpercentile(v, [2, 98])
            size = 0.95 * spacing * mw * 72 / (np.ptp(x) + 2 * pad)
            ax.scatter(x, y, c=v, s=size ** 2, cmap=style.EXPR_CMAP, vmin=lo, vmax=hi if hi > lo else lo + 1e-6,
                       linewidths=0, rasterized=True)
            if m != "measured":
                fig.text(fxi(xc + mw), fyi(top - 0.015), f"Pearson $\\mathit{{r}}$ = {r_text(b[f'r_{m}'])}",
                         ha="right", va="bottom", fontsize=style.FS_SMALL, color=style.INK)


def draw_expression_key(fig, rect):
    ax = fig.add_axes(rect)
    ax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=style.EXPR_CMAP)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    yc = rect[1] + rect[3] / 2
    fig.text(rect[0] - 0.004, yc, "low", ha="right", va="center", fontsize=style.FS_SMALL, color=style.MUTED)
    fig.text(rect[0] + rect[2] + 0.004, yc, "high", ha="left", va="center", fontsize=style.FS_SMALL,
             color=style.MUTED)


def draw_source_key(fig, x, y):
    from matplotlib.patches import Patch
    fig.legend([Patch(facecolor=style.MODEL_COLOR[s], edgecolor="none") for s in SOURCES_D_SHOWN],
               [LABEL[s] for s in SOURCES_D_SHOWN], loc="lower right", ncol=len(SOURCES_D_SHOWN),
               bbox_to_anchor=(x, y), fontsize=style.FS_SMALL, borderaxespad=0, borderpad=0)


def draw_differences(fig, rect, zdiff):
    ys, heads, ymax = stacked_rows(PROGRAMS, 0.35)
    shown = zdiff[zdiff.source.isin(SOURCES_D_SHOWN)]
    xmax = float(np.ceil((shown["mean"].abs() + shown["sd"].fillna(0)).max() * 2) / 2)
    offset = {s: (i - (len(SOURCES_D_SHOWN) - 1) / 2) * 0.2 for i, s in enumerate(SOURCES_D_SHOWN)}
    expected = {m: e for _, ms in PROGRAMS for m, e in ms}
    x0, y0, w, h = rect
    lab_w, gap = 0.15 * w, 0.06 * w
    fw = (w - lab_w - gap) / 2 - 0.02 * w
    xlab_h = 0.12 * h
    ticks = np.arange(-np.floor(xmax), np.floor(xmax) + 1)
    axes = []
    for k, (a, b) in enumerate(CONTRASTS):
        ax = fig.add_axes([x0 + lab_w + k * (fw + gap), y0 + xlab_h, fw, h - xlab_h - 0.07 * h])
        axes.append(ax)
        for src in SOURCES_D_SHOWN:
            sub = shown[(shown.contrast == f"{a} vs {b}") & (shown.source == src)].set_index("marker").reindex(MARKERS)
            ax.errorbar(sub["mean"], np.array([ys[m] for m in MARKERS]) + offset[src],
                        xerr=sub["sd"] if sub["sd"].notna().any() else None, fmt="o", ms=2.8,
                        color=style.MODEL_COLOR[src], mec=style.PANEL, mew=0, elinewidth=style.LW, capsize=0,
                        zorder=3 if src == "measured" else 2)
        for m in MARKERS:
            if a in expected[m]:
                ax.text(1.04, ys[m], "↑" if expected[m][a] > 0 else "↓", ha="left", va="center",
                        fontsize=style.FS_LAB, color=style.RULE, transform=ax.get_yaxis_transform())
        ax.axvline(0, color=style.RULE, lw=style.LW, zorder=1)
        for xv in ticks[ticks != 0]:
            ax.axvline(xv, color=style.HAIR, lw=style.LW, zorder=0)
        for _, yh in heads[1:]:
            ax.axhline(yh - 0.68, color=style.HAIR, lw=style.LW, zorder=0)
        ax.set_xlim(-xmax - 0.05, xmax + 0.05)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}".replace("-", "−") for t in ticks])
        ax.set_ylim(ymax + 0.6, -0.6)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_title(f"{a} vs {b}", loc="left", color=style.INK)
        ax.set_xlabel("Expression difference (z-score)", labelpad=1.5)
        for xa, ha, txt in ((0, "left", f"← higher in {b}"), (1, "right", f"higher in {a} →")):
            ax.annotate(txt, (xa, 0), xycoords="axes fraction", xytext=(0, -10.5), textcoords="offset points",
                        ha=ha, va="top", fontsize=style.FS_SMALL, color=style.MUTED)
    for m in MARKERS:
        axes[0].text(-0.03, ys[m], m, transform=axes[0].get_yaxis_transform(), ha="right", va="center",
                     fontsize=style.FS_LAB, color=style.INK, style="normal" if m == "Fibroblast score" else "italic")
    for name, yh in heads:
        axes[0].text(-lab_w / fw, yh, name.upper(), transform=axes[0].get_yaxis_transform(), ha="left",
                     va="center", fontsize=style.FS_SMALL, color=style.MUTED)


def draw(tables, data_dir):
    import matplotlib.pyplot as plt
    lay = plan_maps(tables["shown_biopsies"].merge(pd.read_csv(data_dir / "shown_biopsies.csv")
                                                   [["biopsy", "pixel_size_um"]], on="biopsy"),
                    data_dir, (RIGHT_MM - LETTER_MM) * style.MM)
    a_top = KEY_MM + HEAD_MM
    c_top = a_top + ROW1_MM + GAP_MM + HEAD_MM
    d_top = c_top + lay["height_in"] / style.MM + GAP_MM + HEAD_MM
    height_mm = d_top + ROW3_MM + 3.5
    fig = plt.figure(figsize=(WIDTH_MM * style.MM, height_mm * style.MM))
    fx = lambda mm: mm / WIDTH_MM
    fy = lambda mm: 1 - mm / height_mm
    rect = lambda top, h, left, right: [fx(left), fy(top + h), fx(right - left), h / height_mm]

    draw_topk(fig, rect(a_top, ROW1_MM, LEFT_MM - 2, SPLIT_MM - 6), tables["topk"], tables["topk_by_split"],
              test_biopsy_counts(data_dir))
    style.header(fig, fx(LETTER_MM), fy(a_top - 2.2), "a", "Accuracy on held-out biopsies")
    draw_exposure_key(fig, fx(LEFT_MM - 2), fy(0.6))

    draw_programs(fig, rect(a_top, ROW1_MM, SPLIT_MM, RIGHT_MM), tables["program_accuracy"])
    style.header(fig, fx(SPLIT_MM - 2), fy(a_top - 2.2), "b", "Accuracy by compartment and program")

    draw_maps(fig, LETTER_MM * style.MM, c_top * style.MM, lay, data_dir)
    style.header(fig, fx(LETTER_MM), fy(c_top - 2.2), "c", "Measured and virtual expression on held-out biopsies")
    draw_expression_key(fig, [fx(RIGHT_MM - 30), fy(c_top - 2.2) + 0.001, fx(12), 1.6 / height_mm])

    draw_differences(fig, rect(d_top, ROW3_MM, LETTER_MM, RIGHT_MM), tables["group_difference_z"])
    style.header(fig, fx(LETTER_MM), fy(d_top - 2.2), "d", "Differences between disease groups, measured and virtual")
    draw_source_key(fig, fx(RIGHT_MM), fy(d_top - 2.2))
    return fig
