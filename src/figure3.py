"""Figure 3: DeepSpot-M Kidney on Open TG-GATEs rat kidneys."""
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage, stats

import style
from style import INK, MUTED, PANEL, RULE, mix

W_MM, H_MM = 183, 230
TILE_MM = 0.056
FS_TITLE, FS_AX, FS_HEAD, FS_TEXT = 9.0, 7.5, 7.0, 6.5
GREY = "#8C8C8C"
DASH = (0, (2.2, 1.6))
GROUP = ["compound", "regimen", "time"]
TIME_HOURS = {"3 hr": 3, "6 hr": 6, "9 hr": 9, "24 hr": 24, "4 day": 96, "8 day": 192, "15 day": 360,
              "29 day": 696}
GRADE_WORD = {0: "none", 1: "minimal", 2: "slight", 3: "moderate", 4: "severe"}
GRADE_LABEL = {0: "No lesion", 1: "Minimal", 2: "Slight", 3: "Moderate", 4: "Severe"}
GRADE_COLOR = {0: "#DCDCDC", 1: "#FCC5AE", 2: "#F68D6E", 3: "#D9442F", 4: "#8C1A12"}
REGIONS = ["cortex", "outer_medulla", "inner_medulla"]
REGION_BORDERS = (0.49, 0.76)
REGION_NAME = {"cortex": "Cortex", "outer_medulla": "Outer medulla", "inner_medulla": "Inner medulla"}
REGION_COLOR = {"cortex": "#E69F00", "outer_medulla": "#009E73", "inner_medulla": "#0072B2"}
REGION_TEXT = {z: mix(c, "#000000", 28) for z, c in REGION_COLOR.items()}
MAP_COLORS = ["#F4F2F8", "#CFC8E6", "#9387CC", style.MODEL_COLOR["kidney"], "#20184A"]
ROC_LESION, ROC_BUN = GRADE_COLOR[3], INK
HE_WHITE_PCT, HE_BLACK, HE_GAMMA, HE_SAT = 99.0, 0.10, 1.10, 1.40

MAP_ROWS = [("cisplatin", "Cisplatin", "15 day", "High", "injury", "outer_medulla", "Virtual injury score",
             None, (1, 99.5)),
            ("gentamicin", "Gentamicin", "29 day", "High", "proximal_tubule_loss", "cortex",
             "Virtual proximal tubule score", "cortex", (2, 98))]
REGION_SETS = [("cisplatin", ["Cisplatin"], "injury", "injury", "outer_medulla"),
               ("gentamicin", ["Gentamicin"], "tubule loss", "proximal_tubule_loss", "cortex"),
               ("bromoethylamine", ["2-Bromo-", "ethylamine"], "injury", "injury", "inner_medulla")]
TIME_COURSES = [("cisplatin", "Cisplatin", "High", "High dose, outer medulla injury", "injury", "outer_medulla"),
                ("puromycin aminonucleoside", "Puromycin aminonucleoside", "Middle",
                 "Middle dose, cortical tubule loss", "proximal_tubule_loss", "cortex"),
                ("gentamicin", "Gentamicin", "High", "High dose, cortical tubule loss", "proximal_tubule_loss",
                 "cortex")]
DAYS = [("4 day", 4), ("8 day", 8), ("15 day", 15), ("29 day", 29)]
C_YLIM = (-2.5, 6.5)
GRID_READOUTS = [("cisplatin", "injury", "outer_medulla"), ("gentamicin", "proximal_tubule_loss", "cortex"),
                 ("bromoethylamine", "injury", "inner_medulla"),
                 ("puromycin aminonucleoside", "proximal_tubule_loss", "cortex"),
                 ("carboplatin", "injury", "outer_medulla")]
GRID_TIMES = [("single", "3 hr"), ("single", "6 hr"), ("single", "9 hr"), ("single", "24 hr"),
              ("repeat", "4 day"), ("repeat", "8 day"), ("repeat", "15 day"), ("repeat", "29 day")]
DOSES = ["Low", "Middle", "High"]
LESION_SCORE = "injury_section"
DETECTION = [("moderate or severe lesion vs no lesion", True),
             ("any lesion vs no lesion", False),
             ("BUN at least 2x matched controls vs all other rats", True)]


def control_baseline(rats, cols):
    """Mean of each rat's matched controls (the other controls for a control rat) and the pooled control SD."""
    x = rats[cols].to_numpy(float)
    k = rats.dose_level.eq("Control").to_numpy(float)[:, None]
    g = rats.groupby(GROUP, sort=False).ngroup().to_numpy()
    tot = np.zeros((g.max() + 1, len(cols)))
    np.add.at(tot, g, x * k)
    n = np.bincount(g, weights=k[:, 0])[:, None]
    base = (tot[g] - x * k) / (n[g] - k)
    sd = np.sqrt((((x - tot[g] / n[g]) * k) ** 2).sum(0) / (n - 1).sum())
    return base, sd


def readouts(rats):
    """Per rat: change from matched controls in pooled control SD units, more damage positive."""
    cols = [c for c in rats.columns if c.startswith(("injury_", "proximal_tubule_"))]
    base, sd = control_baseline(rats, cols + ["bun"])
    z = (rats[cols].to_numpy(float) - base[:, :-1]) / sd[:-1]
    out = rats[["slide_id", "animal", "compound", "regimen", "time", "dose_level", "max_lesion_grade",
                "bun"]].copy()
    for c, v in zip(cols, z.T):
        if c.startswith("proximal_tubule_"):
            out[c.replace("proximal_tubule_", "proximal_tubule_loss_")] = -v
        else:
            out[c] = v
    out["bun_over_controls"] = rats.bun.to_numpy(float) / base[:, -1]
    return out


def section(data_dir, slide_id):
    with np.load(data_dir / "maps" / f"{slide_id}.npz") as f:
        t = pd.DataFrame({"row": f["row"].astype(int), "col": f["col"].astype(int), "score": f["score"],
                          "depth": f["depth"]})
        return t, float(f["he_px_per_tile"])


def map_examples(r, data_dir):
    rows = []
    for cp, _, time, dose, score, region, _, range_region, pct in MAP_ROWS:
        col = f"{score}_{region}"
        at = (r.compound == cp) & (r.regimen == "repeat") & (r.time == time)
        picked = []
        for role, level in [("treated", dose), ("control", "Control")]:
            g = r[at & (r.dose_level == level)].sort_values(col)
            picked.append((role, g.iloc[(len(g) - 1) // 2]))
        tiles = [section(data_dir, x.slide_id)[0] for _, x in picked]
        if range_region:
            tiles = [t[t.depth < REGION_BORDERS[0]] for t in tiles]
        lo, hi = np.percentile(np.concatenate([t.score.to_numpy() for t in tiles]), pct)
        for role, x in picked:
            rows.append(dict(compound=cp, regimen="repeat", time=time, dose_level=dose, role=role,
                             slide_id=x.slide_id, animal=x.animal, score=score, region=region, z=x[col],
                             max_lesion_grade=int(x.max_lesion_grade), bun=x.bun, color_low=lo, color_high=hi))
    return pd.DataFrame(rows)


def wilcoxon_p(x):
    return float(stats.wilcoxon(np.asarray(x, float)).pvalue)


def region_change(r):
    rows, dots = [], []
    for cp, _, _, score, expected in REGION_SETS:
        les = r[(r.compound == cp) & (r.dose_level != "Control") & (r.max_lesion_grade > 0)]
        v = les[[f"{score}_{z}" for z in REGIONS]]
        diff = v[f"{score}_{expected}"] - v[[f"{score}_{z}" for z in REGIONS if z != expected]].mean(axis=1)
        p = wilcoxon_p(diff)
        for z in REGIONS:
            x = v[f"{score}_{z}"]
            rows.append(dict(compound=cp, score=score, region=z, expected=z == expected, n_lesion=len(les),
                             mean_z=float(x.mean()), expected_vs_other_mean=float(diff.mean()),
                             expected_vs_other_p=p, region_p=wilcoxon_p(x)))
            dots.append(pd.DataFrame(dict(compound=cp, animal=les.animal, region=z, z=x)))
    return pd.DataFrame(rows), pd.concat(dots, ignore_index=True)


def dose_time_groups(r):
    rows = []
    for cp, score, region in GRID_READOUTS:
        for dose in DOSES:
            for regimen, time in GRID_TIMES:
                at = (r.compound == cp) & (r.regimen == regimen) & (r.time == time)
                g, c = r[at & (r.dose_level == dose)], r[at & (r.dose_level == "Control")]
                row = dict(compound=cp, regimen=regimen, time=time, dose_level=dose, n=len(g))
                if len(g):
                    row.update(n_lesion=int((g.max_lesion_grade > 0).sum()),
                               max_lesion_grade=int(g.max_lesion_grade.max()), score=score, region=region,
                               z=g[f"{score}_{region}"].mean(), bun=g.bun.mean(), bun_controls=c.bun.mean())
                    row["bun_2x"] = bool(row["bun"] >= 2 * row["bun_controls"])
                rows.append(row)
    return pd.DataFrame(rows)


def time_course(r, groups):
    g = groups.set_index(["compound", "regimen", "dose_level", "time"])
    rows = []
    for cp, _, dose, _, score, region in TIME_COURSES:
        col = f"{score}_{region}"
        treated = r[(r.compound == cp) & (r.regimen == "repeat") & (r.dose_level == dose)]
        rho, p = stats.spearmanr(treated[col], treated.time.map(TIME_HOURS))
        for time, day in DAYS:
            x = g.loc[(cp, "repeat", dose, time)]
            rows.append(dict(compound=cp, dose_level=dose, time=time, day=day, score=score, region=region, z=x.z,
                             n=int(x.n), n_lesion=int(x.n_lesion), max_lesion_grade=int(x.max_lesion_grade),
                             lesion=GRADE_WORD[int(x.max_lesion_grade)], rho=float(rho), p=float(p),
                             n_rats=len(treated)))
    return pd.DataFrame(rows)


def lesion_score_rats(r):
    out = r[["slide_id", "animal", "compound", "regimen", "time", "dose_level", "max_lesion_grade", LESION_SCORE,
             "bun", "bun_over_controls"]].copy()
    out["bun_2x"] = out.bun_over_controls >= 2
    out["moderate_or_severe"] = out.max_lesion_grade >= 3
    out["any_lesion"] = out.max_lesion_grade > 0
    return out


def groups_of(rats):
    none = ~rats.any_lesion
    return [(rats.moderate_or_severe, none), (rats.any_lesion, none), (rats.bun_2x, rats.bun_over_controls < 2)]


def auc_ci(a, b, n_boot=2000):
    """95 % CI of the AUC by a stratified bootstrap with a fixed random state."""
    rng = np.random.default_rng(0)
    v = np.empty(n_boot)
    for i in range(n_boot):
        x, y = rng.choice(a, len(a)), rng.choice(b, len(b))
        r = stats.rankdata(np.concatenate([x, y]))
        v[i] = (r[:len(x)].sum() - len(x) * (len(x) + 1) / 2) / (len(x) * len(y))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def detection_auc(rats):
    s = rats[LESION_SCORE].to_numpy(float)
    rows = []
    for (name, _), (pos, neg) in zip(DETECTION, groups_of(rats)):
        a, b = s[pos.to_numpy()], s[neg.to_numpy()]
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        lo, hi = auc_ci(a, b)
        rows.append(dict(score=LESION_SCORE, comparison=name, n_pos=len(a), n_neg=len(b),
                         auc=u.statistic / (len(a) * len(b)), ci_lo=lo, ci_hi=hi, p=u.pvalue))
    return pd.DataFrame(rows)


def roc(rats):
    s0 = rats[LESION_SCORE].to_numpy(float)
    curves = []
    for (name, shown), (pos, neg) in zip(DETECTION, groups_of(rats)):
        if not shown:
            continue
        pos, neg = pos.to_numpy(), neg.to_numpy()
        s = np.r_[s0[pos], s0[neg]]
        y = np.r_[np.ones(pos.sum()), np.zeros(neg.sum())]
        o = np.argsort(-s, kind="mergesort")
        s, y = s[o], y[o]
        cut = np.r_[np.where(np.diff(s))[0], len(s) - 1]
        tp, fp = np.cumsum(y)[cut], np.cumsum(1 - y)[cut]
        curves.append(pd.DataFrame(dict(curve=name, fpr=np.r_[0, fp / fp[-1]], tpr=np.r_[0, tp / tp[-1]])))
    return pd.concat(curves, ignore_index=True)


def tables(data_dir):
    rats = pd.read_csv(data_dir / "rats.csv", float_precision="round_trip")
    r = readouts(rats)
    groups = dose_time_groups(r)
    region, region_rats = region_change(r)
    lesion = lesion_score_rats(r)
    return {"map_examples": map_examples(r, data_dir), "region_change": region,
            "region_change_rats": region_rats, "dose_time_groups": groups, "time_course": time_course(r, groups),
            "lesion_score_rats": lesion, "detection_auc": detection_auc(lesion), "roc": roc(lesion)}


def fx(mm):
    return mm / W_MM


def fy(mm):
    return 1 - mm / H_MM


def add_axes(fig, x, y, w, h):
    return fig.add_axes([fx(x), fy(y + h), w / W_MM, h / H_MM])


def text(fig, x, y, s, size=FS_TEXT, color=INK, ha="left", va="baseline", **kw):
    return fig.text(fx(x), fy(y), s, fontsize=size, color=color, ha=ha, va=va, **kw)


def line(fig, x0, y0, x1, y1, color=RULE, lw=1.0, **kw):
    from matplotlib.lines import Line2D
    fig.add_artist(Line2D([fx(x0), fx(x1)], [fy(y0), fy(y1)], color=color, lw=lw, transform=fig.transFigure,
                          **kw))


def head(fig, x, y, letter, title):
    from matplotlib.transforms import offset_copy
    fig.text(fx(x), fy(y), letter, **style.LETTER)
    fig.text(fx(x), fy(y), title, transform=offset_copy(fig.transFigure, fig=fig, x=style.FS_LETTER * 1.05,
                                                         units="points"),
             fontsize=FS_TITLE, color=INK, ha="left", va="baseline")


def clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def ticks(ax):
    ax.tick_params(labelsize=FS_TEXT, length=2.5, pad=2.0)


def lattice(t, col):
    r0, c0 = t.row.min(), t.col.min()
    M = np.full((t.row.max() - r0 + 1, t.col.max() - c0 + 1), np.nan, np.float32)
    M[t.row.to_numpy() - r0, t.col.to_numpy() - c0] = t[col].to_numpy()
    return M


def smooth_nan(M, s=1.5):
    m = np.isfinite(M)
    out = ndimage.gaussian_filter(np.where(m, M, 0.0), s) / np.maximum(ndimage.gaussian_filter(m.astype(float), s),
                                                                         1e-6)
    out[~m] = np.nan
    return out


def he_view(path):
    """H&E for display: glass set to white, then white balance, contrast and saturation."""
    from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
    x = np.asarray(Image.open(path).convert("RGB")).astype(float) / 255
    flat = x.reshape(-1, 3)
    bg = flat[(flat.min(1) > 0.8) & (np.ptp(flat, 1) < 0.08)]
    if len(bg) > 1000:
        x = np.clip(x / np.median(bg, 0), 0, 1)
    x = (x * 255).round() / 255
    x = np.clip(x / np.maximum(np.percentile(x.reshape(-1, 3), HE_WHITE_PCT, axis=0), 0.5), 0, 1)
    x = np.clip((x - HE_BLACK) / (1 - HE_BLACK), 0, 1) ** HE_GAMMA
    hsv = rgb_to_hsv(x)
    hsv[..., 1] = np.clip(hsv[..., 1] * HE_SAT, 0, 1)
    return (hsv_to_rgb(hsv) * 255).round().astype(np.uint8)


def region_labels(fig, ax, t, cx, cy, hw, hh, x_img, y0, IMW, IMH, x_text, y_min, angle=42.0):
    """Region names on the H&E, with leaders to points on one ray from the papilla tip to the capsule."""
    r, c = t.row.to_numpy(), t.col.to_numpy()
    r0, c0 = r.min(), c.min()
    D = smooth_nan(lattice(t, "depth"))
    x, y = (c + 0.5) * TILE_MM, (r + 0.5) * TILE_MM
    d = t.depth.to_numpy(float)
    ok = np.isfinite(d)
    deep = ok & (d >= np.nanpercentile(d[ok], 97))
    ox, oy = x[deep].mean(), y[deep].mean()

    def depth_at(px, py):
        i, j = int(np.floor(py / TILE_MM)) - r0, int(np.floor(px / TILE_MM)) - c0
        return D[i, j] if 0 <= i < D.shape[0] and 0 <= j < D.shape[1] else np.nan

    def ray(ang):
        u = np.array([np.cos(ang), np.sin(ang)])
        ss = np.arange(0.0, 2 * max(hw, hh), TILE_MM / 4)
        dd = np.array([depth_at(ox + u[0] * v, oy + u[1] * v) for v in ss])
        fin = np.flatnonzero(np.isfinite(dd))
        if len(fin) < 8 or np.diff(fin).max(initial=1) > 3:
            return None
        seg, dv = ss[fin[0]:fin[-1] + 1], dd[fin[0]:fin[-1] + 1]
        b1, b2 = REGION_BORDERS
        if not ((dv < b1).any() and (dv < b2).any() and (dv >= b2).any()):
            return None
        return u, seg[0], seg[np.flatnonzero(dv < b2)[0]], seg[np.flatnonzero(dv < b1)[0]], seg[-1] + TILE_MM / 2

    best = None
    for deg in range(90, 181, 3):
        rr = ray(np.deg2rad(deg))
        if rr is not None:
            sc = min(rr[2] - rr[1], rr[3] - rr[2], rr[4] - rr[3])
            if best is None or sc > best[0]:
                best = (sc, rr)
    u, s_in, s_im, s_om, s_out = best[1] if best else ray(np.deg2rad(180 - angle))
    targets = []
    for z, sa, sb in [("inner_medulla", s_in, s_im), ("outer_medulla", s_im, s_om), ("cortex", s_om, s_out)]:
        P = np.array([ox, oy]) + u * (sa + sb) / 2
        ax.scatter([P[0]], [P[1]], s=14, color=REGION_COLOR[z], edgecolors=PANEL, linewidths=1.0, zorder=8,
                   clip_on=False)
        targets.append((REGION_NAME[z], REGION_TEXT[z], REGION_COLOR[z], P[0], P[1], False))
    E = np.argwhere(~np.isfinite(lattice(t, "depth")))
    if len(E):
        ex, ey = (E[:, 1] + c0 + 0.5) * TILE_MM, (E[:, 0] + r0 + 0.5) * TILE_MM
        k = np.argmin((ex - ox) ** 2 + (ey - oy) ** 2)
        targets.insert(0, ("Renal pelvis", INK, INK, ex[k], ey[k], True))
    to_x = lambda v: x_img + (v - (cx - hw)) / (2 * hw) * IMW
    to_y = lambda v: y0 + (v - (cy - hh)) / (2 * hh) * IMH
    ly = [max(to_y(tg[4]), y_min) for tg in targets]
    for i in range(1, len(ly)):
        ly[i] = max(ly[i], ly[i - 1] + 3.0)
    ly = [v - max(0.0, ly[-1] - (y0 + IMH - 1.0)) for v in ly]
    for (name, tc, lc, px, py, dot), yl in zip(targets, ly):
        text(fig, x_text, yl + 0.85, name, color=tc, ha="right")
        line(fig, x_text + 0.6, yl, to_x(px), to_y(py), color=lc, zorder=6)
        if dot:
            ax.scatter([px], [py], s=10, color=lc, edgecolors=PANEL, linewidths=1.0, zorder=7, clip_on=False)


def panel_a(fig, maps, data_dir, top):
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    cmap = LinearSegmentedColormap.from_list("score", MAP_COLORS)
    XS = [21.0, 47.0, 78.5]
    IMW, IMH = 23.5, 22.5
    head(fig, 2, top, "a", "Virtual maps localize the injury")
    ys = top + 7.2
    for x0, x1, s in [(XS[0], XS[1] + IMW, "Treated rat"), (XS[2], XS[2] + IMW, "Time-matched control")]:
        text(fig, (x0 + x1) / 2, ys, s, size=FS_AX, ha="center")
        line(fig, x0, ys + 1.3, x1, ys + 1.3, color=mix(RULE, PANEL, 35))
    rows_y = [top + 14.5, top + 14.5 + IMH + 9.0]
    secs = {s: section(data_dir, s) for s in maps.slide_id}
    k = min(min(IMW / ((t.col.max() - t.col.min() + 1) * TILE_MM), IMH / ((t.row.max() - t.row.min() + 1) * TILE_MM))
            for t, _ in secs.values())
    hw, hh = IMW / k / 2, IMH / k / 2
    for i, (cp, label, time, dose, _, _, maphead, _, _) in enumerate(MAP_ROWS):
        y0 = rows_y[i]
        m = maps[maps.compound == cp].set_index("role")
        lo, hi = m.color_low.iloc[0], m.color_high.iloc[0]
        text(fig, XS[0] + IMW / 2, y0 - 1.7, "H&E", size=FS_HEAD, ha="center")
        text(fig, (XS[1] + XS[2] + IMW) / 2, y0 - 1.7, maphead, size=FS_HEAD, ha="center")
        for role in ["treated", "control"]:
            x = m.loc[role]
            t, px = secs[x.slide_id]
            r0, r1, c0, c1 = t.row.min(), t.row.max(), t.col.min(), t.col.max()
            ext = [c0 * TILE_MM, (c1 + 1) * TILE_MM, (r1 + 1) * TILE_MM, r0 * TILE_MM]
            cx, cy = (c0 + c1 + 1) / 2 * TILE_MM, (r0 + r1 + 1) / 2 * TILE_MM
            D = smooth_nan(lattice(t, "depth"))
            Xc, Yc = (np.arange(c0, c1 + 1) + 0.5) * TILE_MM, (np.arange(r0, r1 + 1) + 0.5) * TILE_MM
            kinds = [("he", XS[0]), ("map", XS[1])] if role == "treated" else [("map", XS[2])]
            for kind, xk in kinds:
                ax = add_axes(fig, xk, y0, IMW, IMH)
                if kind == "he":
                    im = he_view(data_dir / "he" / f"{x.slide_id}.jpg")
                    crop = im[int(r0 * px):int((r1 + 1) * px), int(c0 * px):int((c1 + 1) * px)]
                    ax.imshow(crop, extent=ext, interpolation="antialiased")
                    lc, la = INK, 0.8
                else:
                    ax.imshow(lattice(t, "score"), extent=ext, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
                    lc, la = "#5E5872", 0.9
                cs = ax.contour(Xc, Yc, np.ma.masked_invalid(D), levels=list(REGION_BORDERS), colors=[lc],
                                alpha=la, linewidths=1.0, linestyles="dashed")
                cs.set_linestyle([DASH])
                ax.set_xlim(cx - hw, cx + hw)
                ax.set_ylim(cy + hh, cy - hh)
                ax.set_aspect("auto")
                clean(ax)
                if i == 1 and kind == "he":
                    yb = y0 + IMH + 2.4
                    line(fig, XS[0] + 1.0, yb, XS[0] + 1.0 + 2 * k, yb, color=INK, solid_capstyle="butt")
                    text(fig, XS[0] + 2.2 + 2 * k, yb + 0.9, "2 mm")
                if i == 0 and kind == "he":
                    region_labels(fig, ax, t, cx, cy, hw, hh, XS[0], y0, IMW, IMH, XS[0] - 1.0, y_min=y0 + 9.6)
            g = int(x.max_lesion_grade)
            text(fig, kinds[-1][1] + IMW / 2, y0 + IMH + 3.0,
                 "Pathology: " + (f"{GRADE_WORD[g]} lesion" if g else "no lesion"), ha="center")
        text(fig, 2, y0 + 2.6, label, size=FS_AX)
        text(fig, 2, y0 + 5.7, f"{dose} dose, {time.split()[0]} d", color=MUTED)
        xb = XS[2] + IMW + 1.3
        cax = add_axes(fig, xb, y0 + 3.4, 1.4, IMH - 6.8)
        cb = fig.colorbar(ScalarMappable(Normalize(lo, hi), cmap), cax=cax)
        cb.set_ticks([])
        cb.outline.set_visible(False)
        text(fig, xb + 0.7, y0 + 2.5, "High", color=MUTED, ha="center")
        text(fig, xb + 0.7, y0 + IMH - 0.4, "Low", color=MUTED, ha="center")
    yk = rows_y[1] + IMH + 7.0
    line(fig, 2, yk - 0.6, 6, yk - 0.6, color=INK, ls=DASH)
    text(fig, 7.2, yk, "Region borders", color=MUTED)
    return yk


def panel_b(fig, region, region_rats, top, X0=116.0):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    head(fig, X0, top, "b", "Change by region")
    AX_X, AX_Y, AX_H = X0 + 11.0, top + 19.0, 43.0
    AX_W = 182.0 - AX_X
    fig.legend([Patch(facecolor=REGION_COLOR[z], edgecolor="none") for z in REGIONS],
               [REGION_NAME[z] for z in REGIONS], loc="center left", ncol=3, bbox_to_anchor=(fx(X0), fy(top + 5.6)),
               fontsize=FS_TEXT, labelcolor=INK, borderaxespad=0, borderpad=0, handlelength=1.1, handleheight=0.8,
               handletextpad=0.4, columnspacing=1.0)
    hs = [Line2D([0], [0], marker="D", lw=0, ms=4.0, mfc=INK, mec=INK, mew=1.0),
          Line2D([0], [0], marker="o", lw=0, ms=4.4, mfc=PANEL, mec=INK, mew=1.0),
          Line2D([0], [0], marker="o", lw=0, ms=2.0, color=INK, alpha=0.6, mew=0)]
    fig.legend(hs, ["Mean, region of expected injury", "Mean, other regions", "One rat with a lesion"],
               loc="center left", ncol=2, bbox_to_anchor=(fx(X0), fy(top + 11.4)), fontsize=FS_TEXT, labelcolor=MUTED,
               borderaxespad=0, borderpad=0, handlelength=1.6, handleheight=0.8, handletextpad=0.4,
               columnspacing=1.6, labelspacing=0.35)
    ax = add_axes(fig, AX_X, AX_Y, AX_W, AX_H)
    vals = []
    for gi, (cp, _, _, _, expected) in enumerate(REGION_SETS):
        for zi, z in enumerate(REGIONS):
            v = region_rats[(region_rats.compound == cp) & (region_rats.region == z)].z.to_numpy()
            m = region[(region.compound == cp) & (region.region == z)].mean_z.iloc[0]
            x = gi * 4 + zi
            jit = np.random.default_rng(gi * 10 + zi).uniform(-0.2, 0.2, len(v))
            ax.scatter(x + jit, v, s=2.6, color=INK, alpha=0.6, lw=0, zorder=3)
            ax.scatter([x], [m], marker="D" if z == expected else "o", s=30 if z == expected else 34, linewidths=1.0,
                       facecolors=REGION_COLOR[z] if z == expected else PANEL, edgecolors=REGION_COLOR[z], zorder=5)
            vals.append(v)
    v = np.concatenate(vals)
    yhi = v.max() + 2.4
    ax.set_ylim(v.min() - 0.5, yhi)
    ax.set_xlim(-0.7, 10.7)
    ax.axhline(0, color=GREY, lw=1.0, zorder=1)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_ylabel("Standardized change\nvs matched controls", fontsize=FS_AX, labelpad=2)
    ticks(ax)
    text(fig, AX_X - 2.0, AX_Y - 1.6, "↑ more injury or tubule loss", color=MUTED)
    to_x = lambda xd: AX_X + (xd + 0.7) / 11.4 * AX_W
    for gi, (cp, labels, words, _, _) in enumerate(REGION_SETS):
        r = region[region.compound == cp].iloc[0]
        lines = [(s, FS_HEAD, INK) for s in labels] + [(words, FS_TEXT, MUTED), (f"{r.n_lesion} rats", FS_TEXT, MUTED)]
        for li, (s, fs, c) in enumerate(lines):
            text(fig, to_x(gi * 4 + 1), AX_Y + AX_H + 3.0 + li * 2.75, s, size=fs, color=c, ha="center")
        yb = yhi - 1.2
        ax.plot([gi * 4 - 0.39, gi * 4 + 2.39], [yb, yb], color=INK, lw=1.0)
        ax.text(gi * 4 + 1, yb + 0.25, style.fmt_p(r.expected_vs_other_p), fontsize=FS_TEXT, color=INK, ha="center",
                va="bottom")
    return AX_Y + AX_H + 3.0 + 4 * 2.75


def grade_key(fig, x, y):
    from matplotlib.lines import Line2D
    text(fig, x, y, "Maximum lesion grade", size=FS_HEAD)
    hs = [Line2D([0], [0], marker="o", lw=0, ms=4.4, mfc=GRADE_COLOR[v], mec=INK, mew=1.0) for v in range(5)]
    fig.legend(hs, [GRADE_LABEL[v] for v in range(5)], loc="upper left", ncol=1,
               bbox_to_anchor=(fx(x - 0.4), fy(y + 1.4)), fontsize=FS_TEXT, labelcolor=INK, borderaxespad=0,
               borderpad=0, handlelength=1.0, handletextpad=0.5, labelspacing=0.45)


def panel_c(fig, course, top, X0=2.0):
    from matplotlib.ticker import NullLocator
    head(fig, X0, top, "c", "Scores rise with the duration of dosing")
    AX0, AW, AG, AY, AH = X0 + 15.0, 37.0, 6.5, top + 11.5, 24.0
    days = [d for _, d in DAYS]
    for k, (cp, label, _, words, _, _) in enumerate(TIME_COURSES):
        g = course[course.compound == cp]
        x = AX0 + k * (AW + AG)
        ax = add_axes(fig, x, AY, AW, AH)
        ax.axhline(0, color=GREY, lw=1.0, zorder=1)
        ax.plot(days, g.z, color=MUTED, lw=1.0, zorder=2)
        ax.scatter(days, g.z, s=26, c=[GRADE_COLOR[v] for v in g.max_lesion_grade], edgecolors=INK, linewidths=1.0,
                   zorder=3)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticks(days)
        ax.set_xticklabels([str(d) for d in days])
        ax.set_xlim(3.2, 37)
        ax.set_ylim(*C_YLIM)
        ax.set_yticks([-2, 0, 2, 4, 6])
        if k == 0:
            ax.set_ylabel("Standardized change\nvs matched controls", fontsize=FS_AX, labelpad=2)
            ax.text(0.04, 0.97, "↑ more damage", transform=ax.transAxes, fontsize=FS_TEXT, color=MUTED, ha="left",
                    va="top")
        else:
            ax.set_yticklabels([])
        ticks(ax)
        ax.text(0.97, 0.04, f"Spearman ρ = {g.rho.iloc[0]:.2f}", transform=ax.transAxes, fontsize=FS_TEXT,
                color=INK, ha="right", va="bottom")
        text(fig, x + AW / 2, top + 6.2, label, size=FS_AX, ha="center")
        text(fig, x + AW / 2, top + 9.3, words, color=MUTED, ha="center")
    xr = AX0 + 3 * AW + 2 * AG
    y0 = AY + AH * C_YLIM[1] / (C_YLIM[1] - C_YLIM[0])
    text(fig, xr + 1.0, y0 - 0.3, "Matched", color=GREY)
    text(fig, xr + 1.0, y0 + 2.4, "controls", color=GREY)
    text(fig, AX0 + (3 * AW + 2 * AG) / 2, AY + AH + 6.4, "Repeated daily dosing (d)", size=FS_AX, ha="center")
    grade_key(fig, xr + 14.0, AY + 2.0)
    return AY + AH + 6.4


def panel_d(fig, rats, top, X0=2.0):
    head(fig, X0, top, "d", "Injury score tracks lesion severity")
    score = rats[LESION_SCORE].to_numpy(float)
    grade = rats.max_lesion_grade.to_numpy()
    AY, AH, BX, BW = top + 10.5, 35.0, X0 + 15.0, 61.0
    ax = add_axes(fig, BX, AY, BW, AH)
    for gv in range(5):
        v = score[grade == gv]
        bp = ax.boxplot(v, positions=[gv], widths=0.58, showfliers=False, patch_artist=True, whis=1.5,
                        medianprops=dict(color=PANEL if gv >= 3 else INK, lw=1.0),
                        boxprops=dict(lw=1.0, color=RULE), whiskerprops=dict(lw=1.0, color=RULE),
                        capprops=dict(lw=1.0, color=RULE))
        bp["boxes"][0].set_facecolor(GRADE_COLOR[gv])
        jit = np.random.default_rng(gv).uniform(-0.2, 0.2, len(v))
        ax.scatter(gv + jit, v, s=1.3, color=INK, alpha=0.3 if gv == 0 else 0.6, lw=0, zorder=3)
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"{GRADE_LABEL[g]}\n{int((grade == g).sum())}" for g in range(5)])
    ticks(ax)
    ax.tick_params(axis="x", length=0, pad=2.5)
    ax.axhline(0, color=GREY, lw=1.0, zorder=1)
    ax.set_xlim(-0.6, 4.6)
    ax.set_ylim(np.floor(score.min()) - 0.3, score.max() + 0.8)
    ax.set_ylabel("Injury score\n(whole section, standardized)", fontsize=FS_AX, labelpad=2)
    text(fig, BX - 2.0, AY - 1.6, "↑ more injury", color=MUTED)
    lo, hi = ax.get_ylim()
    y0 = AY + AH * hi / (hi - lo)
    text(fig, BX + BW + 1.0, y0 - 0.3, "Matched", color=GREY)
    text(fig, BX + BW + 1.0, y0 + 2.4, "controls", color=GREY)
    text(fig, BX + BW / 2, AY + AH + 8.8, "Maximum lesion grade (number of rats)", size=FS_AX, ha="center")


def panel_e(fig, curves, auc, top, X0=92.0):
    from matplotlib.lines import Line2D
    head(fig, X0, top, "e", "Detection of lesions and kidney dysfunction")
    SZ, RX, AY = 34.0, X0 + 9.0, top + 11.0
    ax = add_axes(fig, RX, AY, SZ, SZ)
    ax.plot([0, 1], [0, 1], color=mix(GREY, PANEL, 35), lw=1.0, ls=DASH, zorder=1)
    ax.text(0.78, 0.71, "Chance", rotation=45, rotation_mode="anchor", fontsize=FS_TEXT, color=GREY, ha="center",
            va="center", transform=ax.transAxes)
    names = [name for name, shown in DETECTION if shown]
    for name, c in zip(names, [ROC_LESION, ROC_BUN]):
        cv = curves[curves.curve == name]
        ax.plot(cv.fpr, cv.tpr, color=c, lw=1.1, zorder=3)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_yticklabels(["0", "0.5", "1"])
    ticks(ax)
    ax.set_xlabel("1 − specificity", fontsize=FS_AX, labelpad=2)
    ax.set_ylabel("Sensitivity", fontsize=FS_AX, labelpad=2)
    ax.set_aspect("equal")
    les, bun = (auc[auc.comparison == n].iloc[0] for n in names)
    LX = RX + SZ + 3.5
    text(fig, LX, AY + 1.6, "Injury score", color=MUTED)
    ci = lambda r: f"(95% CI {r.ci_lo:.2f}–{r.ci_hi:.2f})"
    entries = [(ROC_LESION, ["Moderate or severe lesion", f"({les.n_pos} rats) vs no lesion ({les.n_neg})",
                             f"AUC {les.auc:.2f} {ci(les)}"]),
               (ROC_BUN, ["Blood urea nitrogen ≥2× controls", f"({bun.n_pos} rats) vs all others ({bun.n_neg})",
                          f"AUC {bun.auc:.2f} {ci(bun)}"])]
    y = AY + 6.4
    for c, lines in entries:
        fig.add_artist(Line2D([fx(LX), fx(LX + 3.6)], [fy(y - 0.8)] * 2, color=c, lw=1.4, transform=fig.transFigure))
        for li, s in enumerate(lines):
            text(fig, LX + 5.0, y + li * 2.75, s)
        y += 3 * 2.75 + 2.6


def draw(tables, data_dir):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(W_MM * style.MM, H_MM * style.MM))
    ya = panel_a(fig, tables["map_examples"], data_dir, 4.0)
    yb = panel_b(fig, tables["region_change"], tables["region_change_rats"], 4.0)
    top = max(ya, yb) + 5.5
    top = panel_c(fig, tables["time_course"], top) + 6.0
    panel_d(fig, tables["lesion_score_rats"], top)
    panel_e(fig, tables["roc"], tables["detection_auc"], top)
    return fig
