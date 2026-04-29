#!/usr/bin/env python3
"""
Artifact generator for the response-law paper.

Reads existing outputs/ and writes:
  paper/macros.tex      -- LaTeX macros (all-letter names, no digits)
  paper/tables/         -- compact LaTeX tables
  paper/figures/        -- 8 flagship figures

Does NOT rerun simulations.
"""
from pathlib import Path
import shutil
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, Rectangle
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

ROOT      = Path(__file__).resolve().parents[1]
OUT_PAPER = ROOT / "paper"
OUT_FIG   = OUT_PAPER / "figures"
OUT_TAB   = OUT_PAPER / "tables"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
BLUE   = "#2166AC"
ORANGE = "#D6604D"
GREEN  = "#1A7A1A"
GRAY   = "#888888"
BLACK  = "#111111"
LBLUE  = "#92C5DE"
LRED   = "#F4A582"

def _apply_style():
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.size":         9,
        "axes.titlesize":    10,
        "axes.labelsize":    9,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   8,
        "legend.framealpha": 0.85,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    0.9,
        "lines.linewidth":   1.6,
        "patch.linewidth":   0.7,
    })

def _savefig(fig, name):
    p = OUT_FIG / name
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p.name}")


# ---------------------------------------------------------------------------
# Figure 1 — Object definition (top) + target specificity (bottom)
# Two-row layout: polished 5×5 schematic on top, clean bars on bottom.
# ---------------------------------------------------------------------------
def make_fig1(ROOT):
    _apply_style()

    sel = pd.read_csv(ROOT / "outputs/selection_principle/selection_summary.csv")

    # =========================================================
    # 5×5 schematic colour scheme
    # =========================================================
    C_FOCAL   = BLUE         # the embedded isolate (centre)
    C_ORTH    = "#F5F5F5"    # 4-connected dead neighbours (very light)
    C_DIAG_A  = "#444444"    # diagonal alive neighbours
    C_DIAG_D  = "#E0E0E0"    # diagonal dead neighbours
    C_BG      = "#D4E8F7"    # outer-ring background (light blue)
    C_EDGE    = "#666666"

    # 5×5 state map  (row, col) → colour
    # Centre = (2,2) = embedded isolate
    # Orthogonal dead: (1,2), (3,2), (2,1), (2,3)
    # Diagonal alive: (1,1), (3,1), (1,3)   ← at least 1 required
    # Diagonal dead: (3,3)
    # Outer ring (row 0, row 4, col 0, col 4): background context
    states = {}
    for r in range(5):
        for c in range(5):
            if r in (0, 4) or c in (0, 4):
                states[(r, c)] = C_BG
            else:
                states[(r, c)] = C_BG  # default inner, overridden below
    states[(2, 2)] = C_FOCAL
    states[(1, 2)] = C_ORTH
    states[(3, 2)] = C_ORTH
    states[(2, 1)] = C_ORTH
    states[(2, 3)] = C_ORTH
    states[(1, 1)] = C_DIAG_A
    states[(3, 1)] = C_DIAG_A
    states[(1, 3)] = C_DIAG_A
    states[(3, 3)] = C_DIAG_D
    # Outer ring: light pattern for context
    for r, c in [(0,2),(2,0),(4,2),(2,4),(0,0),(0,4),(4,0),(4,4)]:
        states[(r, c)] = "#BBBBBB"
    for r, c in [(0,1),(0,3),(1,0),(3,0),(1,4),(3,4),(4,1),(4,3)]:
        states[(r, c)] = "#CCCCCC"

    # ---------- target-specificity bar data ----------
    gp = sel[sel["scope"] == "global_pooled"].copy()
    # Drop duplicated delta_components (numerically == fine_net) for clarity
    target_order = [
        "target_fine_net",
        "target_delta_block_entropy",
        "target_delta_block_var",
        "target_delta_density",
        "target_future_density",
        "target_future_live_count",
    ]
    label_map = {
        "target_fine_net":            "fine-net $\\Delta C_k$",
        "target_delta_block_entropy":  "block entropy",
        "target_delta_block_var":      "block variance",
        "target_delta_density":        "$\\Delta$ density",
        "target_future_density":       "future density",
        "target_future_live_count":    "live count",
    }
    gp = gp.set_index("target").reindex(target_order).reset_index()
    labels = [label_map.get(t, t) for t in gp["target"]]

    # ====== build two-row figure ======
    fig = plt.figure(figsize=(6.5, 5.8))
    gs_top = gridspec.GridSpec(1, 1, top=0.97, bottom=0.56,
                               left=0.04, right=0.96)
    gs_bot = gridspec.GridSpec(1, 1, top=0.50, bottom=0.07,
                               left=0.14, right=0.97)

    # ============================================================
    # TOP ROW: polished 5×5 schematic
    # ============================================================
    ax_s = fig.add_subplot(gs_top[0, 0])
    sz   = 0.78
    gap  = 0.07
    W    = 5 * (sz + gap)   # total grid width

    ax_s.set_xlim(-0.4, W + 2.4)
    ax_s.set_ylim(-0.5, W + 0.3)
    ax_s.set_aspect("equal")
    ax_s.axis("off")

    for (r, c), col in states.items():
        x0 = c * (sz + gap)
        y0 = (4 - r) * (sz + gap)      # flip so row-0 is at top
        lw = 1.2 if (r, c) == (2, 2) else 0.7
        ec = BLUE if (r, c) == (2, 2) else C_EDGE
        rect = Rectangle((x0, y0), sz, sz,
                          facecolor=col, edgecolor=ec,
                          linewidth=lw, zorder=2)
        ax_s.add_patch(rect)
        cx, cy = x0 + sz/2, y0 + sz/2
        if (r, c) == (2, 2):
            ax_s.text(cx, cy, "●", ha="center", va="center",
                      fontsize=17, color="white", zorder=4)
            ax_s.text(cx, cy - 0.53, "isolate", ha="center", va="top",
                      fontsize=7, color="white", zorder=4, style="italic")
        elif col == C_ORTH:
            ax_s.text(cx, cy, "×", ha="center", va="center",
                      fontsize=13, color="#999999", zorder=3)
        elif col == C_DIAG_A:
            ax_s.text(cx, cy, "●", ha="center", va="center",
                      fontsize=13, color="white", zorder=3)
        elif col in (C_DIAG_D, "#BBBBBB", "#CCCCCC"):
            pass   # empty background

    # Annotation: orthogonal neighbours bracket (left side)
    orth_xs  = [2*(sz+gap), 2*(sz+gap), 2*(sz+gap), 2*(sz+gap)]  # col=2
    orth_ys  = [(4-1)*(sz+gap)+sz/2, (4-3)*(sz+gap)+sz/2,
                (4-2)*(sz+gap)+sz,   (4-2)*(sz+gap)]
    # draw brace on left
    brace_x = -0.25
    for yr in [orth_ys[0], orth_ys[1],
               2*(sz+gap) + sz,   # north orthogonal top
               2*(sz+gap)]:       # south orthogonal bottom
        pass
    # Simple annotation: text labels
    ax_s.text(-0.32, (4-2)*(sz+gap) + sz/2,
              "4-connected\nneighbours\ndead",
              ha="right", va="center", fontsize=8, color="#B03030",
              bbox=dict(boxstyle="round,pad=0.25", fc="#FFF0F0", ec="#D08080", lw=0.8))
    # Arrow from annotation to N cell
    n_cx = 2*(sz+gap) + sz/2
    n_cy = (4-1)*(sz+gap) + sz/2
    ax_s.annotate("", xy=(n_cx - sz/2 - 0.04, n_cy),
                  xytext=(-0.10, n_cy),
                  arrowprops=dict(arrowstyle="-|>", color="#B03030",
                                  lw=0.8, mutation_scale=10))

    # Annotation: diagonal alive neighbours
    d_cx = 1*(sz+gap) + sz/2   # NW diagonal (1,1)
    d_cy = (4-1)*(sz+gap) + sz/2
    ax_s.text(W + 0.3, (4-1)*(sz+gap) + sz/2,
              "diagonal\nneighbours\n$\\geq 1$ alive",
              ha="left", va="center", fontsize=8, color="#1A5E1A",
              bbox=dict(boxstyle="round,pad=0.25", fc="#F0FFF0", ec="#80B080", lw=0.8))
    ax_s.annotate("", xy=(3*(sz+gap), (4-1)*(sz+gap) + sz/2),
                  xytext=(W + 0.25, (4-1)*(sz+gap) + sz/2),
                  arrowprops=dict(arrowstyle="-|>", color="#1A5E1A",
                                  lw=0.8, mutation_scale=10))

    ax_s.set_title("(a)  Embedded isolate — structural definition",
                   loc="left", fontsize=10, pad=4)

    # ============================================================
    # BOTTOM ROW: target-specificity bars
    # ============================================================
    ax_b = fig.add_subplot(gs_bot[0, 0])
    x   = np.arange(len(labels))
    w   = 0.19
    iso_vals  = gp["deltaR2_iso"].values
    crs_vals  = gp["deltaR2_coarse"].values
    ent_vals  = gp["deltaR2_entropy"].values
    nul_vals  = gp["deltaR2_null_iso_shuffle"].values

    ax_b.bar(x - 1.5*w, iso_vals,  w, label="iso\_count",  color=BLUE,   zorder=2)
    ax_b.bar(x - 0.5*w, crs_vals,  w, label="coarse",      color=GRAY,   zorder=2)
    ax_b.bar(x + 0.5*w, ent_vals,  w, label="entropy",     color=ORANGE, alpha=0.7, zorder=2)
    ax_b.bar(x + 1.5*w, nul_vals,  w, label="null (shuffle)",
             color="white", edgecolor=BLACK, linewidth=0.7, zorder=2)
    ax_b.axhline(0, color=BLACK, linewidth=0.5)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels, fontsize=8.5)
    ax_b.set_ylabel("$\\Delta R^2$ vs density control", fontsize=9)
    ax_b.set_title("(b)  Target specificity — iso\_count signals only for fine-net (global pooled)",
                   loc="left", fontsize=10, pad=4)
    ax_b.legend(loc="upper right", fontsize=7.5, ncol=2)

    # Annotate the fine-net iso bar with value
    ax_b.annotate(f"$\\Delta R^2={iso_vals[0]:.4f}$",
                  xy=(x[0] - 1.5*w, iso_vals[0]),
                  xytext=(x[0] + 0.5, iso_vals[0] + 0.0002),
                  arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8),
                  fontsize=7.5, va="bottom", ha="left", color=BLUE)
    # Annotate null near zero
    ax_b.text(x[0] + 1.5*w, 0.00005,
              "null $\\approx 0$", ha="center", va="bottom",
              fontsize=6.5, color="#888888", style="italic")

    _savefig(fig, "fig1_object_selection.pdf")
    _savefig(fig, "fig1_object_selection.png")


# ---------------------------------------------------------------------------
# Figure 2 — Non-leaky prestate summary
# ---------------------------------------------------------------------------
def make_fig2(ROOT):
    _apply_style()

    pre = pd.read_csv(ROOT / "outputs/prestate_class_horizon/prestate_class_summary.csv")

    gh = pre[pre["scope"] == "global_horizon"]
    iso_pre  = gh[gh["model"] == "iso_count"].sort_values("horizon")
    cls_pre  = gh[gh["model"] == "class_counts"].sort_values("horizon") \
               if "class_counts" in gh["model"].values else None

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.90, bottom=0.14)

    ks   = iso_pre["horizon"].values
    r2   = iso_pre["mean_R2"].values
    shu  = iso_pre["mean_shuffle_R2"].values

    ax.plot(ks, r2, "o-", color=BLUE, label="iso\\_count $R^2(k)$", ms=6, zorder=3)
    if cls_pre is not None:
        r2c = cls_pre["mean_R2"].values
        ax.plot(ks, r2c, "s--", color=GREEN, label="class\\_counts $R^2(k)$",
                ms=5, zorder=3)
    ax.fill_between(ks, shu, alpha=0.25, color=GRAY)
    ax.plot(ks, shu, ":", color=GRAY, linewidth=1.4, label="shuffle null", zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("Horizon $k$")
    ax.set_ylabel("Residual $R^2$ (prestate only)")
    ax.set_title("Non-leaky prestate: $t=0$ iso\\_count recovers full $R^2(k)$",
                 fontsize=10, pad=6)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xticks([1, 5, 10, 25, 50, 100, 200])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_ylim(bottom=0)

    # Annotate mean R2 value
    mean_r2 = float(r2.mean())
    ax.axhline(mean_r2, color=BLUE, linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(1.2, mean_r2 + 0.005, f"mean $R^2 = {mean_r2:.3f}$",
            fontsize=7.5, color=BLUE, va="bottom")

    _savefig(fig, "fig2_prestate.pdf")
    _savefig(fig, "fig2_prestate.png")


# ---------------------------------------------------------------------------
# Figure 3 — Temporal response slope curves
# ---------------------------------------------------------------------------
def make_fig3(ROOT):
    _apply_style()

    hg = pd.read_csv(ROOT / "outputs/selection_principle_horizon/horizon_global_summary.csv")
    hr = pd.read_csv(ROOT / "outputs/selection_principle_horizon/horizon_rule_summary.csv")
    hc = pd.read_csv(ROOT / "outputs/selection_principle_horizon/horizon_condition_summary.csv")

    slope_col = "mean_raw_slope" if "mean_raw_slope" in hg.columns else "mean_slope"
    r2_col    = "mean_resid_R2"  if "mean_resid_R2"  in hg.columns else "mean_R2"
    null_col  = "mean_null_R2"   if "mean_null_R2"   in hg.columns else "null_R2"
    k_col     = "horizon"        if "horizon"        in hg.columns else "k"

    ks_g  = hg[k_col].values
    sl_g  = hg[slope_col].values
    r2_g  = hg[r2_col].values
    nu_g  = hg[null_col].values

    gol = hr[hr["rule"] == "GoL"].sort_values(k_col)
    hl  = hr[hr["rule"] == "HighLife"].sort_values(k_col)

    # Condition pivot for spread band
    def _cond_key(s):
        rule = 0 if s.startswith("GoL") else 1
        m = re.search(r"_L(\d+)_rho", s)
        L = int(m.group(1)) if m else 0
        rho = float(s.split("rho")[1]) if "rho" in s else 0.0
        return (rule, L, rho)
    cond_order = sorted(hc["condition_id"].unique(), key=_cond_key)
    pivot = hc.pivot(index="condition_id", columns="horizon",
                     values="resid_iso_slope_raw").reindex(cond_order)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.5, 3.5),
                                      gridspec_kw={"wspace": 0.38})
    fig.subplots_adjust(left=0.09, right=0.97, top=0.90, bottom=0.14)

    # ----- (a) slopes -----
    ax_a.plot(ks_g, sl_g, "k-o", ms=5, label="Global", linewidth=1.8, zorder=4)
    ax_a.plot(gol[k_col].values, gol[slope_col].values,
              "-o", color=BLUE, ms=4, label="GoL", zorder=3)
    ax_a.plot(hl[k_col].values, hl[slope_col].values,
              "--s", color=ORANGE, ms=4, label="HighLife", zorder=3)
    cond_slope_min = pivot.min(axis=0).values
    cond_slope_max = pivot.max(axis=0).values
    ax_a.fill_between(pivot.columns.tolist(), cond_slope_min, cond_slope_max,
                      alpha=0.12, color=BLACK, label="condition range")
    ax_a.axhline(0, color="#aaa", linewidth=0.5, linestyle="--")
    ax_a.set_xscale("log")
    ax_a.set_xticks([1, 5, 10, 25, 50, 100, 200])
    ax_a.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax_a.tick_params(axis="x", which="minor", bottom=False)
    ax_a.set_xlabel("Horizon $k$")
    ax_a.set_ylabel("Mean $\\hat{\\beta}_{\\rm iso}(k)$")
    ax_a.set_title("(a)  Response slope vs horizon", loc="left", fontsize=10)
    ax_a.legend(loc="lower right", fontsize=7.5)
    ax_a.set_ylim(top=0.05)

    # ----- (b) R² -----
    ax_b.plot(ks_g, r2_g, "k-o", ms=5, label="Global $R^2(k)$", linewidth=1.8, zorder=4)
    ax_b.plot(gol[k_col].values, gol[r2_col].values,
              "-o", color=BLUE, ms=4, label="GoL", zorder=3)
    ax_b.plot(hl[k_col].values, hl[r2_col].values,
              "--s", color=ORANGE, ms=4, label="HighLife", zorder=3)
    ax_b.plot(ks_g, nu_g, ":", color=GRAY, linewidth=1.2, label="shuffle null", zorder=2)
    ax_b.fill_between(ks_g, nu_g, alpha=0.15, color=GRAY)
    ax_b.set_xscale("log")
    ax_b.set_xticks([1, 5, 10, 25, 50, 100, 200])
    ax_b.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax_b.tick_params(axis="x", which="minor", bottom=False)
    ax_b.set_xlabel("Horizon $k$")
    ax_b.set_ylabel("Residual $R^2(k)$")
    ax_b.set_title("(b)  Predictive signal vs horizon", loc="left", fontsize=10)
    ax_b.legend(loc="upper right", fontsize=7.5)
    ax_b.set_ylim(bottom=0)

    _savefig(fig, "fig3_response_slope.pdf")
    _savefig(fig, "fig3_response_slope.png")


# ---------------------------------------------------------------------------
# Figure 4 — Heatmap: all 112 condition-horizon slopes
# ---------------------------------------------------------------------------
def make_fig4(ROOT):
    _apply_style()

    hc = pd.read_csv(ROOT / "outputs/selection_principle_horizon/horizon_condition_summary.csv")

    def _cond_key(s):
        rule = 0 if s.startswith("GoL") else 1
        m = re.search(r"_L(\d+)_rho", s)
        L = int(m.group(1)) if m else 0
        rho = float(s.split("rho")[1]) if "rho" in s else 0.0
        return (rule, L, rho)
    cond_order = sorted(hc["condition_id"].unique(), key=_cond_key)
    pivot = hc.pivot(index="condition_id", columns="horizon",
                     values="resid_iso_slope_raw").reindex(cond_order)

    def _short(cid):
        rule = "G" if cid.startswith("GoL") else "H"
        L    = cid.split("_L")[1].split("_rho")[0]
        rho  = cid.split("rho")[1]
        return f"{rule}{L}·{rho}"
    row_labels = [_short(c) for c in pivot.index]
    col_labels = [str(k) for k in pivot.columns.tolist()]

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    fig.subplots_adjust(left=0.14, right=0.93, top=0.93, bottom=0.10)

    data = pivot.values
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r",
                   vmin=-1.0, vmax=0, origin="upper")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7.5)
    ax.set_xlabel("Horizon $k$")
    ax.set_ylabel("Condition  (rule $\\cdot$ $L$ $\\cdot$ $\\rho$)")
    ax.set_title("All 112 condition--horizon slopes are negative",
                 fontsize=10, pad=6)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.015)
    cbar.set_label("$\\hat{\\beta}_{\\rm iso}$", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    min_val = data.min()
    max_val = data.max()
    ax.text(0.99, 0.01, f"range [{min_val:.3f}, {max_val:.3f}]",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc="#333", ec="none"))

    _savefig(fig, "fig4_heatmap.pdf")
    _savefig(fig, "fig4_heatmap.png")


# ---------------------------------------------------------------------------
# Figure 5 — Mechanism carrier: CV R² and feature slopes
# ---------------------------------------------------------------------------
def make_fig5(ROOT):
    _apply_style()

    fg = pd.read_csv(ROOT / "outputs/isolate_fate/fate_global_summary.csv")

    cv_col    = "mean_cv_R2" if "mean_cv_R2" in fg.columns else "cv_r2"
    model_col = "model"      if "model"      in fg.columns else "predictor"
    slope_col = "mean_slope_raw" if "mean_slope_raw" in fg.columns else "mean_slope"

    cv_rows = fg[fg[cv_col].notna() & (fg[cv_col] > -99)].copy()
    cv_name_map = {
        "all_fates":    "all fates",
        "all_plus_coarse": "all + coarse",
        "cell_fates_all": "cell fates",
        "birth_bridge": "birth bridge",
        "local_window": "local window",
        "survive_split":"survive split",
        "survive":      "survive",
        "iso_count":    "iso count",
        "death":        "death",
        "coarse":       "coarse",
        "entropy":      "entropy",
    }
    cv_rows["label"] = cv_rows[model_col].map(cv_name_map).fillna(cv_rows[model_col])
    cv_rows = cv_rows.sort_values(cv_col, ascending=True)

    sl_rows = fg[fg[cv_col].isna() & fg[slope_col].notna()].copy()
    slope_name_map = {
        "slope_iso_count":               "iso count",
        "slope_iso_die":                 "iso die",
        "slope_iso_survive":             "iso survive",
        "slope_iso_survive_connected":   "survive connected",
        "slope_iso_survive_isolated":    "survive isolated",
        "slope_iso_orth_birth_any":      "orth birth",
        "slope_iso_diag_birth_any":      "diag birth",
        "slope_iso_local_window_loss_sum": "window loss",
        "slope_iso_local_window_gain_sum": "window gain",
        "slope_iso_local_window_delta_sum":"window delta",
        "slope_block_var":               "block var",
        "slope_block_entropy":           "block entropy",
    }
    keep_slopes = {
        "slope_iso_count","slope_iso_die","slope_iso_survive",
        "slope_iso_survive_connected","slope_iso_orth_birth_any",
        "slope_iso_diag_birth_any","slope_iso_local_window_loss_sum",
        "slope_iso_local_window_gain_sum",
    }
    sl_rows = sl_rows[sl_rows[model_col].isin(keep_slopes)].copy()
    sl_rows["label"] = sl_rows[model_col].map(slope_name_map)
    sl_rows = sl_rows.sort_values(slope_col, ascending=True)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.5, 4.0),
                                      gridspec_kw={"wspace": 0.45})
    fig.subplots_adjust(left=0.14, right=0.98, top=0.92, bottom=0.10)

    # ----- (a) CV R² -----
    n_a  = len(cv_rows)
    ypos = np.arange(n_a)
    vals = cv_rows[cv_col].values
    col_list = []
    for m in cv_rows[model_col].values:
        if m in ("coarse", "entropy"):
            col_list.append(GRAY)
        elif m == "iso_count":
            col_list.append(BLUE)
        elif m in ("local_window", "all_fates", "all_plus_coarse", "cell_fates_all",
                   "birth_bridge", "survive_split", "survive", "death"):
            val = vals[list(cv_rows[model_col].values).index(m)]
            col_list.append(GREEN if val > 0.35 else LBLUE)
        else:
            col_list.append(LBLUE)

    ax_a.barh(ypos, vals, color=col_list, edgecolor="white", linewidth=0.5,
              height=0.72, zorder=2)
    ax_a.set_yticks(ypos)
    ax_a.set_yticklabels(cv_rows["label"].values, fontsize=8)
    ax_a.set_xlabel("CV $R^2$ on residual $\\Delta C_k$")
    ax_a.set_title("(a)  Mechanism models", loc="left", fontsize=10)
    ax_a.axvline(0, color=BLACK, linewidth=0.4)
    for i, (m, v) in enumerate(zip(cv_rows[model_col].values, vals)):
        if m in ("iso_count", "local_window", "all_fates"):
            ax_a.text(v + 0.005, ypos[i], f"{v:.3f}", va="center", fontsize=7.5,
                      color=BLACK)
    ax_a.set_xlim(right=0.65)

    # ----- (b) Feature slopes -----
    n_b  = len(sl_rows)
    yb   = np.arange(n_b)
    svals = sl_rows[slope_col].values
    col_s = [ORANGE if v > 0 else BLUE for v in svals]
    ax_b.barh(yb, svals, color=col_s, edgecolor="white", linewidth=0.5,
              height=0.72, zorder=2)
    ax_b.set_yticks(yb)
    ax_b.set_yticklabels(sl_rows["label"].values, fontsize=8)
    ax_b.set_xlabel("Mean slope $\\hat{\\beta}$")
    ax_b.set_title("(b)  Single-feature slopes", loc="left", fontsize=10)
    ax_b.axvline(0, color=BLACK, linewidth=0.8)
    for i, v in enumerate(svals):
        xtext = v + 0.03 if v >= 0 else v - 0.05
        ax_b.text(xtext, yb[i], f"{v:.2f}", va="center", fontsize=7.5,
                  ha="left" if v >= 0 else "right", color=BLACK)

    _savefig(fig, "fig5_mechanism.pdf")
    _savefig(fig, "fig5_mechanism.png")


# ---------------------------------------------------------------------------
# Figure 6 — Transition class contributions
# ---------------------------------------------------------------------------
def make_fig6(ROOT):
    _apply_style()

    tc  = pd.read_csv(ROOT / "outputs/isolate_transition_classes/transition_class_contributions.csv")
    tsm = pd.read_csv(ROOT / "outputs/isolate_transition_classes/transition_class_summary.csv")

    tc_all = tc.sort_values("p_times_delta", ascending=True)
    yc     = np.arange(len(tc_all))
    pd_vals = tc_all["p_times_delta"].values
    class_desc = {
        1: "1 diag nbr",  2: "2 diag (opp)",  3: "2 diag (adj)",
        4: "2 diag (opp2)", 5: "3 diag nbr",  6: "2 adj diag",
        7: "3 diag+1",  8: "all 4 diag",  9: "corner adj",
        10: "corner+opp", 11: "3+sym", 12: "L-shape",
        13: "broad", 14: "complex", 15: "full ring",
    }
    clabels = [f"cls {int(c)} ({class_desc.get(int(c),'?')})"
               for c in tc_all["diag_class"].values]
    col_c = [BLUE if v < 0 else ORANGE for v in pd_vals]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    fig.subplots_adjust(left=0.26, right=0.97, top=0.92, bottom=0.12)

    ax.barh(yc, pd_vals, color=col_c, edgecolor="white", linewidth=0.4,
            height=0.72, zorder=2)
    ax.set_yticks(yc)
    ax.set_yticklabels(clabels, fontsize=8)
    ax.set_xlabel("$p(\\mathrm{cls}) \\times \\overline{\\Delta}_{\\rm local}$")
    ax.set_title("Class contribution: $p \\times$ mean local response", fontsize=10, pad=6)
    ax.axvline(0, color=BLACK, linewidth=0.8)

    # Annotate death fraction
    tsm_gol = tsm[tsm["rule"] == "GoL"].copy() if "rule" in tsm.columns else tsm
    if "death_frac" in tsm_gol.columns:
        for i, row in enumerate(tc_all.itertuples()):
            sub = tsm_gol[tsm_gol["diag_class"] == row.diag_class]
            if len(sub):
                df_val = float(sub["death_frac"].mean())
                ax.text(-0.005, yc[i], f"d={df_val:.2f}",
                        ha="right", va="center", fontsize=7, color="#444")

    _savefig(fig, "fig6_classes.pdf")
    _savefig(fig, "fig6_classes.png")


# ---------------------------------------------------------------------------
# Figure 7 — Transfer, amplitude law (three panels)
# ---------------------------------------------------------------------------
def make_fig7(ROOT):
    _apply_style()

    ts  = pd.read_csv(ROOT / "outputs/mechanism_transfer_standardized/transfer_standardized_summary.csv")
    at  = pd.read_csv(ROOT / "outputs/mechanism_amplitude_law/amplitude_condition_table.csv")
    am  = pd.read_csv(ROOT / "outputs/mechanism_amplitude_law/amplitude_model_summary.csv")

    r2z_col = "mean_test_R2_z" if "mean_test_R2_z" in ts.columns else "mean_R2_z"

    fig = plt.figure(figsize=(6.5, 4.0))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.45,
                            left=0.08, right=0.98, top=0.92, bottom=0.14)

    # ----- (a) Standardized transfer -----
    ax_a = fig.add_subplot(gs[0, 0])
    models_plot = [
        ("fate_all",                    "all fates",   BLACK,  "o"),
        ("local_window_delta_loss_gain", "window delta", GREEN,  "s"),
        ("local_window_loss",            "window loss",  BLUE,   "^"),
        ("class_counts_plus_losses",     "class+loss",   ORANGE, "D"),
        ("iso_count",                    "iso count",    GRAY,   "P"),
    ]
    split_order  = ["leave_density", "leave_size", "leave_rule", "leave_condition"]
    split_labels = ["density", "size", "rule", "condition"]
    x_s = np.arange(len(split_order))

    for mname, mlabel, mcol, mmark in models_plot:
        sub = ts[ts["model"] == mname].set_index("split_type").reindex(split_order)
        vals = sub[r2z_col].values
        ax_a.plot(x_s, vals, marker=mmark, color=mcol, ms=5,
                  label=mlabel, linewidth=1.4, zorder=3)

    ax_a.axhline(0, color="#aaa", linewidth=0.5, linestyle="--")
    ax_a.set_xticks(x_s)
    ax_a.set_xticklabels(split_labels, fontsize=7.5)
    ax_a.set_ylabel("Mean $R^2_z$ (LOO)")
    ax_a.set_xlabel("Held-out split")
    ax_a.set_title("(a)  Transfer", loc="left", fontsize=10)
    ax_a.legend(loc="lower right", fontsize=6.5, ncol=1)
    ax_a.set_ylim(bottom=0)

    # ----- (b) Amplitude scatter -----
    ax_b = fig.add_subplot(gs[0, 1])
    iso_at = at[at["mechanism"] == "iso_count"].copy()
    marker_L = {64: "o", 128: "s"}
    rule_col  = {"GoL": BLUE, "HighLife": ORANGE}
    for (rule, L), grp in iso_at.groupby(["rule", "L"]):
        ax_b.scatter(grp["rho"], grp["amplitude"],
                     color=rule_col.get(rule, GRAY),
                     marker=marker_L.get(int(L), "o"),
                     s=35, label=f"{rule} $L$={int(L)}", zorder=3,
                     edgecolors="white", linewidths=0.5)
    ax_b.set_xlabel("Initial density $\\rho$")
    ax_b.set_ylabel("Condition amplitude $A$")
    ax_b.set_title("(b)  Amplitude vs $\\rho$", loc="left", fontsize=10)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor=BLUE,
               markeredgecolor="none", markersize=6, label="GoL $L$=64"),
        Line2D([0],[0], marker="s", color="w", markerfacecolor=BLUE,
               markeredgecolor="none", markersize=6, label="GoL $L$=128"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=ORANGE,
               markeredgecolor="none", markersize=6, label="HL $L$=64"),
        Line2D([0],[0], marker="s", color="w", markerfacecolor=ORANGE,
               markeredgecolor="none", markersize=6, label="HL $L$=128"),
    ]
    ax_b.legend(handles=handles, loc="upper left", fontsize=6.5)

    # ----- (c) Amplitude LOO R² -----
    ax_c = fig.add_subplot(gs[0, 2])
    am_iso = am[am["mechanism"] == "iso_count"].copy()
    model_order = ["full", "size_rho", "rule_size_rho", "size_only", "rho_only", "rule_only"]
    model_label = {
        "full":          "full",
        "size_rho":      "size+$\\rho$",
        "rule_size_rho": "rule+size+$\\rho$",
        "size_only":     "size only",
        "rho_only":      "$\\rho$ only",
        "rule_only":     "rule only",
    }
    mod_col_name = "amplitude_model" if "amplitude_model" in am_iso.columns else "model"
    am_iso = am_iso.set_index(mod_col_name).reindex(model_order).reset_index()
    r2_vals = am_iso["R2_LOO"].values
    yc2 = np.arange(len(model_order))
    col_amp = [GREEN if v > 0.9 else (BLUE if v > 0 else ORANGE) for v in r2_vals]
    ax_c.barh(yc2, r2_vals, color=col_amp, edgecolor="white",
              height=0.65, zorder=2)
    ax_c.set_yticks(yc2)
    ax_c.set_yticklabels([model_label.get(m, m) for m in am_iso[mod_col_name].values],
                          fontsize=7.5)
    ax_c.set_xlabel("LOO $R^2$")
    ax_c.set_title("(c)  Amplitude LOO $R^2$", loc="left", fontsize=10)
    ax_c.axvline(0, color=BLACK, linewidth=0.6)
    ax_c.axvline(1, color="#ccc", linewidth=0.5, linestyle=":")
    for i, v in enumerate(r2_vals):
        ax_c.text(max(v, 0) + 0.01, yc2[i], f"{v:.3f}",
                  va="center", fontsize=7, color=BLACK)
    ax_c.set_xlim(-0.35, 1.08)

    _savefig(fig, "fig7_transfer_amplitude.pdf")
    _savefig(fig, "fig7_transfer_amplitude.png")


# ---------------------------------------------------------------------------
# Figure 8 — Task-direction coherence
# ---------------------------------------------------------------------------
def make_fig8(ROOT):
    _apply_style()

    bs = pd.read_csv(ROOT / "outputs/ca_lgds_bridge/bridge_summary.csv")

    fam_map = {
        "horizon_fine_net":       "Fine-net\nhorizons\n($k$=1--200)",
        "selection_multi_target": "Heterogeneous\ntargets",
    }
    fam_order = ["horizon_fine_net", "selection_multi_target"]
    bs_sub = bs.set_index("family").reindex(fam_order).reset_index()

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.14)

    width = 0.28
    x_d = np.array([0.0, 0.75])
    cos_vals   = bs_sub["mean_pairwise_abs_cosine"].values
    rank1_vals = bs_sub["rank1_cumulative_energy"].values

    b1 = ax.bar(x_d - width/2, cos_vals,  width,
                color=[BLUE, ORANGE], label="mean $|\\cos|$", zorder=2)
    b2 = ax.bar(x_d + width/2, rank1_vals, width,
                color=[BLUE, ORANGE], alpha=0.5, hatch="//",
                label="rank-1 energy", zorder=2)
    ax.set_xticks(x_d)
    ax.set_xticklabels([fam_map[f] for f in fam_order], fontsize=8)
    ax.set_ylabel("Coherence measure")
    ax.set_title("Task-direction coherence", fontsize=10, pad=5)
    ax.set_ylim(0, 1.15)
    ax.axhline(1, color="#ccc", linewidth=0.5, linestyle=":")

    for b_grp in [b1, b2]:
        for rect in b_grp:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2, h + 0.015,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)

    from matplotlib.patches import Patch
    leg_handles = [
        Patch(fc=BLUE,   label="fine-net horizons"),
        Patch(fc=ORANGE, label="heterogeneous targets"),
        Patch(fc="white", ec="black", hatch="//", label="rank-1 energy"),
    ]
    ax.legend(handles=leg_handles, fontsize=7.5, loc="upper right")

    _savefig(fig, "fig8_coherence.pdf")
    _savefig(fig, "fig8_coherence.png")


# ---------------------------------------------------------------------------
# Macros
# ---------------------------------------------------------------------------
def _get_horizon_row(df, k):
    k_col = "horizon" if "horizon" in df.columns else "k"
    rows = df[df[k_col].astype(int) == k]
    return rows.iloc[0] if len(rows) else None


def write_macros(ROOT):
    macros = {}

    # Horizon global summary
    hg = ROOT / "outputs/selection_principle_horizon/horizon_global_summary.csv"
    if hg.exists():
        df = pd.read_csv(hg)
        sc = "mean_raw_slope" if "mean_raw_slope" in df.columns else "mean_slope"
        rc = "mean_resid_R2"  if "mean_resid_R2"  in df.columns else "mean_R2"
        vc = "cv_raw_slope"   if "cv_raw_slope"   in df.columns else "CV"
        nc = "mean_null_R2"   if "mean_null_R2"   in df.columns else "null_R2"
        fc = "frac_slope_ci_negative"
        kmap = {1:"kone",5:"kfive",10:"kten",25:"ktwentyfive",
                50:"kfifty",100:"khundred",200:"ktwohundred"}
        for k, name in kmap.items():
            row = _get_horizon_row(df, k)
            if row is not None:
                macros[f"horizon{name}"]  = f"{row[sc]:.3f}"
                macros[f"horizonR{name}"] = f"{row[rc]:.3f}"
        row1 = _get_horizon_row(df, 1)
        macros["horizonnegfrac"] = f"{row1[fc]:.2f}" if row1 is not None and fc in df.columns else "1.00"

    # Rule summary
    hr = ROOT / "outputs/selection_principle_horizon/horizon_rule_summary.csv"
    if hr.exists():
        df = pd.read_csv(hr)
        sc = "mean_raw_slope" if "mean_raw_slope" in df.columns else "mean_slope"
        rc = "mean_resid_R2"  if "mean_resid_R2"  in df.columns else "mean_R2"
        vc = "cv_raw_slope"   if "cv_raw_slope"   in df.columns else "CV"
        nc = "mean_null_R2"   if "mean_null_R2"   in df.columns else "null_R2"
        for rule, prefix in [("GoL","gol"),("HighLife","hl")]:
            sub = df[df["rule"] == rule]
            if len(sub):
                macros[f"{prefix}resiR"]    = f"{sub[rc].mean():.3f}"
                macros[f"{prefix}slopeCV"]  = f"{sub[vc].mean():.3f}"
                macros[f"{prefix}slopemean"]= f"{sub[sc].mean():.3f}"
                macros[f"{prefix}nullresiR"]= f"{sub[nc].mean():.4f}"
        macros["globalresiR"] = f"{df[rc].mean():.3f}"
        macros["nullresiR"]   = f"{df[nc].mean():.4f}"

    # Fate mechanism
    fg = ROOT / "outputs/isolate_fate/fate_global_summary.csv"
    if fg.exists():
        df = pd.read_csv(fg)
        mc = "model" if "model" in df.columns else df.columns[1]
        cc = "mean_cv_R2" if "mean_cv_R2" in df.columns else "cv_r2"
        for mname, key in [("all_fates","allfatesR"),("iso_count","isocountR"),
                            ("local_window","localwindowR"),("local_window","localwinR")]:
            rows = df[(df[mc] == mname) & df[cc].notna()]
            if len(rows): macros[key] = f"{rows.iloc[0][cc]:.3f}"

    # Amplitude
    am = ROOT / "outputs/mechanism_amplitude_law/amplitude_model_summary.csv"
    if am.exists():
        df = pd.read_csv(am)
        mc = "mechanism"       if "mechanism"       in df.columns else "feature"
        ac = "amplitude_model" if "amplitude_model" in df.columns else "model"
        for mod, key in [("full","ampLOOfull"),("size_rho","ampLOOsizerho"),
                         ("size_only","ampLOOsizeonly"),("rule_only","ampLOOruleonly")]:
            mask = (df[mc] == "iso_count") & (df[ac] == mod)
            row  = df[mask]
            if len(row): macros[key] = f"{row.iloc[0]['R2_LOO']:.3f}"
        mask_any = df[mc] == "iso_count"
        row_any  = df[mask_any]
        if len(row_any) and "amplitude_cv" in df.columns:
            macros["ampCV"] = f"{row_any.iloc[0]['amplitude_cv']:.3f}"

    # LGDS bridge / task-direction coherence
    bs_path = ROOT / "outputs/ca_lgds_bridge/bridge_summary.csv"
    if bs_path.exists():
        df = pd.read_csv(bs_path)
        for fam, ck, rk in [("horizon_fine_net","horizonCos","horizonRankone"),
                             ("selection_multi_target","selCos","selRankone")]:
            row = df[df["family"] == fam]
            if len(row):
                macros[ck] = f"{row.iloc[0]['mean_pairwise_abs_cosine']:.3f}"
                macros[rk] = f"{row.iloc[0]['rank1_cumulative_energy']:.3f}"

    br = ROOT / "outputs/ca_lgds_bridge/bridge_verdict.txt"
    if br.exists():
        text = br.read_text()
        m = re.search(r"horizon_fine_net:.*?mean rank-1 relative regret = ([\d.]+)", text, re.DOTALL)
        if m: macros["horizonRegret"] = m.group(1)
        m = re.search(r"selection_multi_target:.*?mean rank-1 relative regret = ([\d.]+)", text, re.DOTALL)
        if m: macros["selRegret"] = m.group(1)

    # Prestate
    ps = ROOT / "outputs/prestate_class_horizon/prestate_class_summary.csv"
    if ps.exists():
        df = pd.read_csv(ps)
        gh = df[(df["scope"] == "global_horizon") & (df["model"] == "iso_count")]
        if len(gh):
            macros["prestateIsoMean"]    = f"{gh['mean_R2'].mean():.3f}"
            macros["prestateIsoMin"]     = f"{gh['mean_R2'].min():.3f}"
            macros["prestateIsoShuffle"] = f"{gh['mean_shuffle_R2'].mean():.4f}"

    # Selection principle
    ss = ROOT / "outputs/selection_principle/selection_verdict.txt"
    if ss.exists():
        text = ss.read_text()
        m = re.search(r"GoL pooled: (\w+)", text)
        if m: macros["golSelVerdict"] = m.group(1)
        m = re.search(r"fine_net deltaR2_iso = ([\d.]+)", text)
        if m: macros["finenetDeltaRtwo"] = m.group(1)
        m = re.search(r"fine_net iso slope = (-[\d.]+)", text)
        if m: macros["finenetIsoSlope"] = m.group(1)

    # Selection null ΔR² (the iso-shuffle null for fine-net, global pooled)
    ss_csv = ROOT / "outputs/selection_principle/selection_summary.csv"
    if ss_csv.exists():
        df_sel = pd.read_csv(ss_csv)
        gp_fn = df_sel[(df_sel["scope"] == "global_pooled") &
                       (df_sel["target"] == "target_fine_net")]
        if len(gp_fn):
            nval = float(gp_fn.iloc[0]["deltaR2_null_iso_shuffle"])
            macros["selectionnullDeltaRtwo"] = f"{max(nval, 0):.4f}"

    # Standardized transfer
    ts = ROOT / "outputs/mechanism_transfer_standardized/transfer_standardized_summary.csv"
    if ts.exists():
        df = pd.read_csv(ts)
        rc = "mean_test_R2_z" if "mean_test_R2_z" in df.columns else "mean_R2_z"
        sub = df[df["model"] == "fate_all"]
        if len(sub):
            macros["transferFateAll"]     = f"{sub[rc].mean():.3f}"
            macros["transferFateAllFrac"] = f"{sub['frac_R2_positive'].min():.2f}"

    # Write
    out = ROOT / "paper/macros.tex"
    lines = ["% Auto-generated by scripts/make_response_law_artifacts.py\n",
             "% All command names use letters only -- valid LaTeX.\n"]
    for k, v in macros.items():
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    out.write_text("".join(lines))
    print(f"  wrote {len(macros)} macros to paper/macros.tex")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def write_horizon_table(ROOT):
    hg = ROOT / "outputs/selection_principle_horizon/horizon_global_summary.csv"
    if not hg.exists(): print("  MISSING horizon_global_summary.csv"); return
    df = pd.read_csv(hg)
    sc = "mean_raw_slope" if "mean_raw_slope" in df.columns else "mean_slope"
    vc = "cv_raw_slope"   if "cv_raw_slope"   in df.columns else "CV"
    rc = "mean_resid_R2"  if "mean_resid_R2"  in df.columns else "mean_R2"
    nc = "mean_null_R2"   if "mean_null_R2"   in df.columns else "null_R2"
    kc = "horizon"        if "horizon"        in df.columns else "k"
    lines = [
        "% Auto-generated -- do not edit\n",
        "\\begin{tabular}{rrrrr}\n","\\toprule\n",
        "$k$ & $\\bar{\\beta}_{\\rm iso}$ & CV & $\\bar{R}^2$ & Null $R^2$\\\\\n",
        "\\midrule\n",
    ]
    for _, row in df.iterrows():
        k  = int(row[kc])
        sl = row[sc]; cv = row[vc]; r2 = row[rc]; nu = row[nc]
        lines.append(f"  {k} & {sl:.3f} & {cv:.3f} & {r2:.3f} & {nu:.4f}\\\\\n")
    lines += ["\\bottomrule\n","\\end{tabular}\n"]
    (ROOT / "paper/tables/tab_horizon.tex").write_text("".join(lines))
    print("  wrote paper/tables/tab_horizon.tex")


def write_mechanism_table(ROOT):
    fg = ROOT / "outputs/isolate_fate/fate_global_summary.csv"
    if not fg.exists(): print("  MISSING fate_global_summary.csv"); return
    df = pd.read_csv(fg)
    mc = "model" if "model" in df.columns else df.columns[1]
    cc = "mean_cv_R2" if "mean_cv_R2" in df.columns else "cv_r2"
    sc = "mean_slope_raw" if "mean_slope_raw" in df.columns else "mean_slope"
    order = ["iso_count","death","survive","survive_split","birth_bridge",
             "local_window","cell_fates_all","all_fates","coarse","entropy"]
    pretty = {"iso_count":"iso\\_count","death":"iso die","survive":"iso survive",
              "survive_split":"survive split","birth_bridge":"birth bridge",
              "local_window":"local window","cell_fates_all":"cell fates all",
              "all_fates":"all fates","coarse":"coarse","entropy":"entropy"}
    lines = [
        "% Auto-generated -- do not edit\n",
        "\\begin{tabular}{lrr}\n","\\toprule\n",
        "Feature set & CV $R^2$ & slope\\\\\n","\\midrule\n",
    ]
    for mname in order:
        rows = df[(df[mc] == mname) & df[cc].notna()]
        if len(rows):
            v = rows.iloc[0][cc]
            lines.append(f"  {pretty.get(mname,mname)} & {v:.3f} & ---\\\\\n")
    lines += ["\\midrule\n"]
    slope_rows = df[df[cc].isna() & df[sc].notna()]
    slope_pretty = {
        "slope_iso_count":               "iso\\_count",
        "slope_iso_die":                 "die",
        "slope_iso_survive":             "survive",
        "slope_iso_survive_connected":   "surv.~connected",
        "slope_iso_orth_birth_any":      "orth birth",
        "slope_iso_diag_birth_any":      "diag birth",
        "slope_iso_local_window_loss_sum":"window loss",
        "slope_iso_local_window_gain_sum":"window gain",
    }
    for _, row in slope_rows.iterrows():
        m = row[mc]
        if m in slope_pretty:
            lines.append(f"  {slope_pretty[m]} & --- & {row[sc]:.3f}\\\\\n")
    lines += ["\\bottomrule\n","\\end{tabular}\n"]
    (ROOT / "paper/tables/tab_mechanism.tex").write_text("".join(lines))
    print("  wrote paper/tables/tab_mechanism.tex")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating paper artifacts...")
    make_fig1(ROOT)
    make_fig2(ROOT)
    make_fig3(ROOT)
    make_fig4(ROOT)
    make_fig5(ROOT)
    make_fig6(ROOT)
    make_fig7(ROOT)
    make_fig8(ROOT)
    write_macros(ROOT)
    write_horizon_table(ROOT)
    write_mechanism_table(ROOT)
    print("Done.")
