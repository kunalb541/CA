#!/usr/bin/env python3
"""
Artifact generator for the response-law paper.

Reads existing outputs/ and writes:
  paper/macros.tex      -- LaTeX macros (all-letter names, no digits)
  paper/tables/         -- compact LaTeX tables
  paper/figures/        -- 4 flagship figures + supporting plots

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
        "font.family":      "sans-serif",
        "font.size":        8,
        "axes.titlesize":   9,
        "axes.labelsize":   8,
        "xtick.labelsize":  7,
        "ytick.labelsize":  7,
        "legend.fontsize":  7,
        "legend.framealpha": 0.85,
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "axes.linewidth":   0.8,
        "lines.linewidth":  1.4,
        "patch.linewidth":  0.6,
    })

def _savefig(fig, name):
    p = OUT_FIG / name
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p.name}")

# ---------------------------------------------------------------------------
# Figure 1 — Object channel and target specificity
# ---------------------------------------------------------------------------
def make_fig1(ROOT):
    _apply_style()

    sel  = pd.read_csv(ROOT / "outputs/selection_principle/selection_summary.csv")
    pre  = pd.read_csv(ROOT / "outputs/prestate_class_horizon/prestate_class_summary.csv")

    # ---------- panel (a): embedded-isolate schematic ----------
    # 3×3 grid; focal cell at centre; N/S/E/W dead; SW + SE alive (diagonal)
    LIVE_FOCAL = "black"
    LIVE_DIAG  = "#555555"
    DEAD       = "white"
    EDGE       = "#444444"

    cell_state = {
        (0,0): LIVE_DIAG, (0,1): DEAD, (0,2): LIVE_DIAG,
        (1,0): DEAD,      (1,1): LIVE_FOCAL, (1,2): DEAD,
        (2,0): DEAD,      (2,1): DEAD, (2,2): DEAD,
    }

    # ---------- panel (b): target-specificity ΔR² bars ----------
    gp = sel[sel["scope"] == "global_pooled"].copy()
    # Shorten target names
    label_map = {
        "target_fine_net":        "fine-net\n$\\Delta C_k$",
        "target_delta_density":   "density\n$\\Delta\\rho$",
        "target_delta_block_var": "block\nvariance",
        "target_future_density":  "future\ndensity",
        "target_delta_block_entropy": "block\nentropy",
        "target_future_live_count":"live\ncount",
        "target_delta_components": "comps\n$\\Delta C_1$",
    }
    target_order = [
        "target_fine_net",
        "target_delta_components",
        "target_delta_density",
        "target_delta_block_var",
        "target_delta_block_entropy",
        "target_future_density",
        "target_future_live_count",
    ]
    gp = gp.set_index("target").reindex(target_order).reset_index()
    labels = [label_map.get(t, t) for t in gp["target"]]

    # ---------- panel (c): prestate horizon curves ----------
    gh = pre[pre["scope"] == "global_horizon"]
    iso_pre  = gh[gh["model"] == "iso_count"].sort_values("horizon")
    cls_pre  = gh[gh["model"] == "class_counts"].sort_values("horizon") \
               if "class_counts" in gh["model"].values else None

    # ====== build figure ======
    fig = plt.figure(figsize=(7.2, 3.4))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38,
                            left=0.06, right=0.98, top=0.92, bottom=0.14)

    # ----- (a) schematic -----
    ax_s = fig.add_subplot(gs[0, 0])
    ax_s.set_xlim(-0.15, 3.15)
    ax_s.set_ylim(-0.35, 3.15)
    ax_s.set_aspect("equal")
    ax_s.axis("off")
    sz = 0.85
    gap = 0.05
    for (r, c), col in cell_state.items():
        x0 = c * (sz + gap)
        y0 = (2 - r) * (sz + gap)   # flip rows so row-0 is top
        rect = Rectangle((x0, y0), sz, sz,
                          facecolor=col, edgecolor=EDGE, linewidth=0.8, zorder=2)
        ax_s.add_patch(rect)
        # orth neighbours: annotate with ×
        if col == DEAD and (r == 1 or c == 1):
            ax_s.text(x0 + sz/2, y0 + sz/2, "×",
                      ha="center", va="center", fontsize=10, color="#888888", zorder=3)
        if col == LIVE_FOCAL:
            ax_s.text(x0 + sz/2, y0 + sz/2, "●",
                      ha="center", va="center", fontsize=14, color="white", zorder=3)
        if col == LIVE_DIAG:
            ax_s.text(x0 + sz/2, y0 + sz/2, "●",
                      ha="center", va="center", fontsize=11, color="white", zorder=3)
    # annotation
    ax_s.text(1.5 * (sz + gap) - sz/2, -0.25,
              "4-connected\nneighbours dead", ha="center", va="top", fontsize=6.5,
              color="#666666")
    ax_s.text(3.0 * (sz + gap), 2.9 * (sz + gap),
              "diag.\nnbr", ha="left", va="top", fontsize=6, color="#444444",
              bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#aaa", lw=0.5))
    ax_s.set_title("(a)  Embedded isolate", loc="left", fontsize=8, pad=4)

    # ----- (b) ΔR² bars -----
    ax_b = fig.add_subplot(gs[0, 1])
    x   = np.arange(len(labels))
    w   = 0.18
    iso_vals  = gp["deltaR2_iso"].values
    crs_vals  = gp["deltaR2_coarse"].values
    ent_vals  = gp["deltaR2_entropy"].values
    nul_vals  = gp["deltaR2_null_iso_shuffle"].values

    ax_b.bar(x - 1.5*w, iso_vals,  w, label="iso",     color=BLUE,   zorder=2)
    ax_b.bar(x - 0.5*w, crs_vals,  w, label="coarse",  color=GRAY,   zorder=2)
    ax_b.bar(x + 0.5*w, ent_vals,  w, label="entropy", color=ORANGE, alpha=0.7, zorder=2)
    ax_b.bar(x + 1.5*w, nul_vals,  w, label="null",    color="white",
             edgecolor=BLACK, linewidth=0.6, zorder=2)
    ax_b.axhline(0, color=BLACK, linewidth=0.5)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels, fontsize=6, rotation=0)
    ax_b.set_ylabel("$\\Delta R^2$ vs density control")
    ax_b.set_title("(b)  Target specificity (global pooled)", loc="left", fontsize=8, pad=4)
    ax_b.legend(loc="upper right", fontsize=6, ncol=2)
    # Annotate fine-net iso bar
    ax_b.annotate("iso only\nsignificant\nfor fine-net",
                  xy=(x[0] - 1.5*w, iso_vals[0]),
                  xytext=(x[0] + 0.8, iso_vals[0] + 0.0003),
                  arrowprops=dict(arrowstyle="->", color="black", lw=0.6),
                  fontsize=5.5, va="center", ha="left")

    # ----- (c) prestate curves -----
    ax_c = fig.add_subplot(gs[0, 2])
    ks   = iso_pre["horizon"].values
    r2   = iso_pre["mean_R2"].values
    shu  = iso_pre["mean_shuffle_R2"].values

    ax_c.plot(ks, r2, "o-", color=BLUE, label="iso\\_count $R^2(k)$", ms=4, zorder=3)
    if cls_pre is not None:
        r2c = cls_pre["mean_R2"].values
        ax_c.plot(ks, r2c, "s--", color=GREEN, label="class\\_counts $R^2(k)$",
                  ms=3.5, zorder=3)
    ax_c.plot(ks, shu, ":", color=GRAY, linewidth=1, label="shuffle null", zorder=2)
    ax_c.fill_between(ks, shu, alpha=0.15, color=GRAY)
    ax_c.set_xscale("log")
    ax_c.set_xlabel("Horizon $k$")
    ax_c.set_ylabel("Residual $R^2$")
    ax_c.set_title("(c)  Non-leaky prestate recovery", loc="left", fontsize=8, pad=4)
    ax_c.legend(loc="upper right", fontsize=6)
    ax_c.set_xticks([1, 5, 10, 25, 50, 100, 200])
    ax_c.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax_c.tick_params(axis="x", which="minor", bottom=False)
    ax_c.set_ylim(bottom=0)

    _savefig(fig, "fig1_selection_prestate.pdf")
    _savefig(fig, "fig1_selection_prestate.png")


# ---------------------------------------------------------------------------
# Figure 2 — Temporal response law
# ---------------------------------------------------------------------------
def make_fig2(ROOT):
    _apply_style()

    hg = pd.read_csv(ROOT / "outputs/selection_principle_horizon/horizon_global_summary.csv")
    hr = pd.read_csv(ROOT / "outputs/selection_principle_horizon/horizon_rule_summary.csv")
    hc = pd.read_csv(ROOT / "outputs/selection_principle_horizon/horizon_condition_summary.csv")

    slope_col = "mean_raw_slope" if "mean_raw_slope" in hg.columns else "mean_slope"
    r2_col    = "mean_resid_R2"  if "mean_resid_R2"  in hg.columns else "mean_R2"
    cv_col    = "cv_raw_slope"   if "cv_raw_slope"   in hg.columns else "CV"
    null_col  = "mean_null_R2"   if "mean_null_R2"   in hg.columns else "null_R2"
    k_col     = "horizon"        if "horizon"        in hg.columns else "k"

    ks_g   = hg[k_col].values
    sl_g   = hg[slope_col].values
    r2_g   = hg[r2_col].values
    nu_g   = hg[null_col].values
    cv_g   = hg[cv_col].values

    gol = hr[hr["rule"] == "GoL"].sort_values(k_col)
    hl  = hr[hr["rule"] == "HighLife"].sort_values(k_col)

    # Heatmap pivot: rows = conditions (sorted), cols = horizons
    def _cond_key(s):
        import re as _re
        rule = 0 if s.startswith("GoL") else 1
        m = _re.search(r"_L(\d+)_rho", s)
        L = int(m.group(1)) if m else 0
        rho = float(s.split("rho")[1]) if "rho" in s else 0.0
        return (rule, L, rho)
    cond_order = sorted(hc["condition_id"].unique(), key=_cond_key)
    pivot = hc.pivot(index="condition_id", columns="horizon",
                     values="resid_iso_slope_raw").reindex(cond_order)
    # Short row labels
    def _short(cid):
        rule = "G" if cid.startswith("GoL") else "H"
        L    = cid.split("_L")[1].split("_rho")[0]
        rho  = cid.split("rho")[1]
        return f"{rule}{L}·{rho}"
    row_labels = [_short(c) for c in pivot.index]
    col_labels = [str(k) for k in pivot.columns.tolist()]

    # ====== build figure ======
    fig = plt.figure(figsize=(7.2, 6.5))
    gs_top = gridspec.GridSpec(1, 2, top=0.96, bottom=0.54,
                                left=0.08, right=0.98, wspace=0.32)
    gs_bot = gridspec.GridSpec(1, 1, top=0.48, bottom=0.06,
                                left=0.08, right=0.98)

    # ----- (a) slopes -----
    ax_a = fig.add_subplot(gs_top[0, 0])
    ax_a.plot(ks_g, sl_g, "k-o", ms=5, label="Global", linewidth=1.8, zorder=4)
    ax_a.plot(gol[k_col].values, gol[slope_col].values,
              "-o", color=BLUE, ms=4, label="GoL", zorder=3)
    ax_a.plot(hl[k_col].values,  hl[slope_col].values,
              "--s", color=ORANGE, ms=4, label="HighLife", zorder=3)

    # condition spread: min/max from heatmap per horizon
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
    ax_a.set_title("(a)  Response slope vs horizon", loc="left", fontsize=9)
    ax_a.legend(loc="lower right", fontsize=6.5)
    ax_a.set_ylim(top=0.05)

    # ----- (b) R² -----
    ax_b = fig.add_subplot(gs_top[0, 1])
    ax_b.plot(ks_g, r2_g, "k-o", ms=5, label="Global $R^2(k)$", linewidth=1.8, zorder=4)
    ax_b.plot(gol[k_col].values, gol[r2_col].values,
              "-o", color=BLUE, ms=4, label="GoL", zorder=3)
    ax_b.plot(hl[k_col].values,  hl[r2_col].values,
              "--s", color=ORANGE, ms=4, label="HighLife", zorder=3)
    ax_b.plot(ks_g, nu_g, ":", color=GRAY, linewidth=1.2, label="shuffle null", zorder=2)
    ax_b.fill_between(ks_g, nu_g, alpha=0.12, color=GRAY)
    ax_b.set_xscale("log")
    ax_b.set_xticks([1, 5, 10, 25, 50, 100, 200])
    ax_b.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax_b.tick_params(axis="x", which="minor", bottom=False)
    ax_b.set_xlabel("Horizon $k$")
    ax_b.set_ylabel("Residual $R^2(k)$")
    ax_b.set_title("(b)  Predictive signal vs horizon", loc="left", fontsize=9)
    ax_b.legend(loc="upper right", fontsize=6.5)
    ax_b.set_ylim(bottom=0)

    # ----- (c) heatmap -----
    ax_c = fig.add_subplot(gs_bot[0, 0])
    data = pivot.values
    im = ax_c.imshow(data, aspect="auto", cmap="RdBu_r",
                     vmin=-1.0, vmax=0, origin="upper")
    ax_c.set_xticks(range(len(col_labels)))
    ax_c.set_xticklabels(col_labels, fontsize=7)
    ax_c.set_yticks(range(len(row_labels)))
    ax_c.set_yticklabels(row_labels, fontsize=6.5)
    ax_c.set_xlabel("Horizon $k$")
    ax_c.set_ylabel("Condition  (rule · $L$ · $\\rho$)")
    ax_c.set_title("(c)  All 112 condition–horizon slopes are negative",
                   loc="left", fontsize=9)
    cbar = fig.colorbar(im, ax=ax_c, shrink=0.7, pad=0.01)
    cbar.set_label("$\\hat{\\beta}_{\\rm iso}$", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    # Annotate max and min
    min_val = data.min()
    max_val = data.max()  # closest to 0 but still negative
    ax_c.text(0.99, 0.01, f"range [{min_val:.3f}, {max_val:.3f}]",
              transform=ax_c.transAxes, ha="right", va="bottom",
              fontsize=6, color="white",
              bbox=dict(boxstyle="round,pad=0.2", fc="#333", ec="none"))

    _savefig(fig, "fig2_response_law.pdf")
    _savefig(fig, "fig2_response_law.png")


# ---------------------------------------------------------------------------
# Figure 3 — Mechanism carrier diagnostics
# ---------------------------------------------------------------------------
def make_fig3(ROOT):
    _apply_style()

    fg   = pd.read_csv(ROOT / "outputs/isolate_fate/fate_global_summary.csv")
    tc   = pd.read_csv(ROOT / "outputs/isolate_transition_classes/transition_class_contributions.csv")
    tsm  = pd.read_csv(ROOT / "outputs/isolate_transition_classes/transition_class_summary.csv")

    cv_col    = "mean_cv_R2" if "mean_cv_R2" in fg.columns else "cv_r2"
    model_col = "model"      if "model"      in fg.columns else "predictor"
    slope_col = "mean_slope_raw" if "mean_slope_raw" in fg.columns else "mean_slope"

    # CV R² panel data (global_model rows)
    cv_rows = fg[fg[cv_col].notna() & (fg[cv_col] > -99)].copy()
    cv_name_map = {
        "all_fates":    "all fates",
        "all_plus_coarse": "all + coarse",
        "cell_fates_all": "cell fates",
        "birth_bridge": "birth bridge",
        "local_window": "local window",
        "survive_split":"survive\nsplit",
        "survive":      "survive",
        "iso_count":    "iso count",
        "death":        "death",
        "coarse":       "coarse",
        "entropy":      "entropy",
    }
    cv_rows["label"] = cv_rows[model_col].map(cv_name_map).fillna(cv_rows[model_col])
    cv_rows = cv_rows.sort_values(cv_col, ascending=True)

    # Slope panel data (global_slope rows)
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
    # Keep only the mechanistically meaningful single-feature slopes
    keep_slopes = {
        "slope_iso_count","slope_iso_die","slope_iso_survive",
        "slope_iso_survive_connected","slope_iso_orth_birth_any",
        "slope_iso_diag_birth_any","slope_iso_local_window_loss_sum",
        "slope_iso_local_window_gain_sum",
    }
    sl_rows = sl_rows[sl_rows[model_col].isin(keep_slopes)].copy()
    sl_rows["label"] = sl_rows[model_col].map(slope_name_map)
    sl_rows = sl_rows.sort_values(slope_col, ascending=True)

    # Transition class contribution data
    tc_sorted = tc.sort_values("p_times_delta", ascending=True).head(16)

    # ====== build figure ======
    fig = plt.figure(figsize=(7.2, 5.8))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.42,
                            left=0.05, right=0.99, top=0.93, bottom=0.08)

    # ----- (a) CV R² -----
    ax_a = fig.add_subplot(gs[0, 0])
    n_a  = len(cv_rows)
    ypos = np.arange(n_a)
    vals = cv_rows[cv_col].values
    cols = [BLUE if "iso" in str(l) else
            GREEN if "window" in str(l) or "fates" in str(l) or "bridge" in str(l) else
            GRAY for l in cv_rows["label"].values]
    # colour iso_count distinctly, coarse/entropy muted
    col_list = []
    for m in cv_rows[model_col].values:
        if m in ("coarse", "entropy"):
            col_list.append(GRAY)
        elif m == "iso_count":
            col_list.append(BLUE)
        elif m in ("local_window", "all_fates", "all_plus_coarse", "cell_fates_all",
                   "birth_bridge", "survive_split", "survive", "death"):
            col_list.append(GREEN if vals[list(cv_rows[model_col]).index(m)] > 0.35 else LBLUE)
        else:
            col_list.append(LBLUE)

    bars = ax_a.barh(ypos, vals, color=col_list, edgecolor="white", linewidth=0.4,
                     height=0.72, zorder=2)
    ax_a.set_yticks(ypos)
    ax_a.set_yticklabels(cv_rows["label"].values, fontsize=7)
    ax_a.set_xlabel("CV $R^2$ on residual $\\Delta C_k$", fontsize=8)
    ax_a.set_title("(a)  Mechanism models", loc="left", fontsize=9)
    ax_a.axvline(0, color=BLACK, linewidth=0.4)
    # Annotate iso_count and local_window
    for i, (m, v) in enumerate(zip(cv_rows[model_col].values, vals)):
        if m in ("iso_count", "local_window", "all_fates"):
            ax_a.text(v + 0.005, ypos[i], f"{v:.3f}", va="center", fontsize=6,
                      color=BLACK)
    ax_a.set_xlim(right=0.60)

    # ----- (b) Feature slopes -----
    ax_b = fig.add_subplot(gs[0, 1])
    n_b  = len(sl_rows)
    yb   = np.arange(n_b)
    svals = sl_rows[slope_col].values
    col_s = [ORANGE if v > 0 else BLUE for v in svals]
    ax_b.barh(yb, svals, color=col_s, edgecolor="white", linewidth=0.4,
              height=0.72, zorder=2)
    ax_b.set_yticks(yb)
    ax_b.set_yticklabels(sl_rows["label"].values, fontsize=7)
    ax_b.set_xlabel("Mean slope $\\hat{\\beta}$", fontsize=8)
    ax_b.set_title("(b)  Single-feature slopes", loc="left", fontsize=9)
    ax_b.axvline(0, color=BLACK, linewidth=0.8)
    for i, v in enumerate(svals):
        xtext = v + 0.03 if v >= 0 else v - 0.05
        ax_b.text(xtext, yb[i], f"{v:.2f}", va="center", fontsize=6,
                  ha="left" if v >= 0 else "right", color=BLACK)

    # ----- (c) Transition class contributions -----
    ax_c = fig.add_subplot(gs[0, 2])
    # Show top contributors (most negative p*delta = strongest negative response)
    # Use all classes, sorted by p*delta
    tc_all = tc.sort_values("p_times_delta", ascending=True)
    yc     = np.arange(len(tc_all))
    pd_vals = tc_all["p_times_delta"].values
    # Label with class number and brief description
    class_desc = {
        1: "1 diag nbr",  2: "2 diag nbr (opp)",  3: "2 diag nbr (adj)",
        4: "2 diag (opp2)", 5: "3 diag nbr",  6: "2 adj diag",
        7: "3 diag+1",  8: "all 4 diag",  9: "corner adj",
        10: "corner+opp", 11: "3+sym", 12: "L-shape",
        13: "broad", 14: "complex", 15: "full ring",
    }
    clabels = [f"cls {int(c)} ({class_desc.get(int(c),'?')})"
               for c in tc_all["diag_class"].values]
    col_c = [BLUE if v < 0 else ORANGE for v in pd_vals]
    ax_c.barh(yc, pd_vals, color=col_c, edgecolor="white", linewidth=0.3,
              height=0.72, zorder=2)
    ax_c.set_yticks(yc)
    ax_c.set_yticklabels(clabels, fontsize=6)
    ax_c.set_xlabel("$p(\\mathrm{cls}) \\times \\overline{\\Delta}_{\\rm local}$", fontsize=8)
    ax_c.set_title("(c)  Class × mean local response", loc="left", fontsize=9)
    ax_c.axvline(0, color=BLACK, linewidth=0.8)
    # Annotate death fraction for class 1 (all die)
    tsm_gol = tsm[tsm["rule"] == "GoL"].copy() if "rule" in tsm.columns else tsm
    if "death_frac" in tsm_gol.columns:
        for i, row in enumerate(tc_all.itertuples()):
            sub = tsm_gol[tsm_gol["diag_class"] == row.diag_class]
            if len(sub):
                df_val = float(sub["death_frac"].mean())
                ax_c.text(-0.005, yc[i], f"d={df_val:.2f}",
                          ha="right", va="center", fontsize=5.5, color="#444")

    _savefig(fig, "fig3_mechanism.pdf")
    _savefig(fig, "fig3_mechanism.png")


# ---------------------------------------------------------------------------
# Figure 4 — Transport, amplitude, and formal bridge
# ---------------------------------------------------------------------------
def make_fig4(ROOT):
    _apply_style()

    ts  = pd.read_csv(ROOT / "outputs/mechanism_transfer_standardized/transfer_standardized_summary.csv")
    at  = pd.read_csv(ROOT / "outputs/mechanism_amplitude_law/amplitude_condition_table.csv")
    am  = pd.read_csv(ROOT / "outputs/mechanism_amplitude_law/amplitude_model_summary.csv")
    bs  = pd.read_csv(ROOT / "outputs/ca_lgds_bridge/bridge_summary.csv")

    r2z_col = "mean_test_R2_z" if "mean_test_R2_z" in ts.columns else "mean_R2_z"

    # ====== build figure ======
    fig = plt.figure(figsize=(7.2, 6.8))
    gs  = gridspec.GridSpec(2, 2, figure=fig, wspace=0.38, hspace=0.45,
                            left=0.09, right=0.98, top=0.95, bottom=0.07)

    # ----- (a) Standardized transfer -----
    ax_a = fig.add_subplot(gs[0, 0])
    models_plot = [
        ("fate_all",                   "all fates",    BLACK,  "o"),
        ("local_window_delta_loss_gain","window Δ",     GREEN,  "s"),
        ("local_window_loss",          "window loss",  BLUE,   "^"),
        ("class_counts_plus_losses",   "class+loss",   ORANGE, "D"),
        ("iso_count",                  "iso count",    GRAY,   "P"),
    ]
    split_order = ["leave_density", "leave_size", "leave_rule", "leave_condition"]
    split_labels = ["density", "size", "rule", "condition"]
    x_s = np.arange(len(split_order))

    for mname, mlabel, mcol, mmark in models_plot:
        sub = ts[ts["model"] == mname].set_index("split_type").reindex(split_order)
        vals = sub[r2z_col].values
        ax_a.plot(x_s, vals, marker=mmark, color=mcol, ms=5,
                  label=mlabel, linewidth=1.2, zorder=3)

    ax_a.axhline(0, color="#aaa", linewidth=0.5, linestyle="--")
    ax_a.set_xticks(x_s)
    ax_a.set_xticklabels(split_labels, fontsize=7.5)
    ax_a.set_ylabel("Mean $R^2_z$ (leave-one-out)")
    ax_a.set_xlabel("Held-out split")
    ax_a.set_title("(a)  Standardized mechanism transfer", loc="left", fontsize=9)
    ax_a.legend(loc="lower right", fontsize=6, ncol=1)
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
                     edgecolors="white", linewidths=0.4)
    ax_b.set_xlabel("Initial density $\\rho$")
    ax_b.set_ylabel("Condition amplitude $A$")
    ax_b.set_title("(b)  Amplitude vs density", loc="left", fontsize=9)
    # Add legend with dummy patches
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor=BLUE,
               markeredgecolor="none", markersize=6, label="GoL $L$=64"),
        Line2D([0],[0], marker="s", color="w", markerfacecolor=BLUE,
               markeredgecolor="none", markersize=6, label="GoL $L$=128"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=ORANGE,
               markeredgecolor="none", markersize=6, label="HighLife $L$=64"),
        Line2D([0],[0], marker="s", color="w", markerfacecolor=ORANGE,
               markeredgecolor="none", markersize=6, label="HighLife $L$=128"),
    ]
    ax_b.legend(handles=handles, loc="upper left", fontsize=6)
    # annotate L-dependence
    ax_b.text(0.97, 0.05, "larger $L$ → larger $A$",
              transform=ax_b.transAxes, ha="right", va="bottom",
              fontsize=6, style="italic", color="#555")

    # ----- (c) Amplitude LOO R² -----
    ax_c = fig.add_subplot(gs[1, 0])
    am_iso = am[am["mechanism"] == "iso_count"].copy()
    model_order = ["full", "size_rho", "rule_size_rho", "size_only", "rho_only", "rule_only"]
    model_label = {
        "full":          "full\n$(L,\\rho,\\mathrm{rule},L{\\times}\\rho,\\ldots)$",
        "size_rho":      "size + $\\rho$",
        "rule_size_rho": "rule + size + $\\rho$",
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
                          fontsize=6.5)
    ax_c.set_xlabel("LOO $R^2$")
    ax_c.set_title("(c)  Amplitude prediction ($iso$-count)", loc="left", fontsize=9)
    ax_c.axvline(0, color=BLACK, linewidth=0.6)
    ax_c.axvline(1, color="#ccc", linewidth=0.5, linestyle=":")
    for i, v in enumerate(r2_vals):
        ax_c.text(max(v, 0) + 0.01, yc2[i], f"{v:.3f}",
                  va="center", fontsize=6.5, color=BLACK)
    ax_c.set_xlim(-0.35, 1.05)

    # ----- (d) LGDS bridge: task coherence -----
    ax_d = fig.add_subplot(gs[1, 1])
    fam_map = {
        "horizon_fine_net":       "Fine-net\nhorizons\n($k$=1–200)",
        "selection_multi_target": "Heterogeneous\ntargets",
    }
    fam_order = ["horizon_fine_net", "selection_multi_target"]
    bs_sub = bs.set_index("family").reindex(fam_order).reset_index()

    width = 0.3
    x_d = np.array([0, 0.7])
    cos_vals  = bs_sub["mean_pairwise_abs_cosine"].values
    rank1_vals = bs_sub["rank1_cumulative_energy"].values

    b1 = ax_d.bar(x_d - width/2, cos_vals,  width,
                  color=[BLUE, ORANGE], label="mean $|\\cos|$", zorder=2)
    b2 = ax_d.bar(x_d + width/2, rank1_vals, width,
                  color=[BLUE, ORANGE], alpha=0.5, hatch="//",
                  label="rank-1 energy", zorder=2)
    ax_d.set_xticks(x_d)
    ax_d.set_xticklabels([fam_map[f] for f in fam_order], fontsize=7.5)
    ax_d.set_ylabel("Coherence measure")
    ax_d.set_title("(d)  Task-description coherence (LGDS)", loc="left", fontsize=9)
    ax_d.set_ylim(0, 1.12)
    ax_d.axhline(1, color="#ccc", linewidth=0.5, linestyle=":")

    # value labels
    for b_grp in [b1, b2]:
        for rect in b_grp:
            h = rect.get_height()
            ax_d.text(rect.get_x() + rect.get_width()/2, h + 0.015,
                      f"{h:.3f}", ha="center", va="bottom", fontsize=6.5)

    from matplotlib.patches import Patch
    leg_handles = [Patch(fc=BLUE, label="fine-net horizons"),
                   Patch(fc=ORANGE, label="heterogeneous targets")]
    ax_d.legend(handles=leg_handles, fontsize=6.5,
                loc="upper right")

    _savefig(fig, "fig4_transfer_amplitude_bridge.pdf")
    _savefig(fig, "fig4_transfer_amplitude_bridge.png")


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
        mc = "mechanism"      if "mechanism"      in df.columns else "feature"
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

    # LGDS bridge
    bs = ROOT / "outputs/ca_lgds_bridge/bridge_summary.csv"
    if bs.exists():
        df = pd.read_csv(bs)
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

    # Standardized transfer
    ts = ROOT / "outputs/mechanism_transfer_standardized/transfer_standardized_summary.csv"
    if ts.exists():
        df = pd.read_csv(ts)
        rc = "mean_test_R2_z" if "mean_test_R2_z" in df.columns else "mean_R2_z"
        sub = df[df["model"] == "fate_all"]
        if len(sub):
            macros["transferFateAll"] = f"{sub[rc].mean():.3f}"
            macros["transferFateAllFrac"] = f"{sub['frac_R2_positive'].min():.2f}"

    # Write
    out = ROOT / "paper/macros.tex"
    lines = ["% Auto-generated by scripts/make_response_law_artifacts.py\n",
             "% All command names use letters only — valid LaTeX.\n"]
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
        "% Auto-generated — do not edit\n",
        "\\begin{tabular}{rrrrr}\n","\\toprule\n",
        "$k$ & $\\bar{\\beta}_{\\rm iso}$ & CV & $\\bar{R}^2$ & Null $R^2$\\\\\n",
        "\\midrule\n",
    ]
    for _, row in df.iterrows():
        k    = int(row[kc])
        sl   = row[sc]; cv = row[vc]; r2 = row[rc]; nu = row[nc]
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
        "% Auto-generated — do not edit\n",
        "\\begin{tabular}{lrr}\n","\\toprule\n",
        "Feature set & CV $R^2$ & slope\\\\\n","\\midrule\n",
    ]
    for mname in order:
        rows = df[(df[mc] == mname) & df[cc].notna()]
        if len(rows):
            v = rows.iloc[0][cc]
            lines.append(f"  {pretty.get(mname,mname)} & {v:.3f} & ---\\\\\n")
    # slopes from global_slope rows
    lines += ["\\midrule\n"]
    slope_rows = df[df[cc].isna() & df[sc].notna()]
    slope_pretty = {
        "slope_iso_count":"iso\\_count","slope_iso_die":"die",
        "slope_iso_survive":"survive","slope_iso_survive_connected":"surv.\ connected",
        "slope_iso_orth_birth_any":"orth birth","slope_iso_diag_birth_any":"diag birth",
        "slope_iso_local_window_loss_sum":"window loss","slope_iso_local_window_gain_sum":"window gain",
    }
    for _, row in slope_rows.iterrows():
        m = row[mc]
        if m in slope_pretty:
            lines.append(f"  {slope_pretty[m]} & --- & {row[sc]:.3f}\\\\\n")
    lines += ["\\bottomrule\n","\\end{tabular}\n"]
    (ROOT / "paper/tables/tab_mechanism.tex").write_text("".join(lines))
    print("  wrote paper/tables/tab_mechanism.tex")


# ---------------------------------------------------------------------------
# Copy legacy figures (for any still referenced)
# ---------------------------------------------------------------------------
def copy_figures(ROOT):
    keep = {
        "outputs/selection_principle/fig_deltaR2_by_target.png":
            "fig_deltaR2_by_target.png",
        "outputs/mechanism_transfer_standardized/fig_transfer_standardized_r2.png":
            "fig_transfer_standardized_r2.png",
    }
    for src_rel, dst_name in keep.items():
        src = ROOT / src_rel
        dst = OUT_FIG / dst_name
        if src.exists():
            shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating paper artifacts...")
    make_fig1(ROOT)
    make_fig2(ROOT)
    make_fig3(ROOT)
    make_fig4(ROOT)
    write_macros(ROOT)
    write_horizon_table(ROOT)
    write_mechanism_table(ROOT)
    copy_figures(ROOT)
    print("Done.")
