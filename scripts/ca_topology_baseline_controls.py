#!/usr/bin/env python3
"""
Topology baseline controls: test whether embedded-isolate count is
genuinely special among simple local topology statistics.

Question:
Does iso_embedded outperform simple alternative topology predictors for
residual future fine-component loss (ΔC_k)?

Protocol:
  rules     = GoL, HighLife
  L         = 64, 128 (periodic torus)
  rho       = 0.20, 0.25, 0.30, 0.35
  n_samples = 1000 per condition (50 in --quick mode)
  horizons  = 1, 5, 10, 25, 50, 100, 200
  target    = fine_net = C(t+k) − C(t)  (4-connected components)
  baseline  = live_count + density  (same as main paper)

Predictors tested at t=0:
  iso_embedded        alive, orth_deg=0, diag_ge1  [main predictor]
  singleton_comp      size-1 4-connected components
  small_comp_2/3/4    components of exactly size 2/3/4
  small_comp_le4      components of size ≤ 4
  orth_deg0           alive with 0 orthogonal live nbrs
  orth_deg0_diag0     alive, 0 orth + 0 diag (truly isolated)
  orth_deg0_diag_ge1  alive, 0 orth + ≥1 diag (= iso_embedded)
  moore_deg0          alive with 0 Moore nbrs
  moore_deg1          alive with exactly 1 Moore nbr
  moore_deg2          alive with exactly 2 Moore nbrs
  boundary_cell       alive with ≥1 orthogonal dead nbr (orth_n < 4)
  component_count_t0  C(t=0)
  block_var           8×8 block density variance
  block_entropy       8×8 block density Shannon entropy

Analysis:
  Per (condition, horizon) cell: residual R² and ΔR² for each predictor.
  Global summary: mean/min/max residual R² and ΔR² per predictor.
  Incremental summary: iso_embedded advantage over each competitor.

Outputs: outputs/topology_baselines/
  topology_baseline_raw.csv
  topology_baseline_condition_summary.csv
  topology_baseline_global_summary.csv
  topology_baseline_incremental_summary.csv
  topology_baseline_verdict.txt
  fig_topology_baseline_r2.png
  fig_topology_incremental.png

Verdict:
  PASS      iso_embedded is best or within 5% of best single predictor,
            mean_resid_R2 > 0, and beats coarse baselines.
  WEAK PASS iso_embedded within 10% of best.
  FAIL      a simpler statistic is clearly a better predictor.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RULES = {
    "GoL":      {"birth": {3},    "survive": {2, 3}},
    "HighLife": {"birth": {3, 6}, "survive": {2, 3}},
}

HORIZONS_DEFAULT = [1, 5, 10, 25, 50, 100, 200]

ALL_PREDICTORS = [
    "iso_embedded",
    "singleton_comp",
    "small_comp_2",
    "small_comp_3",
    "small_comp_4",
    "small_comp_le4",
    "orth_deg0",
    "orth_deg0_diag0",
    "orth_deg0_diag_ge1",
    "moore_deg0",
    "moore_deg1",
    "moore_deg2",
    "boundary_cell",
    "component_count_t0",
    "block_var",
    "block_entropy",
]

# Coarse baselines that iso_embedded should outperform
COARSE_BASELINES = {"component_count_t0", "block_var", "block_entropy"}


# ---------------------------------------------------------------------------
# CA engine (self-contained, no ca.py import)
# ---------------------------------------------------------------------------

def step_ca(grid: np.ndarray, birth: set, survive: set) -> np.ndarray:
    """One toroidal Moore-neighbourhood CA update."""
    n = np.zeros_like(grid, dtype=np.int16)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            n += np.roll(np.roll(grid, dx, axis=0), dy, axis=1)
    born = (~grid) & np.isin(n, list(birth))
    surv = grid & np.isin(n, list(survive))
    return born | surv


def count_components_4_periodic(grid: np.ndarray) -> int:
    """Count 4-connected live components on a torus via union-find."""
    if grid.sum() == 0:
        return 0
    L = grid.shape[0]
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    labels, nlab = ndimage.label(grid, structure=struct)
    if nlab <= 1:
        return int(nlab)

    parent = np.arange(nlab + 1)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        if a == 0 or b == 0:
            return
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for j in range(L):
        union(labels[0, j], labels[L - 1, j])
    for i in range(L):
        union(labels[i, 0], labels[i, L - 1])

    return len({find(x) for x in range(1, nlab + 1)})


def component_sizes_4_periodic(grid: np.ndarray) -> np.ndarray:
    """
    Return array of 4-connected component sizes on a torus.
    Periodic boundaries merged via union-find.
    Returns int array (one entry per component), or empty array if no live cells.
    """
    if grid.sum() == 0:
        return np.array([], dtype=int)
    L = grid.shape[0]
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    labels, nlab = ndimage.label(grid, structure=struct)
    if nlab == 0:
        return np.array([], dtype=int)

    parent = np.arange(nlab + 1)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        if a == 0 or b == 0:
            return
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for j in range(L):
        union(labels[0, j], labels[L - 1, j])
    for i in range(L):
        union(labels[i, 0], labels[i, L - 1])

    # Accumulate pixel counts per merged-component root
    raw_sizes = np.bincount(labels.ravel(), minlength=nlab + 1)  # index 0 = background
    root_sizes: dict[int, int] = {}
    for lbl in range(1, nlab + 1):
        r = find(lbl)
        root_sizes[r] = root_sizes.get(r, 0) + int(raw_sizes[lbl])
    return np.array(list(root_sizes.values()), dtype=int)


# ---------------------------------------------------------------------------
# Topology predictor computation
# ---------------------------------------------------------------------------

def compute_topology_predictors(grid: np.ndarray, block_size: int = 8) -> dict:
    """
    Compute all topology predictors at t=0 in one vectorised pass.

    Neighbour conventions (periodic torus):
      orth neighbours:  up, down, left, right  (4-connectivity)
      diag neighbours:  NW, NE, SW, SE         (diagonal corners)
      Moore neighbours: all 8

    Definitions:
      orth_n[i,j] = number of orthogonal live neighbours of (i,j)
      diag_n[i,j] = number of diagonal live neighbours of (i,j)
      moore_n     = orth_n + diag_n
    """
    L = grid.shape[0]
    alive = grid.astype(bool)

    # --- Neighbour sums (vectorised via roll) ---
    up    = np.roll(grid, 1,  axis=0).astype(int)
    down  = np.roll(grid, -1, axis=0).astype(int)
    left  = np.roll(grid, 1,  axis=1).astype(int)
    right = np.roll(grid, -1, axis=1).astype(int)
    orth_n = up + down + left + right

    nw = np.roll(np.roll(grid, 1,  axis=0), 1,  axis=1).astype(int)
    ne = np.roll(np.roll(grid, 1,  axis=0), -1, axis=1).astype(int)
    sw = np.roll(np.roll(grid, -1, axis=0), 1,  axis=1).astype(int)
    se = np.roll(np.roll(grid, -1, axis=0), -1, axis=1).astype(int)
    diag_n = nw + ne + sw + se

    moore_n = orth_n + diag_n

    # --- Moore-degree classes ---
    moore_deg0 = int((alive & (moore_n == 0)).sum())
    moore_deg1 = int((alive & (moore_n == 1)).sum())
    moore_deg2 = int((alive & (moore_n == 2)).sum())

    # --- Orthogonal-degree classes ---
    orth_deg0          = int((alive & (orth_n == 0)).sum())
    orth_deg0_diag0    = int((alive & (orth_n == 0) & (diag_n == 0)).sum())
    orth_deg0_diag_ge1 = int((alive & (orth_n == 0) & (diag_n > 0)).sum())
    iso_embedded       = orth_deg0_diag_ge1  # identical by definition

    # --- Boundary cells: alive with ≥1 orthogonal dead neighbour ---
    # Each cell has exactly 4 orth neighbours on the torus; orth_n < 4 ⟺ ≥1 dead.
    boundary_cell = int((alive & (orth_n < 4)).sum())

    # --- Component sizes → small-component counts ---
    sizes = component_sizes_4_periodic(grid)
    component_count_t0 = len(sizes)
    if len(sizes) > 0:
        singleton_comp = int((sizes == 1).sum())
        small_comp_2   = int((sizes == 2).sum())
        small_comp_3   = int((sizes == 3).sum())
        small_comp_4   = int((sizes == 4).sum())
        small_comp_le4 = int((sizes <= 4).sum())
    else:
        singleton_comp = small_comp_2 = small_comp_3 = small_comp_4 = small_comp_le4 = 0

    # --- Block features ---
    bs = block_size if (L % block_size == 0) else (4 if L % 4 == 0 else 1)
    if bs > 1:
        blocks = grid.reshape(L // bs, bs, L // bs, bs).mean(axis=(1, 3)).ravel()
        block_var = float(np.var(blocks))
        hist, _ = np.histogram(blocks, bins=10, range=(0.0, 1.0), density=False)
        p = hist.astype(float)
        p = p / p.sum() if p.sum() > 0 else p
        p = p[p > 0]
        block_entropy = float(-(p * np.log2(p)).sum()) if len(p) > 0 else 0.0
    else:
        block_var = float(np.var(grid.ravel().astype(float)))
        block_entropy = 0.0

    return {
        "iso_embedded":        iso_embedded,
        "singleton_comp":      singleton_comp,
        "small_comp_2":        small_comp_2,
        "small_comp_3":        small_comp_3,
        "small_comp_4":        small_comp_4,
        "small_comp_le4":      small_comp_le4,
        "orth_deg0":           orth_deg0,
        "orth_deg0_diag0":     orth_deg0_diag0,
        "orth_deg0_diag_ge1":  orth_deg0_diag_ge1,
        "moore_deg0":          moore_deg0,
        "moore_deg1":          moore_deg1,
        "moore_deg2":          moore_deg2,
        "boundary_cell":       boundary_cell,
        "component_count_t0":  component_count_t0,
        "block_var":           block_var,
        "block_entropy":       block_entropy,
    }


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_dataset(args) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    rows = []
    horizons = sorted(args.horizons)
    max_h = max(horizons)
    horizon_set = set(horizons)

    total = len(RULES) * len(args.sizes) * len(args.densities) * args.n_samples
    pbar = tqdm(total=total, desc="Simulating topology dataset")

    sample_id = 0
    for rule_name, rule in RULES.items():
        for L in args.sizes:
            block_size = 8 if L >= 64 else 4
            for rho in args.densities:
                condition_id = f"{rule_name}_L{L}_rho{rho:.2f}"
                for _ in range(args.n_samples):
                    grid = rng.random((L, L)) < rho
                    live_count = int(grid.sum())
                    density = live_count / (L * L)

                    preds = compute_topology_predictors(grid, block_size)
                    comp0 = preds["component_count_t0"]

                    # Step forward, recording C(t+k)
                    comps: dict[int, int] = {}
                    g = grid.copy()
                    for t in range(1, max_h + 1):
                        g = step_ca(g, rule["birth"], rule["survive"])
                        if t in horizon_set:
                            comps[t] = count_components_4_periodic(g)

                    for h in horizons:
                        row = {
                            "sample_id":     sample_id,
                            "condition_id":  condition_id,
                            "rule":          rule_name,
                            "L":             L,
                            "rho":           rho,
                            "horizon":       h,
                            "live_count":    live_count,
                            "density":       density,
                            "fine_net":      comps[h] - comp0,
                            "components_tk": comps[h],
                        }
                        row.update(preds)
                        rows.append(row)

                    sample_id += 1
                    pbar.update(1)

    pbar.close()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _residual_r2(y: np.ndarray, X_base: np.ndarray, x_pred: np.ndarray) -> float:
    """
    Residual R² of predictor x_pred for target y after partialling out X_base.
    Both residual and predictor are standardised before fitting.
    """
    if np.std(y) < 1e-10:
        return 0.0
    base = LinearRegression().fit(X_base, y)
    resid = y - base.predict(X_base)
    if np.std(resid) < 1e-10 or np.std(x_pred) < 1e-10:
        return 0.0
    r_s = StandardScaler().fit_transform(resid.reshape(-1, 1)).ravel()
    x_s = StandardScaler().fit_transform(x_pred.reshape(-1, 1)).ravel()
    m = LinearRegression().fit(x_s.reshape(-1, 1), r_s)
    return float(r2_score(r_s, m.predict(x_s.reshape(-1, 1))))


def _delta_r2(y: np.ndarray, X_base: np.ndarray, x_pred: np.ndarray) -> float:
    """
    ΔR² = R²(base + predictor) − R²(base only), on raw (unstandardised) scale.
    """
    if np.std(y) < 1e-10:
        return 0.0
    base = LinearRegression().fit(X_base, y)
    r2_base = float(r2_score(y, base.predict(X_base)))
    X_full = np.column_stack([X_base, x_pred])
    full = LinearRegression().fit(X_full, y)
    r2_full = float(r2_score(y, full.predict(X_full)))
    return r2_full - r2_base


def analyze(df: pd.DataFrame, predictors: list, outdir: Path) -> pd.DataFrame:
    """
    Compute residual R² and ΔR² per predictor for every (condition, horizon) cell.
    """
    rows = []
    group_cols = ["condition_id", "rule", "L", "rho", "horizon"]
    for keys, g in tqdm(df.groupby(group_cols), desc="Condition-horizon fits"):
        condition_id, rule, L, rho, horizon = keys
        y = g["fine_net"].to_numpy(float)
        X_base = g[["live_count", "density"]].to_numpy(float)

        row: dict = {
            "condition_id": condition_id,
            "rule":         rule,
            "L":            L,
            "rho":          rho,
            "horizon":      horizon,
            "n":            len(g),
        }
        for pred in predictors:
            x = g[pred].to_numpy(float)
            row[f"resid_R2_{pred}"] = _residual_r2(y, X_base, x)
            row[f"delta_R2_{pred}"] = _delta_r2(y, X_base, x)
        rows.append(row)

    return pd.DataFrame(rows)


def build_global_summary(cond: pd.DataFrame, predictors: list) -> pd.DataFrame:
    """
    Aggregate across all (condition, horizon) cells per predictor.
    """
    records = []
    for pred in predictors:
        col_r = f"resid_R2_{pred}"
        col_d = f"delta_R2_{pred}"
        records.append({
            "predictor":              pred,
            "mean_resid_R2":          float(cond[col_r].mean()),
            "min_resid_R2":           float(cond[col_r].min()),
            "max_resid_R2":           float(cond[col_r].max()),
            "mean_delta_R2":          float(cond[col_d].mean()),
            "frac_positive_resid_R2": float((cond[col_r] > 0).mean()),
        })
    return (
        pd.DataFrame(records)
        .sort_values("mean_resid_R2", ascending=False)
        .reset_index(drop=True)
    )


def build_incremental_summary(cond: pd.DataFrame, predictors: list) -> pd.DataFrame:
    """
    For each competitor, compute iso_embedded's advantage in residual R²
    (condition-horizon-level difference, then aggregated).
    """
    iso_r2 = cond["resid_R2_iso_embedded"].to_numpy(float)
    records = []
    for pred in predictors:
        if pred == "iso_embedded":
            continue
        comp_r2 = cond[f"resid_R2_{pred}"].to_numpy(float)
        diff = iso_r2 - comp_r2
        records.append({
            "competitor":                   pred,
            "mean_iso_minus_comp_resid_R2": float(diff.mean()),
            "frac_iso_ge_comp":             float((iso_r2 >= comp_r2).mean()),
            "frac_iso_within_5pct_of_comp": float((iso_r2 >= comp_r2 * 0.95).mean()),
        })
    return (
        pd.DataFrame(records)
        .sort_values("mean_iso_minus_comp_resid_R2", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(cond: pd.DataFrame, glob: pd.DataFrame, outdir: Path) -> None:
    """
    Fig A: Horizontal bar chart of mean residual R² per predictor (global average).
    Fig B: Line plot of mean residual R² vs horizon for top-N predictors.
    """
    # --- Figure A: Overall ranking ---
    plot_preds = glob["predictor"].tolist()
    means_r2   = glob["mean_resid_R2"].tolist()
    means_dr2  = glob["mean_delta_R2"].tolist()

    colours = ["#d62728" if p == "iso_embedded" else "#1f77b4" for p in plot_preds]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    ax.barh(plot_preds, means_r2, color=colours)
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel("Mean residual R2 (all condition-horizon cells)")
    ax.set_title("Residual R2: iso_embedded vs topology competitors")
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=8)

    ax = axes[1]
    ax.barh(plot_preds, means_dr2, color=colours)
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel("Mean delta R2 beyond density baseline")
    ax.set_title("Delta R2: iso_embedded vs topology competitors")
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    plt.savefig(outdir / "fig_topology_baseline_r2.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- Figure B: Residual R² vs horizon for top-5 + iso_embedded ---
    # Take 4 best competitors by mean_resid_R2, always include iso_embedded
    top_competitors = [p for p in glob["predictor"] if p != "iso_embedded"][:4]
    top_preds = ["iso_embedded"] + top_competitors

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    for pred in top_preds:
        col = f"resid_R2_{pred}"
        vals = cond.groupby("horizon")[col].mean().reset_index().sort_values("horizon")
        lw = 2.5 if pred == "iso_embedded" else 1.2
        ls = "-"  if pred == "iso_embedded" else "--"
        ax2.plot(vals["horizon"], vals[col], label=pred, linewidth=lw, linestyle=ls, marker="o")

    ax2.set_xscale("log")
    ax2.axhline(0, color="k", linewidth=0.8)
    ax2.set_xlabel("Horizon k")
    ax2.set_ylabel("Mean residual R2 (global)")
    ax2.set_title("Residual R2 vs horizon: iso_embedded vs top competitors")
    ax2.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(outdir / "fig_topology_incremental.png", dpi=200, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def write_verdict(
    glob: pd.DataFrame,
    incr: pd.DataFrame,
    args,
    outdir: Path,
) -> str:
    """
    Assess whether iso_embedded is genuinely special.

    PASS:      iso_embedded is best or within 5% of best (by mean_resid_R2),
               mean_resid_R2 > 0, and outperforms all coarse baselines.
    WEAK PASS: iso_embedded within 10% of best and mean_resid_R2 > 0.
    FAIL:      otherwise.
    """
    iso_row  = glob[glob["predictor"] == "iso_embedded"].iloc[0]
    iso_r2   = float(iso_row["mean_resid_R2"])
    best_r2  = float(glob["mean_resid_R2"].max())
    best_pred = str(glob.loc[glob["mean_resid_R2"].idxmax(), "predictor"])

    within_5pct  = iso_r2 >= best_r2 * 0.95
    within_10pct = iso_r2 >= best_r2 * 0.90

    coarse_r2 = {
        p: float(glob.loc[glob["predictor"] == p, "mean_resid_R2"].values[0])
        for p in COARSE_BASELINES
    }
    iso_beats_coarse = all(iso_r2 > v for v in coarse_r2.values())

    if within_5pct and iso_r2 > 0 and iso_beats_coarse:
        verdict = "PASS"
    elif within_10pct and iso_r2 > 0:
        verdict = "WEAK PASS"
    else:
        verdict = "FAIL"

    lines: list[str] = []
    lines.append("=" * 62)
    lines.append("TOPOLOGY BASELINE CONTROLS — VERDICT")
    lines.append("=" * 62)
    lines.append(f"Rules:     {list(RULES.keys())}")
    lines.append(f"Sizes:     {args.sizes}")
    lines.append(f"Densities: {args.densities}")
    lines.append(f"Horizons:  {args.horizons}")
    lines.append(f"n_samples: {args.n_samples}")
    lines.append("")
    lines.append(f"VERDICT: {verdict}")
    lines.append("")
    lines.append(f"iso_embedded mean residual R2 = {iso_r2:.4f}")
    lines.append(f"Best predictor:                {best_pred} = {best_r2:.4f}")
    lines.append(f"iso within 5% of best:         {within_5pct}")
    lines.append(f"iso beats coarse baselines:    {iso_beats_coarse}")
    for p, v in coarse_r2.items():
        lines.append(f"  {p}: {v:.4f} ({'below' if iso_r2 > v else 'ABOVE'} iso)")
    lines.append("")
    lines.append("Full ranking (sorted by mean residual R2):")
    lines.append(f"  {'Predictor':<28}  resid_R2   delta_R2   frac_pos")
    lines.append(f"  {'-'*28}  ---------  ---------  --------")
    for _, row in glob.iterrows():
        marker = "  <-- iso_embedded" if row["predictor"] == "iso_embedded" else ""
        lines.append(
            f"  {row['predictor']:<28}  {row['mean_resid_R2']:9.4f}  "
            f"{row['mean_delta_R2']:9.4f}  {row['frac_positive_resid_R2']:8.3f}{marker}"
        )
    lines.append("")
    lines.append("Incremental comparison (iso_embedded vs each competitor):")
    lines.append(f"  {'Competitor':<28}  iso_minus_comp  frac_iso_ge")
    lines.append(f"  {'-'*28}  --------------  -----------")
    for _, row in incr.iterrows():
        lines.append(
            f"  {row['competitor']:<28}  {row['mean_iso_minus_comp_resid_R2']:+14.4f}  "
            f"{row['frac_iso_ge_comp']:11.3f}"
        )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  PASS      => iso_embedded is genuinely special among simple topology")
    lines.append("               statistics; no simpler count matches its predictive content.")
    lines.append("  WEAK PASS => iso_embedded is competitive but not clearly dominant.")
    lines.append("  FAIL      => a simpler statistic is a better predictor; revisit framing.")
    lines.append("")
    lines.append("Note on orth_deg0_diag_ge1:")
    lines.append("  By definition, orth_deg0_diag_ge1 == iso_embedded (same mask).")
    lines.append("  Identical R² confirms the computation is consistent.")

    text = "\n".join(lines)
    (outdir / "topology_baseline_verdict.txt").write_text(text)
    print(text)
    return verdict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Topology baseline controls for the embedded-isolate paper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-samples",  type=int,   default=1000,
                   help="Worlds per condition")
    p.add_argument("--seed",       type=int,   default=42,
                   help="Global RNG seed")
    p.add_argument("--sizes",      type=int,   nargs="+", default=[64, 128],
                   help="Grid sizes")
    p.add_argument("--densities",  type=float, nargs="+", default=[0.20, 0.25, 0.30, 0.35],
                   help="Initial live-cell densities")
    p.add_argument("--horizons",   type=int,   nargs="+", default=HORIZONS_DEFAULT,
                   help="Forecast horizons")
    p.add_argument("--quick",      action="store_true",
                   help="Quick mode: L=64, 2 densities, n_samples=50")
    p.add_argument("--reuse-raw",  action="store_true",
                   help="Reuse existing topology_baseline_raw.csv if present")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.sizes     = [64]
        args.densities = [0.25, 0.30]
        args.n_samples = min(args.n_samples, 50)

    outdir = ROOT / "outputs" / "topology_baselines"
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / "topology_baseline_raw.csv"

    if args.reuse_raw and raw_path.exists():
        print(f"Reusing raw dataset: {raw_path}")
        df = pd.read_csv(raw_path)
    else:
        df = simulate_dataset(args)
        df.to_csv(raw_path, index=False)
        print(f"Saved raw dataset: {raw_path}  ({len(df):,} rows)")

    print("Fitting models per condition-horizon …")
    cond = analyze(df, ALL_PREDICTORS, outdir)
    cond.to_csv(outdir / "topology_baseline_condition_summary.csv", index=False)

    glob = build_global_summary(cond, ALL_PREDICTORS)
    glob.to_csv(outdir / "topology_baseline_global_summary.csv", index=False)

    incr = build_incremental_summary(cond, ALL_PREDICTORS)
    incr.to_csv(outdir / "topology_baseline_incremental_summary.csv", index=False)

    make_figures(cond, glob, outdir)
    verdict = write_verdict(glob, incr, args, outdir)

    print(f"\nTopology baseline controls complete.  Verdict: {verdict}")
    print("Outputs written to:", outdir)
    for name in [
        "topology_baseline_raw.csv",
        "topology_baseline_condition_summary.csv",
        "topology_baseline_global_summary.csv",
        "topology_baseline_incremental_summary.csv",
        "topology_baseline_verdict.txt",
        "fig_topology_baseline_r2.png",
        "fig_topology_incremental.png",
    ]:
        p = outdir / name
        size_str = f"{p.stat().st_size / 1024:.1f} KB" if p.exists() else "MISSING"
        print(f"  {name:<52} {size_str}")


if __name__ == "__main__":
    main()
