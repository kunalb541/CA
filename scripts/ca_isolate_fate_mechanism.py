#!/usr/bin/env python3
"""
CA isolate-fate mechanism decomposition.

Question:
What local one-step events carry the embedded-isolate residual response?

We already found:
- iso_embedded predicts residual future fine-component change across horizons.
Now we decompose the k=1 response into local fates of embedded isolates.

Definitions:
- periodic torus
- 4-connected fine components
- embedded isolate = alive, 4-isolated orthogonally, at least one diagonal live neighbor

Outputs:
outputs/isolate_fate/fate_raw.csv
outputs/isolate_fate/fate_condition_summary.csv
outputs/isolate_fate/fate_rule_summary.csv
outputs/isolate_fate/fate_global_summary.csv
outputs/isolate_fate/fate_verdict.txt
outputs/isolate_fate/fig_fate_r2.png
outputs/isolate_fate/fig_fate_slopes.png
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import ndimage
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


RULES = {
    "GoL": {"birth": {3}, "survive": {2, 3}},
    "HighLife": {"birth": {3, 6}, "survive": {2, 3}},
}


def step_ca(grid: np.ndarray, birth: set, survive: set) -> np.ndarray:
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
    L = grid.shape[0]
    if grid.sum() == 0:
        return 0

    struct = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=int)
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

    roots = {find(x) for x in range(1, nlab + 1)}
    return len(roots)


def neighbor_counts(grid: np.ndarray):
    up = np.roll(grid, 1, axis=0)
    down = np.roll(grid, -1, axis=0)
    left = np.roll(grid, 1, axis=1)
    right = np.roll(grid, -1, axis=1)
    orth = up.astype(int) + down.astype(int) + left.astype(int) + right.astype(int)

    diag1 = np.roll(np.roll(grid, 1, axis=0), 1, axis=1)
    diag2 = np.roll(np.roll(grid, 1, axis=0), -1, axis=1)
    diag3 = np.roll(np.roll(grid, -1, axis=0), 1, axis=1)
    diag4 = np.roll(np.roll(grid, -1, axis=0), -1, axis=1)
    diag = diag1.astype(int) + diag2.astype(int) + diag3.astype(int) + diag4.astype(int)

    moore = orth + diag
    return orth, diag, moore


def embedded_isolate_mask(grid: np.ndarray) -> np.ndarray:
    orth, diag, _ = neighbor_counts(grid)
    return grid & (orth == 0) & (diag > 0)


def count_local_components_window(grid: np.ndarray, i: int, j: int, radius: int = 2) -> int:
    """Count 4-connected components in a toroidal local window centered at (i,j)."""
    L = grid.shape[0]
    coords_i = [(i + di) % L for di in range(-radius, radius + 1)]
    coords_j = [(j + dj) % L for dj in range(-radius, radius + 1)]
    window = grid[np.ix_(coords_i, coords_j)]
    struct = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=int)
    _, nlab = ndimage.label(window, structure=struct)
    return int(nlab)


def fate_counts(grid0: np.ndarray, grid1: np.ndarray, radius: int = 2) -> dict:
    """
    Count one-step fates of embedded isolates at t0.

    Fates:
    - iso_count
    - iso_die: isolate cell dead at t+1
    - iso_survive: isolate cell alive at t+1
    - iso_survive_isolated: alive and still 4-isolated at t+1
    - iso_survive_connected: alive and has at least one orthogonal live neighbor at t+1
    - iso_orth_birth_any: at least one orthogonal neighbor of the isolate is born at t+1
    - iso_diag_birth_any: at least one diagonal neighbor born at t+1
    - iso_local_window_delta_sum: sum over isolates of local 5x5 component change C1-C0
    - iso_local_window_loss_sum: sum over isolates of max(0, C0-C1)
    - iso_local_window_gain_sum: sum over isolates of max(0, C1-C0)
    """
    L = grid0.shape[0]
    iso_mask = embedded_isolate_mask(grid0)
    coords = np.argwhere(iso_mask)

    orth1, diag1, _ = neighbor_counts(grid1)
    births = (~grid0) & grid1

    out = {
        "iso_count": int(len(coords)),
        "iso_die": 0,
        "iso_survive": 0,
        "iso_survive_isolated": 0,
        "iso_survive_connected": 0,
        "iso_orth_birth_any": 0,
        "iso_diag_birth_any": 0,
        "iso_local_window_delta_sum": 0,
        "iso_local_window_loss_sum": 0,
        "iso_local_window_gain_sum": 0,
    }

    if len(coords) == 0:
        return out

    for i, j in coords:
        alive1 = bool(grid1[i, j])

        if not alive1:
            out["iso_die"] += 1
        else:
            out["iso_survive"] += 1
            if orth1[i, j] == 0:
                out["iso_survive_isolated"] += 1
            else:
                out["iso_survive_connected"] += 1

        orth_neighbors = [
            ((i - 1) % L, j),
            ((i + 1) % L, j),
            (i, (j - 1) % L),
            (i, (j + 1) % L),
        ]
        diag_neighbors = [
            ((i - 1) % L, (j - 1) % L),
            ((i - 1) % L, (j + 1) % L),
            ((i + 1) % L, (j - 1) % L),
            ((i + 1) % L, (j + 1) % L),
        ]

        if any(births[a, b] for a, b in orth_neighbors):
            out["iso_orth_birth_any"] += 1
        if any(births[a, b] for a, b in diag_neighbors):
            out["iso_diag_birth_any"] += 1

        c0 = count_local_components_window(grid0, int(i), int(j), radius=radius)
        c1 = count_local_components_window(grid1, int(i), int(j), radius=radius)
        delta = c1 - c0
        out["iso_local_window_delta_sum"] += int(delta)
        out["iso_local_window_loss_sum"] += int(max(0, -delta))
        out["iso_local_window_gain_sum"] += int(max(0, delta))

    return out


def block_features(grid: np.ndarray, block_size: int = 8) -> tuple[float, float]:
    L = grid.shape[0]
    assert L % block_size == 0
    b = block_size
    blocks = grid.reshape(L // b, b, L // b, b).mean(axis=(1, 3)).ravel()
    var = float(np.var(blocks))
    hist, _ = np.histogram(blocks, bins=10, range=(0, 1), density=False)
    p = hist.astype(float)
    p = p / p.sum() if p.sum() > 0 else p
    p = p[p > 0]
    ent = float(-(p * np.log2(p)).sum()) if len(p) else 0.0
    return var, ent


def simulate(args) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    rows = []
    total = len(RULES) * len(args.sizes) * len(args.densities) * args.n_samples
    pbar = tqdm(total=total, desc="Simulating isolate fate dataset")
    sample_id = 0

    for rule_name, rule in RULES.items():
        for L in args.sizes:
            block_size = 8 if L >= 64 else 4
            for rho in args.densities:
                condition_id = f"{rule_name}_L{L}_rho{rho:.2f}"

                for _ in range(args.n_samples):
                    grid0 = rng.random((L, L)) < rho
                    grid1 = step_ca(grid0, rule["birth"], rule["survive"])

                    comp0 = count_components_4_periodic(grid0)
                    comp1 = count_components_4_periodic(grid1)
                    fine_net_1 = comp1 - comp0

                    live_count = int(grid0.sum())
                    density = live_count / (L * L)
                    block_var, block_entropy = block_features(grid0, block_size)

                    fates = fate_counts(grid0, grid1, radius=args.local_radius)

                    row = {
                        "sample_id": sample_id,
                        "condition_id": condition_id,
                        "rule": rule_name,
                        "L": L,
                        "rho": rho,
                        "live_count": live_count,
                        "density": density,
                        "block_var": block_var,
                        "block_entropy": block_entropy,
                        "components_t0": comp0,
                        "components_t1": comp1,
                        "fine_net_1": fine_net_1,
                    }
                    row.update(fates)
                    rows.append(row)

                    sample_id += 1
                    pbar.update(1)

    pbar.close()
    return pd.DataFrame(rows)


FATE_FEATURES = [
    "iso_count",
    "iso_die",
    "iso_survive",
    "iso_survive_isolated",
    "iso_survive_connected",
    "iso_orth_birth_any",
    "iso_diag_birth_any",
    "iso_local_window_delta_sum",
    "iso_local_window_loss_sum",
    "iso_local_window_gain_sum",
]

MODEL_FEATURES = {
    "iso_count": ["iso_count"],
    "death": ["iso_die"],
    "survive": ["iso_survive"],
    "survive_split": ["iso_survive_isolated", "iso_survive_connected"],
    "birth_bridge": ["iso_orth_birth_any", "iso_diag_birth_any"],
    "local_window": ["iso_local_window_delta_sum", "iso_local_window_loss_sum", "iso_local_window_gain_sum"],
    "cell_fates_all": [
        "iso_die", "iso_survive_isolated", "iso_survive_connected",
        "iso_orth_birth_any", "iso_diag_birth_any"
    ],
    "all_fates": [
        "iso_die", "iso_survive_isolated", "iso_survive_connected",
        "iso_orth_birth_any", "iso_diag_birth_any",
        "iso_local_window_delta_sum", "iso_local_window_loss_sum", "iso_local_window_gain_sum"
    ],
    "coarse": ["block_var"],
    "entropy": ["block_entropy"],
    "all_plus_coarse": [
        "iso_die", "iso_survive_isolated", "iso_survive_connected",
        "iso_orth_birth_any", "iso_diag_birth_any",
        "iso_local_window_delta_sum", "iso_local_window_loss_sum", "iso_local_window_gain_sum",
        "block_var", "block_entropy"
    ],
}


def residualize_target(g: pd.DataFrame) -> np.ndarray:
    X = g[["live_count", "density"]].to_numpy(float)
    y = g["fine_net_1"].to_numpy(float)
    base = LinearRegression().fit(X, y)
    return y - base.predict(X)


def cv_r2_for_features(g: pd.DataFrame, features: list[str], seed: int) -> float:
    y_resid = residualize_target(g)
    X = g[features].to_numpy(float)

    if np.std(y_resid) == 0:
        return np.nan
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if np.all(np.std(X, axis=0) == 0):
        return np.nan

    alphas = np.logspace(-6, 3, 20)
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)

    scores = []
    idx = np.arange(len(g))
    for train, test in cv.split(idx):
        Xtr, Xte = X[train], X[test]
        ytr, yte = y_resid[train], y_resid[test]
        if np.std(ytr) == 0 or np.std(yte) == 0:
            continue
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        scores.append(r2_score(yte, pred))

    return float(np.mean(scores)) if scores else np.nan


def fit_slope_single(g: pd.DataFrame, feature: str, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    y_resid = residualize_target(g)
    x = g[[feature]].to_numpy(float)

    if np.std(y_resid) == 0 or np.std(x) == 0:
        return {
            "slope_raw": np.nan,
            "slope_raw_ci_low": np.nan,
            "slope_raw_ci_high": np.nan,
            "slope_std": np.nan,
            "slope_std_ci_low": np.nan,
            "slope_std_ci_high": np.nan,
            "r2_in_sample": np.nan,
        }

    x_std = StandardScaler().fit_transform(x)
    y_std = StandardScaler().fit_transform(y_resid.reshape(-1, 1)).ravel()

    m_std = LinearRegression().fit(x_std, y_std)
    p_std = m_std.predict(x_std)

    m_raw = LinearRegression().fit(x, y_resid)

    boot_raw = []
    boot_std = []
    idx = np.arange(len(g))

    for _ in range(n_boot):
        bidx = rng.choice(idx, size=len(idx), replace=True)
        gb = g.iloc[bidx]
        yb = residualize_target(gb)
        xb = gb[[feature]].to_numpy(float)

        if np.std(yb) == 0 or np.std(xb) == 0:
            continue

        xb_std = StandardScaler().fit_transform(xb)
        yb_std = StandardScaler().fit_transform(yb.reshape(-1, 1)).ravel()

        boot_raw.append(float(LinearRegression().fit(xb, yb).coef_[0]))
        boot_std.append(float(LinearRegression().fit(xb_std, yb_std).coef_[0]))

    def ci(vals):
        vals = np.asarray(vals, dtype=float)
        if len(vals) == 0:
            return np.nan, np.nan
        return np.percentile(vals, [2.5, 97.5])

    raw_lo, raw_hi = ci(boot_raw)
    std_lo, std_hi = ci(boot_std)

    return {
        "slope_raw": float(m_raw.coef_[0]),
        "slope_raw_ci_low": raw_lo,
        "slope_raw_ci_high": raw_hi,
        "slope_std": float(m_std.coef_[0]),
        "slope_std_ci_low": std_lo,
        "slope_std_ci_high": std_hi,
        "r2_in_sample": float(r2_score(y_std, p_std)),
    }


def analyze_condition(g: pd.DataFrame, seed: int, n_boot: int) -> list[dict]:
    rows = []
    condition_id = g["condition_id"].iloc[0]
    rule = g["rule"].iloc[0]
    L = int(g["L"].iloc[0])
    rho = float(g["rho"].iloc[0])

    # Model R2 rows
    for model_name, features in MODEL_FEATURES.items():
        r2 = cv_r2_for_features(g, features, seed=seed)
        rows.append({
            "scope": "condition",
            "condition_id": condition_id,
            "rule": rule,
            "L": L,
            "rho": rho,
            "model": model_name,
            "features": ",".join(features),
            "cv_R2": r2,
            "slope_feature": "",
            "slope_raw": np.nan,
            "slope_raw_ci_low": np.nan,
            "slope_raw_ci_high": np.nan,
            "slope_std": np.nan,
            "slope_std_ci_low": np.nan,
            "slope_std_ci_high": np.nan,
            "r2_in_sample": np.nan,
        })

    # Single-feature slopes for interpretability
    for feat in FATE_FEATURES + ["block_var", "block_entropy"]:
        slope = fit_slope_single(g, feat, n_boot=n_boot, seed=seed + 100)
        rows.append({
            "scope": "condition_slope",
            "condition_id": condition_id,
            "rule": rule,
            "L": L,
            "rho": rho,
            "model": f"slope_{feat}",
            "features": feat,
            "cv_R2": np.nan,
            "slope_feature": feat,
            **slope,
        })

    return rows


def analyze(df: pd.DataFrame, args, outdir: Path):
    all_rows = []

    for condition_id, g in tqdm(df.groupby("condition_id"), desc="Analyzing fate mechanism"):
        all_rows.extend(analyze_condition(g, seed=args.seed, n_boot=args.n_boot))

    cond = pd.DataFrame(all_rows)
    cond.to_csv(outdir / "fate_condition_summary.csv", index=False)

    # Aggregate model R2 by rule and global
    model_rows = cond[cond["scope"].eq("condition")].copy()
    slope_rows = cond[cond["scope"].eq("condition_slope")].copy()

    rule_summary = []
    for (rule, model), g in model_rows.groupby(["rule", "model"]):
        vals = g["cv_R2"].to_numpy(float)
        rule_summary.append({
            "scope": "rule_model",
            "rule": rule,
            "model": model,
            "mean_cv_R2": np.nanmean(vals),
            "sd_cv_R2": np.nanstd(vals, ddof=1),
            "min_cv_R2": np.nanmin(vals),
            "max_cv_R2": np.nanmax(vals),
            "n_conditions": len(g),
        })

    for (rule, feat), g in slope_rows.groupby(["rule", "slope_feature"]):
        vals = g["slope_raw"].to_numpy(float)
        r2s = g["r2_in_sample"].to_numpy(float)
        rule_summary.append({
            "scope": "rule_slope",
            "rule": rule,
            "model": f"slope_{feat}",
            "mean_cv_R2": np.nan,
            "sd_cv_R2": np.nan,
            "min_cv_R2": np.nan,
            "max_cv_R2": np.nan,
            "n_conditions": len(g),
            "mean_slope_raw": np.nanmean(vals),
            "sd_slope_raw": np.nanstd(vals, ddof=1),
            "cv_slope_raw": np.nanstd(vals, ddof=1) / (abs(np.nanmean(vals)) + 1e-12),
            "frac_slope_CI_negative": np.nanmean(g["slope_raw_ci_high"].to_numpy(float) < 0),
            "frac_slope_CI_positive": np.nanmean(g["slope_raw_ci_low"].to_numpy(float) > 0),
            "mean_in_sample_R2": np.nanmean(r2s),
        })

    rule_df = pd.DataFrame(rule_summary)
    rule_df.to_csv(outdir / "fate_rule_summary.csv", index=False)

    global_summary = []

    for model, g in model_rows.groupby("model"):
        vals = g["cv_R2"].to_numpy(float)
        global_summary.append({
            "scope": "global_model",
            "model": model,
            "mean_cv_R2": np.nanmean(vals),
            "sd_cv_R2": np.nanstd(vals, ddof=1),
            "min_cv_R2": np.nanmin(vals),
            "max_cv_R2": np.nanmax(vals),
            "n_conditions": len(g),
        })

    for feat, g in slope_rows.groupby("slope_feature"):
        vals = g["slope_raw"].to_numpy(float)
        r2s = g["r2_in_sample"].to_numpy(float)
        global_summary.append({
            "scope": "global_slope",
            "model": f"slope_{feat}",
            "mean_cv_R2": np.nan,
            "sd_cv_R2": np.nan,
            "min_cv_R2": np.nan,
            "max_cv_R2": np.nan,
            "n_conditions": len(g),
            "mean_slope_raw": np.nanmean(vals),
            "sd_slope_raw": np.nanstd(vals, ddof=1),
            "cv_slope_raw": np.nanstd(vals, ddof=1) / (abs(np.nanmean(vals)) + 1e-12),
            "frac_slope_CI_negative": np.nanmean(g["slope_raw_ci_high"].to_numpy(float) < 0),
            "frac_slope_CI_positive": np.nanmean(g["slope_raw_ci_low"].to_numpy(float) > 0),
            "mean_in_sample_R2": np.nanmean(r2s),
        })

    global_df = pd.DataFrame(global_summary)
    global_df.to_csv(outdir / "fate_global_summary.csv", index=False)

    return cond, rule_df, global_df


def make_figures(global_df: pd.DataFrame, outdir: Path):
    models = [
        "iso_count", "death", "survive", "survive_split", "birth_bridge",
        "local_window", "cell_fates_all", "all_fates", "coarse", "entropy", "all_plus_coarse"
    ]
    g = global_df[global_df["scope"].eq("global_model")].copy()
    g = g[g["model"].isin(models)]
    g["model"] = pd.Categorical(g["model"], categories=models, ordered=True)
    g = g.sort_values("model")

    plt.figure(figsize=(10, 5))
    plt.bar(g["model"].astype(str), g["mean_cv_R2"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean CV R² on residual ΔC₁")
    plt.title("One-step isolate-fate mechanism models")
    plt.tight_layout()
    plt.savefig(outdir / "fig_fate_r2.png", dpi=200)
    plt.close()

    feats = [
        "iso_count", "iso_die", "iso_survive", "iso_survive_isolated",
        "iso_survive_connected", "iso_orth_birth_any", "iso_diag_birth_any",
        "iso_local_window_delta_sum", "iso_local_window_loss_sum", "iso_local_window_gain_sum",
        "block_var", "block_entropy"
    ]
    s = global_df[global_df["scope"].eq("global_slope")].copy()
    s["feature"] = s["model"].str.replace("slope_", "", regex=False)
    s = s[s["feature"].isin(feats)]
    s["feature"] = pd.Categorical(s["feature"], categories=feats, ordered=True)
    s = s.sort_values("feature")

    plt.figure(figsize=(11, 5))
    plt.bar(s["feature"].astype(str), s["mean_slope_raw"])
    plt.axhline(0, linewidth=1)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean raw slope on residual ΔC₁")
    plt.title("Single-feature fate slopes")
    plt.tight_layout()
    plt.savefig(outdir / "fig_fate_slopes.png", dpi=200)
    plt.close()


def write_verdict(global_df: pd.DataFrame, rule_df: pd.DataFrame, outdir: Path):
    gm = global_df[global_df["scope"].eq("global_model")].set_index("model")
    gs = global_df[global_df["scope"].eq("global_slope")].copy()
    gs["feature"] = gs["model"].str.replace("slope_", "", regex=False)
    gs = gs.set_index("feature")

    iso_r2 = float(gm.loc["iso_count", "mean_cv_R2"])
    death_r2 = float(gm.loc["death", "mean_cv_R2"])
    local_r2 = float(gm.loc["local_window", "mean_cv_R2"])
    all_fates_r2 = float(gm.loc["all_fates", "mean_cv_R2"])
    all_plus_r2 = float(gm.loc["all_plus_coarse", "mean_cv_R2"])
    coarse_r2 = float(gm.loc["coarse", "mean_cv_R2"])
    entropy_r2 = float(gm.loc["entropy", "mean_cv_R2"])

    lines = []
    lines.append("CA Isolate Fate Mechanism Decomposition")
    lines.append("")
    lines.append("Question: what one-step local events carry the embedded-isolate residual ΔC1 response?")
    lines.append("")
    lines.append("Global CV R2 on residual ΔC1:")
    for model in ["iso_count", "death", "survive", "survive_split", "birth_bridge", "local_window", "cell_fates_all", "all_fates", "coarse", "entropy", "all_plus_coarse"]:
        if model in gm.index:
            lines.append(f"  {model:18s}: {gm.loc[model, 'mean_cv_R2']:.4f}")

    lines.append("")
    lines.append("Global single-feature slopes:")
    for feat in ["iso_count", "iso_die", "iso_survive", "iso_survive_isolated", "iso_survive_connected", "iso_orth_birth_any", "iso_diag_birth_any", "iso_local_window_delta_sum", "iso_local_window_loss_sum", "iso_local_window_gain_sum", "block_var", "block_entropy"]:
        if feat in gs.index:
            lines.append(
                f"  {feat:30s}: "
                f"slope={gs.loc[feat, 'mean_slope_raw']:.4f}, "
                f"frac_CI_neg={gs.loc[feat, 'frac_slope_CI_negative']:.2f}, "
                f"frac_CI_pos={gs.loc[feat, 'frac_slope_CI_positive']:.2f}, "
                f"in_sample_R2={gs.loc[feat, 'mean_in_sample_R2']:.4f}"
            )

    lines.append("")
    if all_fates_r2 > iso_r2 + 0.02 and all_fates_r2 > max(coarse_r2, entropy_r2) + 0.02:
        verdict = "PASS"
        lines.append("VERDICT: PASS")
        lines.append("Detailed fate features improve over iso_count and dominate coarse/entropy baselines.")
    elif abs(all_fates_r2 - iso_r2) <= 0.02 and iso_r2 > max(coarse_r2, entropy_r2) + 0.05:
        verdict = "WEAK PASS"
        lines.append("VERDICT: WEAK PASS")
        lines.append("iso_count itself captures most of the one-step mechanism; detailed fates add little.")
    elif local_r2 > iso_r2 + 0.02:
        verdict = "LOCAL WINDOW PASS"
        lines.append("VERDICT: LOCAL WINDOW PASS")
        lines.append("Local window component changes improve over iso_count, suggesting neighborhood/component-context mechanism.")
    else:
        verdict = "FAIL/CHECK"
        lines.append("VERDICT: FAIL/CHECK")
        lines.append("Detailed one-step fates do not clearly explain the residual response beyond iso_count.")

    lines.append("")
    lines.append("Interpretation guide:")
    lines.append("- PASS means one-step fate decomposition identifies event carriers.")
    lines.append("- WEAK PASS means embedded-isolate count is already the sufficient object-level mechanism summary.")
    lines.append("- LOCAL WINDOW PASS means mechanism is local component-context rather than simple cell fate.")
    lines.append("- FAIL/CHECK means the residual response may be multi-step or nonlocal.")

    (outdir / "fate_verdict.txt").write_text("\n".join(lines))
    return verdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--n-boot", type=int, default=300)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--sizes", type=int, nargs="+", default=[64, 128])
    p.add_argument("--densities", type=float, nargs="+", default=[0.20, 0.25, 0.30, 0.35])
    p.add_argument("--local-radius", type=int, default=2)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--reuse-raw", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.sizes = [64]
        args.densities = [0.25, 0.30]
        args.n_samples = min(args.n_samples, 300)
        args.n_boot = min(args.n_boot, 100)

    outdir = ROOT / "outputs" / "isolate_fate"
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / "fate_raw.csv"

    if args.reuse_raw and raw_path.exists():
        print(f"Reusing raw file: {raw_path}")
        df = pd.read_csv(raw_path)
    else:
        df = simulate(args)
        df.to_csv(raw_path, index=False)
        print(f"Saved raw dataset: {raw_path}")

    cond, rule_df, global_df = analyze(df, args, outdir)
    make_figures(global_df, outdir)
    verdict = write_verdict(global_df, rule_df, outdir)

    print("\nCA isolate fate mechanism test complete.")
    print("Verdict:", verdict)
    print("Outputs:")
    for name in [
        "fate_raw.csv",
        "fate_condition_summary.csv",
        "fate_rule_summary.csv",
        "fate_global_summary.csv",
        "fate_verdict.txt",
        "fig_fate_r2.png",
        "fig_fate_slopes.png",
    ]:
        print(" ", outdir / name)


if __name__ == "__main__":
    main()
