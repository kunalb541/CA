from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ca import comp_count_periodic, embedded_isolated_coords, gol_step

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = str(ROOT / "outputs" / "selection_principle")

RULES = {
    "GoL": {"birth": {3}, "survive": {2, 3}},
    "HighLife": {"birth": {3, 6}, "survive": {2, 3}},
}

TARGET_COLUMNS = [
    "target_fine_net",
    "target_delta_components",
    "target_delta_density",
    "target_delta_block_var",
    "target_delta_block_entropy",
    "target_future_density",
    "target_future_live_count",
]

NON_ALIAS_OTHER_TARGETS = [
    target for target in TARGET_COLUMNS
    if target not in {"target_fine_net", "target_delta_components"}
]

AGGREGATE_TARGETS = [
    "target_delta_density",
    "target_delta_block_var",
    "target_delta_block_entropy",
    "target_future_density",
    "target_future_live_count",
]

MODEL_COLUMNS = {
    "controls": ["live_count", "density"],
    "iso": ["live_count", "density", "iso_embedded"],
    "coarse": ["live_count", "density", "block_var"],
    "entropy": ["live_count", "density", "block_entropy"],
    "all": ["live_count", "density", "iso_embedded", "block_var", "block_entropy", "perimeter_count"],
    "null_iso_shuffle": ["live_count", "density", "iso_embedded_shuffled"],
}

RESIDUAL_MODEL_COLUMNS = {
    "iso_only": ["iso_embedded"],
    "coarse_only": ["block_var"],
    "entropy_only": ["block_entropy"],
    "all_resid": ["iso_embedded", "block_var", "block_entropy"],
    "null_iso_only": ["iso_embedded_shuffled"],
}


@dataclass(frozen=True)
class RunConfig:
    quick: bool
    sizes: list[int]
    densities: list[float]
    n_samples_per_condition: int
    k: int
    seed: int
    n_boot: int
    alphas: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CA ODD selection-principle test")
    parser.add_argument("--quick", action="store_true", help="Run quick mode.")
    parser.add_argument("--n-boot", "--n-bootstrap", dest="n_boot", type=int, default=500, help="Bootstrap iterations.")
    parser.add_argument("--n-samples", type=int, default=None, help="Override samples per condition.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument(
        "--reuse-raw",
        action="store_true",
        help="Reuse outputs/selection_principle/selection_raw.csv instead of rerunning CA simulation.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RunConfig:
    quick_n = 300 if args.n_samples is None else args.n_samples
    full_n = 1000 if args.n_samples is None else args.n_samples
    if args.quick:
        return RunConfig(
            quick=True,
            sizes=[64],
            densities=[0.25, 0.30],
            n_samples_per_condition=quick_n,
            k=100,
            seed=args.seed,
            n_boot=args.n_boot,
            alphas=np.logspace(-6, 3, 20),
        )
    return RunConfig(
        quick=False,
        sizes=[64, 128],
        densities=[0.20, 0.25, 0.30, 0.35],
        n_samples_per_condition=full_n,
        k=100,
        seed=args.seed,
        n_boot=args.n_boot,
        alphas=np.logspace(-6, 3, 20),
    )


def ensure_dirs() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)


def life_like_step(grid: np.ndarray, birth: set[int], survive: set[int]) -> np.ndarray:
    if birth == {3} and survive == {2, 3}:
        return gol_step(grid)

    g = grid.astype(np.int8, copy=False)
    neighbors = (
        np.roll(g, 1, 0) + np.roll(g, -1, 0) +
        np.roll(g, 1, 1) + np.roll(g, -1, 1) +
        np.roll(np.roll(g, 1, 0), 1, 1) +
        np.roll(np.roll(g, 1, 0), -1, 1) +
        np.roll(np.roll(g, -1, 0), 1, 1) +
        np.roll(np.roll(g, -1, 0), -1, 1)
    )
    birth_mask = (grid == 0) & np.isin(neighbors, list(birth))
    survive_mask = (grid == 1) & np.isin(neighbors, list(survive))
    return (birth_mask | survive_mask).astype(np.uint8)


def block_density_view(grid: np.ndarray, block_size: int) -> np.ndarray:
    n = grid.shape[0]
    nb = n // block_size
    trimmed = grid[: nb * block_size, : nb * block_size]
    blocks = trimmed.reshape(nb, block_size, nb, block_size)
    return blocks.mean(axis=(1, 3))


def block_var(grid: np.ndarray, block_size: int) -> float:
    bd = block_density_view(grid, block_size)
    return float(np.var(bd, ddof=0))


def block_entropy(grid: np.ndarray, block_size: int, n_bins: int = 10) -> float:
    bd = block_density_view(grid, block_size).ravel()
    hist, _ = np.histogram(bd, bins=n_bins, range=(0.0, 1.0))
    total = hist.sum()
    if total == 0:
        return 0.0
    probs = hist[hist > 0] / total
    return float(-(probs * np.log2(probs)).sum())


def perimeter_count(grid: np.ndarray) -> int:
    vertical = np.count_nonzero(grid != np.roll(grid, -1, axis=0))
    horizontal = np.count_nonzero(grid != np.roll(grid, -1, axis=1))
    return int(vertical + horizontal)


def compute_state_features(grid: np.ndarray) -> dict[str, float]:
    L = int(grid.shape[0])
    live_count = int(grid.sum())
    density = float(live_count / (L * L))
    iso_embedded = int(len(embedded_isolated_coords(grid)))
    bsize = 8 if L >= 64 else 4
    return {
        "live_count": live_count,
        "density": density,
        "iso_embedded": iso_embedded,
        "block_var": block_var(grid, bsize),
        "block_entropy": block_entropy(grid, bsize),
        "perimeter_count": perimeter_count(grid),
        "fine_components": int(comp_count_periodic(grid)),
    }


def make_condition_id(rule: str, L: int, rho: float) -> str:
    return f"{rule}_L{L}_rho{rho:.2f}"


def sample_condition(
    rule_name: str,
    rule_cfg: dict[str, set[int]],
    L: int,
    rho: float,
    n_samples: int,
    k: int,
    rng: np.random.Generator,
    sample_offset: int,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    condition_id = make_condition_id(rule_name, L, rho)

    for local_idx in tqdm(range(n_samples), desc=condition_id, leave=False):
        grid = (rng.random((L, L)) < rho).astype(np.uint8)

        t0 = compute_state_features(grid)
        g = grid.copy()
        for _ in range(k):
            g = life_like_step(g, rule_cfg["birth"], rule_cfg["survive"])
        tk = compute_state_features(g)

        row = {
            "sample_id": sample_offset + local_idx,
            "rule": rule_name,
            "L": L,
            "rho": rho,
            "condition_id": condition_id,
            "live_count": t0["live_count"],
            "density": t0["density"],
            "iso_embedded": t0["iso_embedded"],
            "block_var": t0["block_var"],
            "block_entropy": t0["block_entropy"],
            "perimeter_count": t0["perimeter_count"],
            "fine_components_t0": t0["fine_components"],
            "fine_components_tk": tk["fine_components"],
            "target_fine_net": tk["fine_components"] - t0["fine_components"],
            "target_delta_components": tk["fine_components"] - t0["fine_components"],
            "target_delta_density": tk["density"] - t0["density"],
            "target_delta_block_var": tk["block_var"] - t0["block_var"],
            "target_delta_block_entropy": tk["block_entropy"] - t0["block_entropy"],
            "target_future_density": tk["density"],
            "target_future_live_count": tk["live_count"],
        }
        rows.append(row)
    return rows


def add_condition_level_nulls(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    pieces = []
    for cond_idx, (condition_id, sub) in enumerate(df.groupby("condition_id", sort=False)):
        srng = np.random.default_rng(seed + cond_idx)
        shuffled = sub["iso_embedded"].values.copy()
        srng.shuffle(shuffled)
        sub = sub.copy()
        sub["random_object_count_null"] = shuffled
        sub["iso_embedded_shuffled"] = shuffled
        for target in TARGET_COLUMNS:
            vals = sub[target].values.copy()
            srng.shuffle(vals)
            sub[f"{target}_shuffled"] = vals
        pieces.append(sub)
    return pd.concat(pieces, ignore_index=True)


def build_design_matrix(
    df: pd.DataFrame,
    columns: list[str],
    include_condition_effects: bool,
) -> tuple[np.ndarray, list[str]]:
    base = df[columns].astype(float).copy()
    names = list(columns)
    if include_condition_effects:
        dummies = pd.get_dummies(df["condition_id"], prefix="cond", drop_first=True, dtype=float)
        if not dummies.empty:
            base = pd.concat([base.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
            names.extend(dummies.columns.tolist())
    return base.values.astype(float), names


def cv_r2_scores(X: np.ndarray, y: np.ndarray, seed: int, alphas: np.ndarray) -> np.ndarray:
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    estimator = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=alphas, cv=cv),
    )
    scores = cross_val_score(estimator, X, y, cv=cv, scoring="r2", n_jobs=None)
    return scores.astype(float)


def select_alpha(X: np.ndarray, y: np.ndarray, seed: int, alphas: np.ndarray) -> float:
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    estimator = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=alphas, cv=cv),
    )
    estimator.fit(X, y)
    return float(estimator.named_steps["ridgecv"].alpha_)


def cv_r2_scores_fixed_alpha(X: np.ndarray, y: np.ndarray, seed: int, alpha: float) -> np.ndarray:
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    estimator = make_pipeline(
        StandardScaler(),
        Ridge(alpha=alpha),
    )
    scores = cross_val_score(estimator, X, y, cv=cv, scoring="r2", n_jobs=None)
    return scores.astype(float)


def summarize_scores(scores: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
    }


def standardized_feature_slope(
    df: pd.DataFrame,
    target: str,
    columns: list[str],
    feature_name: str,
    include_condition_effects: bool,
) -> float:
    X, feature_names = build_design_matrix(df, columns, include_condition_effects=include_condition_effects)
    y = df[target].values.astype(float)
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    Xs = x_scaler.fit_transform(X)
    ys = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
    model = LinearRegression().fit(Xs, ys)
    feat_idx = feature_names.index(feature_name)
    return float(model.coef_[feat_idx])


def standardized_iso_slope(
    df: pd.DataFrame,
    target: str,
    include_condition_effects: bool,
) -> float:
    return standardized_feature_slope(
        df=df,
        target=target,
        columns=MODEL_COLUMNS["all"],
        feature_name="iso_embedded",
        include_condition_effects=include_condition_effects,
    )


def stratified_bootstrap_df(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pieces = []
    for _, sub in df.groupby("condition_id", sort=False):
        idx = rng.integers(0, len(sub), size=len(sub))
        pieces.append(sub.iloc[idx].reset_index(drop=True))
    return pd.concat(pieces, ignore_index=True)


def percentile_ci(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5))


def bootstrap_scope_metrics(
    df: pd.DataFrame,
    target: str,
    include_condition_effects: bool,
    cfg: RunConfig,
    seed_offset: int,
    include_delta_ci: bool,
) -> dict[str, float]:
    iso_slopes = []
    null_iso_slopes = []

    delta_iso = []
    delta_coarse = []
    delta_entropy = []
    delta_all = []
    delta_null_iso = []

    alpha_map: dict[str, float] = {}
    if include_delta_ci:
        y_full = df[target].values.astype(float)
        X_controls_full, _ = build_design_matrix(df, MODEL_COLUMNS["controls"], include_condition_effects)
        X_iso_full, _ = build_design_matrix(df, MODEL_COLUMNS["iso"], include_condition_effects)
        X_coarse_full, _ = build_design_matrix(df, MODEL_COLUMNS["coarse"], include_condition_effects)
        X_entropy_full, _ = build_design_matrix(df, MODEL_COLUMNS["entropy"], include_condition_effects)
        X_all_full, _ = build_design_matrix(df, MODEL_COLUMNS["all"], include_condition_effects)
        X_null_iso_full, _ = build_design_matrix(df, MODEL_COLUMNS["null_iso_shuffle"], include_condition_effects)

        alpha_map = {
            "controls": select_alpha(X_controls_full, y_full, cfg.seed, cfg.alphas),
            "iso": select_alpha(X_iso_full, y_full, cfg.seed, cfg.alphas),
            "coarse": select_alpha(X_coarse_full, y_full, cfg.seed, cfg.alphas),
            "entropy": select_alpha(X_entropy_full, y_full, cfg.seed, cfg.alphas),
            "all": select_alpha(X_all_full, y_full, cfg.seed, cfg.alphas),
            "null_iso_shuffle": select_alpha(X_null_iso_full, y_full, cfg.seed, cfg.alphas),
        }

    iterator = tqdm(range(cfg.n_boot), desc=f"Bootstrap {target}", leave=False)
    for boot_idx in iterator:
        bdf = stratified_bootstrap_df(df, cfg.seed + seed_offset * 10000 + boot_idx)
        iso_slopes.append(standardized_iso_slope(bdf, target, include_condition_effects))
        null_iso_slopes.append(
            standardized_feature_slope(
                df=bdf,
                target=target,
                columns=MODEL_COLUMNS["null_iso_shuffle"],
                feature_name="iso_embedded_shuffled",
                include_condition_effects=include_condition_effects,
            )
        )

        if include_delta_ci:
            y = bdf[target].values.astype(float)
            X_controls, _ = build_design_matrix(bdf, MODEL_COLUMNS["controls"], include_condition_effects)
            X_iso, _ = build_design_matrix(bdf, MODEL_COLUMNS["iso"], include_condition_effects)
            X_coarse, _ = build_design_matrix(bdf, MODEL_COLUMNS["coarse"], include_condition_effects)
            X_entropy, _ = build_design_matrix(bdf, MODEL_COLUMNS["entropy"], include_condition_effects)
            X_all, _ = build_design_matrix(bdf, MODEL_COLUMNS["all"], include_condition_effects)
            X_null_iso, _ = build_design_matrix(bdf, MODEL_COLUMNS["null_iso_shuffle"], include_condition_effects)

            r2_controls = np.mean(cv_r2_scores_fixed_alpha(X_controls, y, cfg.seed, alpha_map["controls"]))
            r2_iso = np.mean(cv_r2_scores_fixed_alpha(X_iso, y, cfg.seed, alpha_map["iso"]))
            r2_coarse = np.mean(cv_r2_scores_fixed_alpha(X_coarse, y, cfg.seed, alpha_map["coarse"]))
            r2_entropy = np.mean(cv_r2_scores_fixed_alpha(X_entropy, y, cfg.seed, alpha_map["entropy"]))
            r2_all = np.mean(cv_r2_scores_fixed_alpha(X_all, y, cfg.seed, alpha_map["all"]))
            r2_null_iso = np.mean(cv_r2_scores_fixed_alpha(X_null_iso, y, cfg.seed, alpha_map["null_iso_shuffle"]))

            delta_iso.append(float(r2_iso - r2_controls))
            delta_coarse.append(float(r2_coarse - r2_controls))
            delta_entropy.append(float(r2_entropy - r2_controls))
            delta_all.append(float(r2_all - r2_controls))
            delta_null_iso.append(float(r2_null_iso - r2_controls))

    slope_low, slope_high = percentile_ci(iso_slopes)
    null_slope_low, null_slope_high = percentile_ci(null_iso_slopes)
    if include_delta_ci:
        iso_low, iso_high = percentile_ci(delta_iso)
        coarse_low, coarse_high = percentile_ci(delta_coarse)
        ent_low, ent_high = percentile_ci(delta_entropy)
        all_low, all_high = percentile_ci(delta_all)
        null_low, null_high = percentile_ci(delta_null_iso)
    else:
        iso_low = iso_high = np.nan
        coarse_low = coarse_high = np.nan
        ent_low = ent_high = np.nan
        all_low = all_high = np.nan
        null_low = null_high = np.nan
    return {
        "deltaR2_iso_ci_low": iso_low,
        "deltaR2_iso_ci_high": iso_high,
        "deltaR2_coarse_ci_low": coarse_low,
        "deltaR2_coarse_ci_high": coarse_high,
        "deltaR2_entropy_ci_low": ent_low,
        "deltaR2_entropy_ci_high": ent_high,
        "deltaR2_all_ci_low": all_low,
        "deltaR2_all_ci_high": all_high,
        "deltaR2_null_iso_shuffle_ci_low": null_low,
        "deltaR2_null_iso_shuffle_ci_high": null_high,
        "iso_slope_ci_low": slope_low,
        "iso_slope_ci_high": slope_high,
        "null_iso_slope_ci_low": null_slope_low,
        "null_iso_slope_ci_high": null_slope_high,
    }


def evaluate_scope(
    df: pd.DataFrame,
    scope: str,
    rule: str,
    L: float,
    rho: float,
    include_condition_effects: bool,
    cfg: RunConfig,
    scope_seed_offset: int,
    include_delta_ci: bool,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    rows = []
    null_rows = []

    for target_idx, target in enumerate(tqdm(TARGET_COLUMNS, desc=f"Models {scope}:{rule}", leave=False)):
        y = df[target].values.astype(float)
        y_null = df[f"{target}_shuffled"].values.astype(float)

        X_controls, _ = build_design_matrix(df, MODEL_COLUMNS["controls"], include_condition_effects)
        X_iso, _ = build_design_matrix(df, MODEL_COLUMNS["iso"], include_condition_effects)
        X_coarse, _ = build_design_matrix(df, MODEL_COLUMNS["coarse"], include_condition_effects)
        X_entropy, _ = build_design_matrix(df, MODEL_COLUMNS["entropy"], include_condition_effects)
        X_all, _ = build_design_matrix(df, MODEL_COLUMNS["all"], include_condition_effects)
        X_null_iso, _ = build_design_matrix(df, MODEL_COLUMNS["null_iso_shuffle"], include_condition_effects)

        scores_controls = cv_r2_scores(X_controls, y, cfg.seed, cfg.alphas)
        scores_iso = cv_r2_scores(X_iso, y, cfg.seed, cfg.alphas)
        scores_coarse = cv_r2_scores(X_coarse, y, cfg.seed, cfg.alphas)
        scores_entropy = cv_r2_scores(X_entropy, y, cfg.seed, cfg.alphas)
        scores_all = cv_r2_scores(X_all, y, cfg.seed, cfg.alphas)
        scores_null_iso = cv_r2_scores(X_null_iso, y, cfg.seed, cfg.alphas)
        scores_null_target = cv_r2_scores(X_iso, y_null, cfg.seed, cfg.alphas)

        s_controls = summarize_scores(scores_controls)
        s_iso = summarize_scores(scores_iso)
        s_coarse = summarize_scores(scores_coarse)
        s_entropy = summarize_scores(scores_entropy)
        s_all = summarize_scores(scores_all)
        s_null_iso = summarize_scores(scores_null_iso)
        s_null_target = summarize_scores(scores_null_target)

        iso_slope = standardized_iso_slope(df, target, include_condition_effects)
        boot = bootstrap_scope_metrics(
            df=df,
            target=target,
            include_condition_effects=include_condition_effects,
            cfg=cfg,
            seed_offset=scope_seed_offset * 100 + target_idx,
            include_delta_ci=include_delta_ci,
        )

        delta_iso = s_iso["mean"] - s_controls["mean"]
        delta_coarse = s_coarse["mean"] - s_controls["mean"]
        delta_entropy = s_entropy["mean"] - s_controls["mean"]
        delta_all = s_all["mean"] - s_controls["mean"]
        delta_null_iso = s_null_iso["mean"] - s_controls["mean"]
        delta_null_target = s_null_target["mean"] - s_controls["mean"]
        null_iso_slope = standardized_feature_slope(
            df=df,
            target=target,
            columns=MODEL_COLUMNS["null_iso_shuffle"],
            feature_name="iso_embedded_shuffled",
            include_condition_effects=include_condition_effects,
        )

        rows.append({
            "scope": scope,
            "rule": rule,
            "L": L,
            "rho": rho,
            "target": target,
            "R2_controls": s_controls["mean"],
            "R2_controls_std": s_controls["std"],
            "R2_iso": s_iso["mean"],
            "R2_iso_std": s_iso["std"],
            "R2_coarse": s_coarse["mean"],
            "R2_coarse_std": s_coarse["std"],
            "R2_entropy": s_entropy["mean"],
            "R2_entropy_std": s_entropy["std"],
            "R2_all": s_all["mean"],
            "R2_all_std": s_all["std"],
            "R2_null_iso_shuffle": s_null_iso["mean"],
            "R2_null_iso_shuffle_std": s_null_iso["std"],
            "R2_null_target_shuffle": s_null_target["mean"],
            "R2_null_target_shuffle_std": s_null_target["std"],
            "deltaR2_iso": delta_iso,
            "deltaR2_coarse": delta_coarse,
            "deltaR2_entropy": delta_entropy,
            "deltaR2_all": delta_all,
            "deltaR2_null_iso_shuffle": delta_null_iso,
            "deltaR2_null_target_shuffle": delta_null_target,
            "iso_slope_standardized": iso_slope,
            "null_iso_slope_standardized": null_iso_slope,
            "iso_slope_ci_low": boot["iso_slope_ci_low"],
            "iso_slope_ci_high": boot["iso_slope_ci_high"],
            "null_iso_slope_ci_low": boot["null_iso_slope_ci_low"],
            "null_iso_slope_ci_high": boot["null_iso_slope_ci_high"],
            "deltaR2_iso_ci_low": boot["deltaR2_iso_ci_low"],
            "deltaR2_iso_ci_high": boot["deltaR2_iso_ci_high"],
            "deltaR2_coarse_ci_low": boot["deltaR2_coarse_ci_low"],
            "deltaR2_coarse_ci_high": boot["deltaR2_coarse_ci_high"],
            "deltaR2_entropy_ci_low": boot["deltaR2_entropy_ci_low"],
            "deltaR2_entropy_ci_high": boot["deltaR2_entropy_ci_high"],
            "deltaR2_all_ci_low": boot["deltaR2_all_ci_low"],
            "deltaR2_all_ci_high": boot["deltaR2_all_ci_high"],
            "deltaR2_null_iso_shuffle_ci_low": boot["deltaR2_null_iso_shuffle_ci_low"],
            "deltaR2_null_iso_shuffle_ci_high": boot["deltaR2_null_iso_shuffle_ci_high"],
        })

        null_rows.append({
            "scope": scope,
            "rule": rule,
            "L": L,
            "rho": rho,
            "target": target,
            "null_type": "iso_shuffle",
            "R2": s_null_iso["mean"],
            "R2_std": s_null_iso["std"],
            "deltaR2": delta_null_iso,
            "deltaR2_ci_low": boot["deltaR2_null_iso_shuffle_ci_low"],
            "deltaR2_ci_high": boot["deltaR2_null_iso_shuffle_ci_high"],
        })
        null_rows.append({
            "scope": scope,
            "rule": rule,
            "L": L,
            "rho": rho,
            "target": target,
            "null_type": "target_shuffle",
            "R2": s_null_target["mean"],
            "R2_std": s_null_target["std"],
            "deltaR2": s_null_target["mean"] - s_controls["mean"],
            "deltaR2_ci_low": np.nan,
            "deltaR2_ci_high": np.nan,
        })

    return rows, null_rows


def attach_tsi_and_verdict_fields(summary_df: pd.DataFrame) -> pd.DataFrame:
    out = summary_df.copy()
    out["TSI"] = np.nan
    out["TSI_norm"] = np.nan
    out["verdict_component"] = ""

    for key, idx in out.groupby(["scope", "rule", "L", "rho"], dropna=False).groups.items():
        sub = out.loc[idx].copy()
        fine = sub.loc[sub["target"] == "target_fine_net", "deltaR2_iso"]
        others = sub.loc[sub["target"].isin(NON_ALIAS_OTHER_TARGETS), "deltaR2_iso"]
        if fine.empty or others.empty:
            continue
        fine_val = float(fine.iloc[0])
        other_max = float(others.max())
        tsi = fine_val - other_max
        tsi_norm = fine_val / (1e-9 + other_max)
        out.loc[idx, "TSI"] = tsi
        out.loc[idx, "TSI_norm"] = tsi_norm
    return out


def determine_verdicts(summary_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = summary_df.copy()
    verdict_lines = [
        "ODD Selection-Principle Test",
        "Boundary/connectivity convention: periodic torus, 4-connected fine components (repo definition reused).",
        "Embedded-isolate definition: alive, 4-isolated, at least one diagonal live neighbor (repo definition reused).",
        "Note: in this repo the fine-net target equals future minus present periodic 4-component count, so `target_fine_net` and `target_delta_components` are numerically identical.",
        "Note: `TSI_norm` follows the requested literal formula and can be negative if every non-fine target has negative deltaR2_iso.",
        "",
    ]

    verdict_scopes = out[out["scope"].isin(["rule_pooled", "global_pooled"])].copy()
    for _, sub in verdict_scopes.groupby(["scope", "rule"], sort=False):
        fine = sub[sub["target"] == "target_fine_net"].iloc[0]
        others = sub[sub["target"].isin(NON_ALIAS_OTHER_TARGETS)].copy()
        coarse_beats_iso = False
        for target in AGGREGATE_TARGETS:
            trow = sub[sub["target"] == target]
            if trow.empty:
                continue
            row = trow.iloc[0]
            if max(float(row["deltaR2_coarse"]), float(row["deltaR2_entropy"])) > float(row["deltaR2_iso"]):
                coarse_beats_iso = True
                break

        crit1 = float(fine["deltaR2_iso"]) > 0 and float(fine["deltaR2_iso_ci_low"]) > 0
        crit2 = float(fine["iso_slope_standardized"]) < 0 and float(fine["iso_slope_ci_high"]) < 0
        better_than_all = True
        for _, other in others.iterrows():
            fine_d = float(fine["deltaR2_iso"])
            other_d = float(other["deltaR2_iso"])
            if fine_d <= other_d:
                better_than_all = False
                break
            if not ((fine_d >= 2.0 * other_d) or ((fine_d - other_d) >= 0.05)):
                better_than_all = False
                break
        crit3 = better_than_all

        null_gap_big = (
            (float(fine["deltaR2_iso"]) - float(fine["deltaR2_null_iso_shuffle"]) >= 0.05)
            or (float(fine["deltaR2_iso"]) >= 2.0 * max(float(fine["deltaR2_null_iso_shuffle"]), 1e-9))
            or (float(fine["deltaR2_null_iso_shuffle"]) <= 0.2 * float(fine["deltaR2_iso"]))
        )
        crit4 = null_gap_big
        crit5 = coarse_beats_iso

        real_slope_lo = float(fine["iso_slope_ci_low"])
        real_slope_hi = float(fine["iso_slope_ci_high"])
        null_slope_lo = float(fine["null_iso_slope_ci_low"])
        null_slope_hi = float(fine["null_iso_slope_ci_high"])
        null_slope_matches = (
            (null_slope_hi < 0)
            and (max(real_slope_lo, null_slope_lo) <= min(real_slope_hi, null_slope_hi))
        )
        null_similar = (
            (float(fine["deltaR2_null_iso_shuffle"]) >= 0.8 * float(fine["deltaR2_iso"]))
            or null_slope_matches
        )

        if null_similar:
            verdict = "HARD FAIL"
        elif crit1 and crit2 and crit3 and crit4 and crit5:
            verdict = "PASS"
        elif crit1 and crit2 and float(fine["TSI"]) > 0:
            verdict = "WEAK PASS"
        else:
            verdict = "FAIL"

        scope_mask = (out["scope"] == sub.iloc[0]["scope"]) & (out["rule"] == sub.iloc[0]["rule"])
        out.loc[scope_mask & (out["target"] == "target_fine_net"), "verdict_component"] = verdict

        scope_label = "Global pooled" if sub.iloc[0]["scope"] == "global_pooled" else f"{sub.iloc[0]['rule']} pooled"
        verdict_lines.extend([
            f"{scope_label}: {verdict}",
            f"  fine_net deltaR2_iso = {float(fine['deltaR2_iso']):.4f} "
            f"[{float(fine['deltaR2_iso_ci_low']):.4f}, {float(fine['deltaR2_iso_ci_high']):.4f}]",
            f"  fine_net iso slope = {float(fine['iso_slope_standardized']):.4f} "
            f"[{float(fine['iso_slope_ci_low']):.4f}, {float(fine['iso_slope_ci_high']):.4f}]",
            f"  fine_net null deltaR2 = {float(fine['deltaR2_null_iso_shuffle']):.4f}, "
            f"null slope CI = [{float(fine['null_iso_slope_ci_low']):.4f}, {float(fine['null_iso_slope_ci_high']):.4f}]",
            f"  TSI = {float(fine['TSI']):.4f}, TSI_norm = {float(fine['TSI_norm']):.4f}",
            f"  Criteria: c1={crit1}, c2={crit2}, c3={crit3}, c4={crit4}, c5={crit5}, hard_fail={null_similar}",
            "",
        ])

    return out, verdict_lines


def plot_delta_r2(summary_df: pd.DataFrame) -> None:
    plot_df = summary_df[summary_df["scope"] == "rule_pooled"].copy()
    rules = plot_df["rule"].dropna().unique().tolist()
    targets = TARGET_COLUMNS
    series = [
        ("iso", "deltaR2_iso", "deltaR2_iso_ci_low", "deltaR2_iso_ci_high"),
        ("coarse", "deltaR2_coarse", "deltaR2_coarse_ci_low", "deltaR2_coarse_ci_high"),
        ("entropy", "deltaR2_entropy", "deltaR2_entropy_ci_low", "deltaR2_entropy_ci_high"),
        ("null iso shuffled", "deltaR2_null_iso_shuffle", "deltaR2_null_iso_shuffle_ci_low", "deltaR2_null_iso_shuffle_ci_high"),
    ]

    fig, axes = plt.subplots(1, len(rules), figsize=(7 * max(len(rules), 1), 5), sharey=True)
    if len(rules) == 1:
        axes = [axes]

    width = 0.18
    x = np.arange(len(targets))
    for ax, rule in zip(axes, rules):
        sub = plot_df[plot_df["rule"] == rule].set_index("target").reindex(targets)
        for idx, (label, val_col, lo_col, hi_col) in enumerate(series):
            vals = sub[val_col].values.astype(float)
            lo = np.maximum(0.0, vals - sub[lo_col].values.astype(float))
            hi = np.maximum(0.0, sub[hi_col].values.astype(float) - vals)
            xpos = x + (idx - 1.5) * width
            ax.bar(xpos, vals, width=width, label=label)
            ax.errorbar(xpos, vals, yerr=np.vstack([lo, hi]), fmt="none", ecolor="black", elinewidth=0.8, capsize=2)
        ax.set_title(rule)
        ax.set_xticks(x)
        ax.set_xticklabels(targets, rotation=30, ha="right")
        ax.set_ylabel("delta R2")
        ax.axhline(0.0, color="black", linewidth=0.8)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_deltaR2_by_target.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_iso_slope(summary_df: pd.DataFrame) -> None:
    plot_df = summary_df[summary_df["scope"] == "rule_pooled"].copy()
    rules = plot_df["rule"].dropna().unique().tolist()
    targets = TARGET_COLUMNS

    fig, axes = plt.subplots(1, len(rules), figsize=(6 * max(len(rules), 1), 5), sharey=True)
    if len(rules) == 1:
        axes = [axes]

    x = np.arange(len(targets))
    for ax, rule in zip(axes, rules):
        sub = plot_df[plot_df["rule"] == rule].set_index("target").reindex(targets)
        vals = sub["iso_slope_standardized"].values.astype(float)
        lo = np.maximum(0.0, vals - sub["iso_slope_ci_low"].values.astype(float))
        hi = np.maximum(0.0, sub["iso_slope_ci_high"].values.astype(float) - vals)
        ax.bar(x, vals, width=0.65)
        ax.errorbar(x, vals, yerr=np.vstack([lo, hi]), fmt="none", ecolor="black", elinewidth=0.8, capsize=2)
        ax.set_title(rule)
        ax.set_xticks(x)
        ax.set_xticklabels(targets, rotation=30, ha="right")
        ax.set_ylabel("standardized iso coefficient")
        ax.axhline(0.0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_iso_slope_by_target.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def add_fine_net_residual(raw_df: pd.DataFrame) -> pd.DataFrame:
    out = raw_df.copy()
    X, _ = build_design_matrix(out, MODEL_COLUMNS["controls"], include_condition_effects=True)
    y = out["target_fine_net"].values.astype(float)
    model = LinearRegression().fit(X, y)
    out["fine_net_resid"] = y - model.predict(X)
    shuffled = out["fine_net_resid"].values.copy()
    np.random.default_rng(10_000 + len(out)).shuffle(shuffled)
    out["fine_net_resid_shuffled"] = shuffled
    return out


def bootstrap_residual_iso_ci(
    df: pd.DataFrame,
    target: str,
    columns: list[str],
    feature_name: str,
    include_condition_effects: bool,
    cfg: RunConfig,
    seed_offset: int,
) -> tuple[float, float]:
    vals = []
    iterator = tqdm(range(cfg.n_boot), desc=f"Residual bootstrap {feature_name}", leave=False)
    for boot_idx in iterator:
        bdf = stratified_bootstrap_df(df, cfg.seed + 50_000 + seed_offset * 1000 + boot_idx)
        vals.append(
            standardized_feature_slope(
                df=bdf,
                target=target,
                columns=columns,
                feature_name=feature_name,
                include_condition_effects=include_condition_effects,
            )
        )
    return percentile_ci(vals)


def residual_target_analysis(raw_df: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    df = add_fine_net_residual(raw_df)
    rows: list[dict[str, float]] = []
    scopes = [("global_residual_pooled", "ALL", df, True)]
    for rule in RULES:
        scopes.append((f"rule_residual_pooled", rule, df[df["rule"] == rule].reset_index(drop=True), True))

    model_specs = [
        ("iso_only", RESIDUAL_MODEL_COLUMNS["iso_only"], "iso_embedded", False),
        ("coarse_only", RESIDUAL_MODEL_COLUMNS["coarse_only"], None, False),
        ("entropy_only", RESIDUAL_MODEL_COLUMNS["entropy_only"], None, False),
        ("all_resid", RESIDUAL_MODEL_COLUMNS["all_resid"], "iso_embedded", False),
        ("null_iso_only", RESIDUAL_MODEL_COLUMNS["null_iso_only"], "iso_embedded_shuffled", True),
    ]

    for scope_idx, (scope, rule, sub, include_condition_effects) in enumerate(scopes):
        y = sub["fine_net_resid"].values.astype(float)
        y_null = sub["fine_net_resid_shuffled"].values.astype(float)
        for model_idx, (model_name, cols, slope_feature, use_null_target) in enumerate(model_specs):
            X, _ = build_design_matrix(sub, cols, include_condition_effects)
            scores = cv_r2_scores(X, y_null if use_null_target else y, cfg.seed, cfg.alphas)
            row = {
                "scope": scope,
                "rule": rule,
                "target": "fine_net_resid",
                "model": model_name,
                "R2": float(np.mean(scores)),
                "R2_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                "iso_slope_standardized": np.nan,
                "iso_slope_ci_low": np.nan,
                "iso_slope_ci_high": np.nan,
            }
            if slope_feature is not None:
                slope = standardized_feature_slope(
                    df=sub,
                    target="fine_net_resid",
                    columns=cols,
                    feature_name=slope_feature,
                    include_condition_effects=include_condition_effects,
                )
                ci_low, ci_high = bootstrap_residual_iso_ci(
                    df=sub,
                    target="fine_net_resid",
                    columns=cols,
                    feature_name=slope_feature,
                    include_condition_effects=include_condition_effects,
                    cfg=cfg,
                    seed_offset=scope_idx * 10 + model_idx,
                )
                row["iso_slope_standardized"] = slope
                row["iso_slope_ci_low"] = ci_low
                row["iso_slope_ci_high"] = ci_high
            rows.append(row)
    return pd.DataFrame(rows)


def pooled_scopes(df: pd.DataFrame, cfg: RunConfig) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    summary_rows: list[dict[str, float]] = []
    null_rows: list[dict[str, float]] = []

    seed_counter = 1000
    for rule in RULES:
        rule_df = df[df["rule"] == rule].reset_index(drop=True)
        rows, nulls = evaluate_scope(
            df=rule_df,
            scope="rule_pooled",
            rule=rule,
            L=np.nan,
            rho=np.nan,
            include_condition_effects=True,
            cfg=cfg,
            scope_seed_offset=seed_counter,
            include_delta_ci=True,
        )
        summary_rows.extend(rows)
        null_rows.extend(nulls)
        seed_counter += 1

    rows, nulls = evaluate_scope(
        df=df.reset_index(drop=True),
        scope="global_pooled",
        rule="ALL",
        L=np.nan,
        rho=np.nan,
        include_condition_effects=True,
        cfg=cfg,
        scope_seed_offset=seed_counter,
        include_delta_ci=True,
    )
    summary_rows.extend(rows)
    null_rows.extend(nulls)
    return summary_rows, null_rows


def condition_scopes(df: pd.DataFrame, cfg: RunConfig) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    summary_rows: list[dict[str, float]] = []
    null_rows: list[dict[str, float]] = []

    conds = df[["rule", "L", "rho", "condition_id"]].drop_duplicates().reset_index(drop=True)
    for cond_idx, row in tqdm(conds.iterrows(), total=len(conds), desc="Condition modeling"):
        sub = df[df["condition_id"] == row["condition_id"]].reset_index(drop=True)
        rows, nulls = evaluate_scope(
            df=sub,
            scope="condition",
            rule=str(row["rule"]),
            L=float(row["L"]),
            rho=float(row["rho"]),
            include_condition_effects=False,
            cfg=cfg,
            scope_seed_offset=cond_idx,
            include_delta_ci=False,
        )
        summary_rows.extend(rows)
        null_rows.extend(nulls)
    return summary_rows, null_rows


def write_verdict(verdict_lines: list[str]) -> None:
    verdict_path = os.path.join(OUT_DIR, "selection_verdict.txt")
    with open(verdict_path, "w", encoding="utf-8") as f:
        f.write("\n".join(verdict_lines).rstrip() + "\n")


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    ensure_dirs()
    raw_path = os.path.join(OUT_DIR, "selection_raw.csv")
    if args.reuse_raw:
        raw_df = pd.read_csv(raw_path)
        if "iso_embedded_shuffled" not in raw_df.columns:
            raw_df = add_condition_level_nulls(raw_df, cfg.seed)
            raw_df.to_csv(raw_path, index=False)
    else:
        rng = np.random.default_rng(cfg.seed)
        rows: list[dict[str, float]] = []
        sample_offset = 0

        total_conditions = len(RULES) * len(cfg.sizes) * len(cfg.densities)
        cond_bar = tqdm(total=total_conditions, desc="Simulation conditions")
        for rule_name, rule_cfg in RULES.items():
            for L in cfg.sizes:
                for rho in cfg.densities:
                    cond_rows = sample_condition(
                        rule_name=rule_name,
                        rule_cfg=rule_cfg,
                        L=L,
                        rho=rho,
                        n_samples=cfg.n_samples_per_condition,
                        k=cfg.k,
                        rng=rng,
                        sample_offset=sample_offset,
                    )
                    rows.extend(cond_rows)
                    sample_offset += cfg.n_samples_per_condition
                    cond_bar.update(1)
        cond_bar.close()

        raw_df = pd.DataFrame(rows)
        raw_df = add_condition_level_nulls(raw_df, cfg.seed)
        raw_df.to_csv(raw_path, index=False)

    cond_summary, cond_nulls = condition_scopes(raw_df, cfg)
    pooled_summary, pooled_nulls = pooled_scopes(raw_df, cfg)

    summary_df = pd.DataFrame(cond_summary + pooled_summary)
    summary_df = attach_tsi_and_verdict_fields(summary_df)
    summary_df, verdict_lines = determine_verdicts(summary_df)
    summary_df.to_csv(os.path.join(OUT_DIR, "selection_summary.csv"), index=False)

    nulls_df = pd.DataFrame(cond_nulls + pooled_nulls)
    nulls_df.to_csv(os.path.join(OUT_DIR, "selection_nulls.csv"), index=False)

    residual_df = residual_target_analysis(raw_df, cfg)
    residual_df.to_csv(os.path.join(OUT_DIR, "selection_residual_summary.csv"), index=False)

    plot_delta_r2(summary_df)
    plot_iso_slope(summary_df)
    write_verdict(verdict_lines)

    print("ODD Selection-Principle Test Complete")
    print("Main question:")
    print("Does iso_embedded specifically predict fine-scale component loss?")
    print("Key outputs:")
    print("- selection_summary.csv")
    print("- fig_deltaR2_by_target.png")
    print("- fig_iso_slope_by_target.png")
    print("- selection_verdict.txt")
    print("Interpretation:")
    print("PASS supports target-specific description privilege.")
    print("FAIL suggests iso_embedded may be a generic fragility proxy rather than a target-specific object.")


if __name__ == "__main__":
    main()
