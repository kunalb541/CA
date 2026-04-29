#!/usr/bin/env python3
"""
CA horizon-response test for ODD selection principle.

Question:
Does iso_embedded define a residual temporal response curve beta_iso(k)
for future fine-component change?

Definitions locked to current repo convention from selection test:
- periodic torus
- 4-connected fine components
- embedded isolate = alive, 4-isolated orthogonally, at least one diagonal live neighbor

Outputs:
outputs/selection_principle_horizon/horizon_raw.csv
outputs/selection_principle_horizon/horizon_condition_summary.csv
outputs/selection_principle_horizon/horizon_rule_summary.csv
outputs/selection_principle_horizon/horizon_global_summary.csv
outputs/selection_principle_horizon/fig_beta_iso_vs_horizon.png
outputs/selection_principle_horizon/fig_residR2_vs_horizon.png
outputs/selection_principle_horizon/horizon_verdict.txt
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import ndimage
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RULES = {
    "GoL": {"birth": {3}, "survive": {2, 3}},
    "HighLife": {"birth": {3, 6}, "survive": {2, 3}},
}

HORIZONS_DEFAULT = [1, 5, 10, 25, 50, 100, 200]


def step_ca(grid: np.ndarray, birth: set, survive: set) -> np.ndarray:
    """One toroidal Moore-neighborhood CA update."""
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
    """
    Count 4-connected live components on a torus.

    Uses scipy label on a 3x3 tiled grid, then merges labels touching through
    periodic copies. Simpler robust method for L=64/128.
    """
    L = grid.shape[0]
    if grid.sum() == 0:
        return 0

    # Label on original with 4-connectivity.
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

    # Merge top-bottom periodic neighbors.
    for j in range(L):
        union(labels[0, j], labels[L - 1, j])

    # Merge left-right periodic neighbors.
    for i in range(L):
        union(labels[i, 0], labels[i, L - 1])

    roots = {find(x) for x in range(1, nlab + 1)}
    return len(roots)


def iso_embedded_count(grid: np.ndarray) -> int:
    """
    Repo definition used in selection test:
    alive, 4-isolated orthogonally, at least one diagonal live neighbor.
    """
    up = np.roll(grid, 1, axis=0)
    down = np.roll(grid, -1, axis=0)
    left = np.roll(grid, 1, axis=1)
    right = np.roll(grid, -1, axis=1)

    orth_n = up.astype(int) + down.astype(int) + left.astype(int) + right.astype(int)

    diag1 = np.roll(np.roll(grid, 1, axis=0), 1, axis=1)
    diag2 = np.roll(np.roll(grid, 1, axis=0), -1, axis=1)
    diag3 = np.roll(np.roll(grid, -1, axis=0), 1, axis=1)
    diag4 = np.roll(np.roll(grid, -1, axis=0), -1, axis=1)
    diag_n = diag1.astype(int) + diag2.astype(int) + diag3.astype(int) + diag4.astype(int)

    iso = grid & (orth_n == 0) & (diag_n > 0)
    return int(iso.sum())


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


def simulate_dataset(args) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    rows = []
    horizons = sorted(args.horizons)
    max_h = max(horizons)

    total = len(RULES) * len(args.sizes) * len(args.densities) * args.n_samples

    pbar = tqdm(total=total, desc="Simulating CA horizon dataset")

    sample_id = 0
    for rule_name, rule in RULES.items():
        for L in args.sizes:
            block_size = 8 if L >= 64 else 4
            for rho in args.densities:
                condition_id = f"{rule_name}_L{L}_rho{rho:.2f}"

                for _ in range(args.n_samples):
                    grid = rng.random((L, L)) < rho

                    live0 = int(grid.sum())
                    density0 = live0 / (L * L)
                    iso0 = iso_embedded_count(grid)
                    block_var0, block_entropy0 = block_features(grid, block_size)
                    comp0 = count_components_4_periodic(grid)

                    # Record components at horizons.
                    comps = {}
                    g = grid.copy()
                    for t in range(1, max_h + 1):
                        g = step_ca(g, rule["birth"], rule["survive"])
                        if t in horizons:
                            comps[t] = count_components_4_periodic(g)

                    for h in horizons:
                        fine_net = comps[h] - comp0
                        rows.append({
                            "sample_id": sample_id,
                            "condition_id": condition_id,
                            "rule": rule_name,
                            "L": L,
                            "rho": rho,
                            "horizon": h,
                            "live_count": live0,
                            "density": density0,
                            "iso_embedded": iso0,
                            "block_var": block_var0,
                            "block_entropy": block_entropy0,
                            "components_t0": comp0,
                            "components_tk": comps[h],
                            "fine_net": fine_net,
                        })

                    sample_id += 1
                    pbar.update(1)

    pbar.close()
    return pd.DataFrame(rows)


def residual_slope_for_group(g: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """
    Within group, residualize fine_net against live_count+density,
    then fit residual ~ iso_embedded. Also shuffled iso null.
    """
    rng = np.random.default_rng(seed)
    n = len(g)

    X_base = g[["live_count", "density"]].to_numpy(float)
    y = g["fine_net"].to_numpy(float)
    iso = g[["iso_embedded"]].to_numpy(float)

    base = LinearRegression().fit(X_base, y)
    resid = y - base.predict(X_base)

    # Real standardized slope/R2
    iso_std = StandardScaler().fit_transform(iso)
    resid_std = StandardScaler().fit_transform(resid.reshape(-1, 1)).ravel()
    real = LinearRegression().fit(iso_std, resid_std)
    pred = real.predict(iso_std)
    r2 = float(r2_score(resid_std, pred))
    slope_std = float(real.coef_[0])

    # Raw slope
    raw = LinearRegression().fit(iso, resid)
    slope_raw = float(raw.coef_[0])
    intercept_raw = float(raw.intercept_)

    # Null with shuffled iso
    iso_shuf = iso.copy()
    rng.shuffle(iso_shuf[:, 0])
    iso_shuf_std = StandardScaler().fit_transform(iso_shuf)
    null = LinearRegression().fit(iso_shuf_std, resid_std)
    null_pred = null.predict(iso_shuf_std)
    null_r2 = float(r2_score(resid_std, null_pred))
    null_slope_std = float(null.coef_[0])
    null_raw = LinearRegression().fit(iso_shuf, resid)
    null_slope_raw = float(null_raw.coef_[0])

    boot_slope_raw = []
    boot_slope_std = []
    boot_r2 = []
    boot_null_r2 = []
    idx = np.arange(n)

    for _ in range(n_boot):
        bidx = rng.choice(idx, size=n, replace=True)
        gb = g.iloc[bidx]

        Xb = gb[["live_count", "density"]].to_numpy(float)
        yb = gb["fine_net"].to_numpy(float)
        isob = gb[["iso_embedded"]].to_numpy(float)

        if np.std(yb) == 0 or np.std(isob) == 0:
            continue

        base_b = LinearRegression().fit(Xb, yb)
        rb = yb - base_b.predict(Xb)

        if np.std(rb) == 0:
            continue

        isob_std = StandardScaler().fit_transform(isob)
        rb_std = StandardScaler().fit_transform(rb.reshape(-1, 1)).ravel()

        mb = LinearRegression().fit(isob_std, rb_std)
        pb = mb.predict(isob_std)
        boot_slope_std.append(float(mb.coef_[0]))
        boot_r2.append(float(r2_score(rb_std, pb)))

        mb_raw = LinearRegression().fit(isob, rb)
        boot_slope_raw.append(float(mb_raw.coef_[0]))

        isob_shuf = isob.copy()
        rng.shuffle(isob_shuf[:, 0])
        isob_shuf_std = StandardScaler().fit_transform(isob_shuf)
        nb = LinearRegression().fit(isob_shuf_std, rb_std)
        npred = nb.predict(isob_shuf_std)
        boot_null_r2.append(float(r2_score(rb_std, npred)))

    def ci(vals):
        vals = np.asarray(vals, dtype=float)
        if vals.size == 0:
            return np.nan, np.nan
        return tuple(np.percentile(vals, [2.5, 97.5]))

    slope_raw_lo, slope_raw_hi = ci(boot_slope_raw)
    slope_std_lo, slope_std_hi = ci(boot_slope_std)
    r2_lo, r2_hi = ci(boot_r2)
    null_r2_lo, null_r2_hi = ci(boot_null_r2)

    return {
        "n": n,
        "resid_R2_iso": r2,
        "resid_R2_iso_ci_low": r2_lo,
        "resid_R2_iso_ci_high": r2_hi,
        "resid_iso_slope_raw": slope_raw,
        "resid_iso_slope_raw_ci_low": slope_raw_lo,
        "resid_iso_slope_raw_ci_high": slope_raw_hi,
        "resid_iso_slope_std": slope_std,
        "resid_iso_slope_std_ci_low": slope_std_lo,
        "resid_iso_slope_std_ci_high": slope_std_hi,
        "resid_iso_intercept_raw": intercept_raw,
        "null_R2_shuffled_iso": null_r2,
        "null_R2_shuffled_iso_ci_low": null_r2_lo,
        "null_R2_shuffled_iso_ci_high": null_r2_hi,
        "null_slope_std": null_slope_std,
        "null_slope_raw": null_slope_raw,
    }


def analyze(df: pd.DataFrame, args, outdir: Path):
    condition_rows = []

    group_cols = ["condition_id", "rule", "L", "rho", "horizon"]
    for keys, g in tqdm(df.groupby(group_cols), desc="Condition horizon fits"):
        condition_id, rule, L, rho, horizon = keys
        res = residual_slope_for_group(g, args.n_boot, args.seed + int(horizon) + int(L))
        res.update({
            "scope": "condition",
            "condition_id": condition_id,
            "rule": rule,
            "L": L,
            "rho": rho,
            "horizon": horizon,
        })
        condition_rows.append(res)

    cond = pd.DataFrame(condition_rows)
    cond.to_csv(outdir / "horizon_condition_summary.csv", index=False)

    # Rule/horizon summary from condition-level results
    rule_rows = []
    for (rule, horizon), g in cond.groupby(["rule", "horizon"]):
        slopes = g["resid_iso_slope_raw"].to_numpy(float)
        std_slopes = g["resid_iso_slope_std"].to_numpy(float)
        r2s = g["resid_R2_iso"].to_numpy(float)
        nulls = g["null_R2_shuffled_iso"].to_numpy(float)

        mean_raw = float(np.mean(slopes))
        sd_raw = float(np.std(slopes, ddof=1))
        cv_raw = float(sd_raw / (abs(mean_raw) + 1e-12))

        mean_std = float(np.mean(std_slopes))
        sd_std = float(np.std(std_slopes, ddof=1))
        cv_std = float(sd_std / (abs(mean_std) + 1e-12))

        rule_rows.append({
            "scope": "rule_summary",
            "rule": rule,
            "horizon": horizon,
            "n_conditions": len(g),
            "mean_raw_slope": mean_raw,
            "sd_raw_slope": sd_raw,
            "cv_raw_slope": cv_raw,
            "mean_std_slope": mean_std,
            "sd_std_slope": sd_std,
            "cv_std_slope": cv_std,
            "mean_resid_R2": float(np.mean(r2s)),
            "min_resid_R2": float(np.min(r2s)),
            "max_resid_R2": float(np.max(r2s)),
            "mean_null_R2": float(np.mean(nulls)),
            "all_raw_slopes_negative": bool(np.all(slopes < 0)),
            "frac_slope_ci_negative": float(np.mean(g["resid_iso_slope_raw_ci_high"].to_numpy(float) < 0)),
        })

    rule_summary = pd.DataFrame(rule_rows)
    rule_summary.to_csv(outdir / "horizon_rule_summary.csv", index=False)

    # Global pooled condition-level summary by horizon
    global_rows = []
    for horizon, g in cond.groupby("horizon"):
        slopes = g["resid_iso_slope_raw"].to_numpy(float)
        std_slopes = g["resid_iso_slope_std"].to_numpy(float)
        r2s = g["resid_R2_iso"].to_numpy(float)
        nulls = g["null_R2_shuffled_iso"].to_numpy(float)

        mean_raw = float(np.mean(slopes))
        sd_raw = float(np.std(slopes, ddof=1))
        cv_raw = float(sd_raw / (abs(mean_raw) + 1e-12))

        mean_std = float(np.mean(std_slopes))
        sd_std = float(np.std(std_slopes, ddof=1))
        cv_std = float(sd_std / (abs(mean_std) + 1e-12))

        global_rows.append({
            "scope": "global_summary",
            "horizon": horizon,
            "n_conditions": len(g),
            "mean_raw_slope": mean_raw,
            "sd_raw_slope": sd_raw,
            "cv_raw_slope": cv_raw,
            "mean_std_slope": mean_std,
            "sd_std_slope": sd_std,
            "cv_std_slope": cv_std,
            "mean_resid_R2": float(np.mean(r2s)),
            "min_resid_R2": float(np.min(r2s)),
            "max_resid_R2": float(np.max(r2s)),
            "mean_null_R2": float(np.mean(nulls)),
            "all_raw_slopes_negative": bool(np.all(slopes < 0)),
            "frac_slope_ci_negative": float(np.mean(g["resid_iso_slope_raw_ci_high"].to_numpy(float) < 0)),
        })

    global_summary = pd.DataFrame(global_rows)
    global_summary.to_csv(outdir / "horizon_global_summary.csv", index=False)

    return cond, rule_summary, global_summary


def make_figures(rule_summary: pd.DataFrame, global_summary: pd.DataFrame, outdir: Path):
    # Figure beta vs horizon
    plt.figure(figsize=(8, 5))
    for rule, g in rule_summary.groupby("rule"):
        g = g.sort_values("horizon")
        plt.plot(g["horizon"], g["mean_raw_slope"], marker="o", label=rule)
        plt.fill_between(
            g["horizon"],
            g["mean_raw_slope"] - g["sd_raw_slope"],
            g["mean_raw_slope"] + g["sd_raw_slope"],
            alpha=0.2,
        )
    gg = global_summary.sort_values("horizon")
    plt.plot(gg["horizon"], gg["mean_raw_slope"], marker="s", linestyle="--", label="Global")
    plt.axhline(0, linewidth=1)
    plt.xscale("log")
    plt.xlabel("Horizon k")
    plt.ylabel("Mean raw residual iso slope")
    plt.title("Embedded-isolate residual response curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "fig_beta_iso_vs_horizon.png", dpi=200)
    plt.close()

    # Figure R2 vs horizon
    plt.figure(figsize=(8, 5))
    for rule, g in rule_summary.groupby("rule"):
        g = g.sort_values("horizon")
        plt.plot(g["horizon"], g["mean_resid_R2"], marker="o", label=rule)
        plt.fill_between(
            g["horizon"],
            g["min_resid_R2"],
            g["max_resid_R2"],
            alpha=0.15,
        )
    gg = global_summary.sort_values("horizon")
    plt.plot(gg["horizon"], gg["mean_resid_R2"], marker="s", linestyle="--", label="Global")
    plt.axhline(0, linewidth=1)
    plt.xscale("log")
    plt.xlabel("Horizon k")
    plt.ylabel("Mean residual R² from iso_embedded")
    plt.title("Residual explanatory power vs horizon")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "fig_residR2_vs_horizon.png", dpi=200)
    plt.close()


def write_verdict(rule_summary: pd.DataFrame, global_summary: pd.DataFrame, args, outdir: Path):
    lines = []
    lines.append("CA Horizon Response Test")
    lines.append("Boundary/connectivity: periodic torus, 4-connected fine components.")
    lines.append("Embedded isolate: alive, 4-isolated orthogonally, at least one diagonal live neighbor.")
    lines.append(f"Horizons: {args.horizons}")
    lines.append("")

    def verdict_for(g):
        frac_neg = float(np.mean(g["all_raw_slopes_negative"]))
        frac_ci_neg = float(np.mean(g["frac_slope_ci_negative"] >= 0.75))
        mean_r2 = float(np.mean(g["mean_resid_R2"]))
        mean_null = float(np.mean(g["mean_null_R2"]))
        if frac_neg == 1.0 and frac_ci_neg >= 0.75 and mean_r2 > 0.05 and mean_null < 0.01:
            return "PASS"
        if frac_neg >= 0.75 and mean_r2 > 0.02 and mean_null < 0.01:
            return "WEAK PASS"
        return "FAIL"

    for rule, g in rule_summary.groupby("rule"):
        g = g.sort_values("horizon")
        verdict = verdict_for(g)
        lines.append(f"{rule}: {verdict}")
        lines.append(f"  mean residual R2 over horizons = {g['mean_resid_R2'].mean():.4f}")
        lines.append(f"  mean null R2 over horizons = {g['mean_null_R2'].mean():.4f}")
        lines.append(f"  all horizons have all condition slopes negative? {bool(np.all(g['all_raw_slopes_negative']))}")
        lines.append("  horizon table:")
        for _, row in g.iterrows():
            lines.append(
                f"    k={int(row['horizon']):3d}: "
                f"mean_slope={row['mean_raw_slope']:.4f}, "
                f"CV={row['cv_raw_slope']:.3f}, "
                f"mean_R2={row['mean_resid_R2']:.4f}, "
                f"null_R2={row['mean_null_R2']:.4f}, "
                f"frac_CI_neg={row['frac_slope_ci_negative']:.2f}"
            )
        lines.append("")

    gg = global_summary.sort_values("horizon")
    lines.append(f"Global: {verdict_for(gg)}")
    lines.append(f"  mean residual R2 over horizons = {gg['mean_resid_R2'].mean():.4f}")
    lines.append(f"  mean null R2 over horizons = {gg['mean_null_R2'].mean():.4f}")
    lines.append(f"  all horizons have all condition slopes negative? {bool(np.all(gg['all_raw_slopes_negative']))}")
    lines.append("")
    lines.append("Interpretation lock:")
    lines.append("PASS supports a temporal residual response curve beta_iso(k).")
    lines.append("FAIL means the k=100 residual law may be horizon-specific or unstable.")

    (outdir / "horizon_verdict.txt").write_text("\n".join(lines))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--n-boot", type=int, default=300)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--sizes", type=int, nargs="+", default=[64, 128])
    p.add_argument("--densities", type=float, nargs="+", default=[0.20, 0.25, 0.30, 0.35])
    p.add_argument("--horizons", type=int, nargs="+", default=HORIZONS_DEFAULT)
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

    outdir = ROOT / "outputs" / "selection_principle_horizon"
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / "horizon_raw.csv"

    if args.reuse_raw and raw_path.exists():
        print(f"Reusing raw file: {raw_path}")
        df = pd.read_csv(raw_path)
    else:
        df = simulate_dataset(args)
        df.to_csv(raw_path, index=False)
        print(f"Saved raw dataset: {raw_path}")

    cond, rule_summary, global_summary = analyze(df, args, outdir)
    make_figures(rule_summary, global_summary, outdir)
    write_verdict(rule_summary, global_summary, args, outdir)

    print("\nCA horizon response test complete.")
    print("Outputs:")
    for name in [
        "horizon_raw.csv",
        "horizon_condition_summary.csv",
        "horizon_rule_summary.csv",
        "horizon_global_summary.csv",
        "fig_beta_iso_vs_horizon.png",
        "fig_residR2_vs_horizon.png",
        "horizon_verdict.txt",
    ]:
        print(" ", outdir / name)


if __name__ == "__main__":
    main()
