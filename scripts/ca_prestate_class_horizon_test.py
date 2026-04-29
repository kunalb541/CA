#!/usr/bin/env python3
"""
CA non-leaky prestate class horizon test.

Registered attack:
Do transition/fate mechanism results rely on outcome-derived local-window/fate features?

This test uses only t=0 information:
- iso_count
- diagonal embedded-isolate class counts

No t+1 fate variables.
No local-window loss.
No target-derived features.

Question:
Do initial isolate classes refine the iso_count horizon response without leakage?

Outputs:
outputs/prestate_class_horizon/
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import ndimage
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs" / "prestate_class_horizon"
OUT.mkdir(parents=True, exist_ok=True)

RULES = {
    "GoL": {"birth": {3}, "survive": {2, 3}},
    "HighLife": {"birth": {3, 6}, "survive": {2, 3}},
}
HORIZONS = [1, 5, 10, 25, 50, 100, 200]


def step_ca(grid, birth, survive):
    n = np.zeros_like(grid, dtype=np.int16)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            n += np.roll(np.roll(grid, dx, axis=0), dy, axis=1)
    return ((~grid) & np.isin(n, list(birth))) | (grid & np.isin(n, list(survive)))


def count_components_4_periodic(grid):
    L = grid.shape[0]
    if grid.sum() == 0:
        return 0
    struct = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=int)
    labels, nlab = ndimage.label(grid, structure=struct)
    if nlab <= 1:
        return int(nlab)

    parent = np.arange(nlab + 1)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a,b):
        if a == 0 or b == 0:
            return
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for j in range(L):
        union(labels[0,j], labels[L-1,j])
    for i in range(L):
        union(labels[i,0], labels[i,L-1])

    return len({find(x) for x in range(1, nlab+1)})


def neighbor_counts(grid):
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
    return orth, diag


def embedded_isolate_mask(grid):
    orth, diag = neighbor_counts(grid)
    return grid & (orth == 0) & (diag > 0)


def diag_class_counts(grid):
    """
    Counts embedded isolates by diagonal-neighbor class at t=0.
    Classes 1..15. Orthogonal neighbors are zero by definition.
    """
    L = grid.shape[0]
    mask = embedded_isolate_mask(grid)
    coords = np.argwhere(mask)

    counts = {f"class_{c:02d}_count": 0 for c in range(1, 16)}

    for i, j in coords:
        i = int(i); j = int(j)
        diag_coords = [
            ((i-1)%L, (j-1)%L), # NW bit 0
            ((i-1)%L, (j+1)%L), # NE bit 1
            ((i+1)%L, (j-1)%L), # SW bit 2
            ((i+1)%L, (j+1)%L), # SE bit 3
        ]
        code = 0
        for k, (a,b) in enumerate(diag_coords):
            code |= int(grid[a,b]) << k
        if code > 0:
            counts[f"class_{code:02d}_count"] += 1

    counts["iso_count"] = int(len(coords))
    return counts


def simulate(args):
    rng = np.random.default_rng(args.seed)
    rows = []
    total = len(RULES) * len(args.sizes) * len(args.densities) * args.n_samples
    pbar = tqdm(total=total, desc="Simulating prestate class horizon data")
    sample_id = 0

    for rule_name, rule in RULES.items():
        for L in args.sizes:
            for rho in args.densities:
                condition_id = f"{rule_name}_L{L}_rho{rho:.2f}"

                for _ in range(args.n_samples):
                    g0 = rng.random((L,L)) < rho
                    comp0 = count_components_4_periodic(g0)
                    live_count = int(g0.sum())
                    density = live_count / (L*L)
                    class_counts = diag_class_counts(g0)

                    comps = {}
                    g = g0.copy()
                    for t in range(1, max(args.horizons)+1):
                        g = step_ca(g, rule["birth"], rule["survive"])
                        if t in args.horizons:
                            comps[t] = count_components_4_periodic(g)

                    for h in args.horizons:
                        row = {
                            "sample_id": sample_id,
                            "condition_id": condition_id,
                            "rule": rule_name,
                            "L": L,
                            "rho": rho,
                            "horizon": h,
                            "live_count": live_count,
                            "density": density,
                            "components_t0": comp0,
                            "components_tk": comps[h],
                            "fine_net": comps[h] - comp0,
                        }
                        row.update(class_counts)
                        rows.append(row)

                    sample_id += 1
                    pbar.update(1)

    pbar.close()
    return pd.DataFrame(rows)


def residualize(g):
    X = g[["live_count", "density"]].to_numpy(float)
    y = g["fine_net"].to_numpy(float)
    m = LinearRegression().fit(X, y)
    return y - m.predict(X)


def fit_model(g, features, shuffled=False, seed=0):
    rng = np.random.default_rng(seed)
    y = residualize(g)
    X = g[features].to_numpy(float)

    if shuffled:
        # shuffle rows of X jointly, preserving feature covariance
        idx = np.arange(len(g))
        rng.shuffle(idx)
        X = X[idx]

    if np.std(y) == 0 or X.shape[1] == 0 or np.all(np.std(X, axis=0) == 0):
        return np.nan, np.nan

    yz = StandardScaler().fit_transform(y.reshape(-1,1)).ravel()
    Xz = X.copy().astype(float)
    for j in range(Xz.shape[1]):
        sd = Xz[:,j].std()
        if sd == 0:
            Xz[:,j] = 0
        else:
            Xz[:,j] = (Xz[:,j] - Xz[:,j].mean()) / sd

    model = RidgeCV(alphas=np.logspace(-6, 3, 20))
    model.fit(Xz, yz)
    pred = model.predict(Xz)
    r2 = float(r2_score(yz, pred))
    corr = float(np.corrcoef(pred, yz)[0,1]) if np.std(pred) > 0 else np.nan
    return r2, corr


def analyze(df):
    class_cols = sorted([c for c in df.columns if c.startswith("class_") and c.endswith("_count")])

    specs = {
        "iso_count": ["iso_count"],
        "class_counts": class_cols,
        "iso_plus_class_counts": ["iso_count"] + class_cols,
    }

    rows = []
    for keys, g in tqdm(df.groupby(["condition_id", "rule", "L", "rho", "horizon"]), desc="Fitting prestate class models"):
        condition_id, rule, L, rho, h = keys
        for model, feats in specs.items():
            r2, corr = fit_model(g, feats, shuffled=False, seed=int(h)+int(L))
            nr2, ncorr = fit_model(g, feats, shuffled=True, seed=999+int(h)+int(L))
            rows.append({
                "scope": "condition",
                "condition_id": condition_id,
                "rule": rule,
                "L": L,
                "rho": rho,
                "horizon": h,
                "model": model,
                "n": len(g),
                "R2": r2,
                "corr": corr,
                "shuffle_R2": nr2,
                "shuffle_corr": ncorr,
                "features": ",".join(feats),
            })

    cond = pd.DataFrame(rows)

    # aggregate by rule/horizon/model
    agg_rows = []
    for (rule, h, model), g in cond.groupby(["rule", "horizon", "model"]):
        agg_rows.append({
            "scope": "rule_horizon",
            "rule": rule,
            "horizon": h,
            "model": model,
            "n_conditions": len(g),
            "mean_R2": g["R2"].mean(),
            "min_R2": g["R2"].min(),
            "max_R2": g["R2"].max(),
            "mean_corr": g["corr"].mean(),
            "mean_shuffle_R2": g["shuffle_R2"].mean(),
            "frac_R2_positive": (g["R2"] > 0).mean(),
        })

    # global by horizon/model
    for (h, model), g in cond.groupby(["horizon", "model"]):
        agg_rows.append({
            "scope": "global_horizon",
            "rule": "ALL",
            "horizon": h,
            "model": model,
            "n_conditions": len(g),
            "mean_R2": g["R2"].mean(),
            "min_R2": g["R2"].min(),
            "max_R2": g["R2"].max(),
            "mean_corr": g["corr"].mean(),
            "mean_shuffle_R2": g["shuffle_R2"].mean(),
            "frac_R2_positive": (g["R2"] > 0).mean(),
        })

    summary = pd.DataFrame(agg_rows)
    return cond, summary


def make_figures(summary):
    g = summary[summary["scope"].eq("global_horizon")].copy()
    plt.figure(figsize=(8,5))
    for model, sub in g.groupby("model"):
        sub = sub.sort_values("horizon")
        plt.plot(sub["horizon"], sub["mean_R2"], marker="o", label=model)
    plt.xscale("log")
    plt.axhline(0, linewidth=1)
    plt.xlabel("Horizon k")
    plt.ylabel("Mean condition R²")
    plt.title("Non-leaky prestate class horizon response")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "fig_prestate_class_R2_vs_horizon.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8,5))
    for model, sub in g.groupby("model"):
        sub = sub.sort_values("horizon")
        plt.plot(sub["horizon"], sub["mean_shuffle_R2"], marker="o", label=model)
    plt.xscale("log")
    plt.axhline(0, linewidth=1)
    plt.xlabel("Horizon k")
    plt.ylabel("Mean shuffled R²")
    plt.title("Prestate class shuffled null")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "fig_prestate_class_shuffle_R2.png", dpi=200)
    plt.close()


def write_verdict(summary):
    g = summary[summary["scope"].eq("global_horizon")]
    lines = []
    lines.append("CA Non-Leaky Prestate Class Horizon Test")
    lines.append("")
    lines.append("Question: do t=0 isolate class counts explain horizon residual response without using t+1 fate/window-loss features?")
    lines.append("")
    for model in ["iso_count", "class_counts", "iso_plus_class_counts"]:
        sub = g[g["model"].eq(model)].sort_values("horizon")
        lines.append(f"{model}:")
        lines.append(f"  mean R2 over horizons = {sub['mean_R2'].mean():.4f}")
        lines.append(f"  mean shuffle R2 over horizons = {sub['mean_shuffle_R2'].mean():.4f}")
        lines.append(f"  min R2 over horizons = {sub['mean_R2'].min():.4f}")
        for _, row in sub.iterrows():
            lines.append(
                f"    k={int(row['horizon']):3d}: "
                f"R2={row['mean_R2']:.4f}, "
                f"shuffle={row['mean_shuffle_R2']:.4f}, "
                f"frac_pos={row['frac_R2_positive']:.2f}"
            )
        lines.append("")

    iso = g[g["model"].eq("iso_count")]["mean_R2"].mean()
    cls = g[g["model"].eq("class_counts")]["mean_R2"].mean()
    shuf = g[g["model"].eq("class_counts")]["mean_shuffle_R2"].mean()

    if cls > iso + 0.02 and shuf < 0.01:
        lines.append("VERDICT: PASS")
        lines.append("Initial isolate class counts refine iso_count without outcome leakage.")
    elif iso > 0.05 and shuf < 0.01:
        lines.append("VERDICT: WEAK PASS")
        lines.append("Initial iso_count is the robust non-leaky object summary; class counts add little.")
    else:
        lines.append("VERDICT: CHECK/FAIL")
        lines.append("Non-leaky prestate class predictors do not cleanly recover the response.")

    lines.append("")
    lines.append("Interpretation:")
    lines.append("- PASS: local t=0 class structure refines the mechanism.")
    lines.append("- WEAK PASS: object count is the sufficient prestate summary; detailed mechanism uses t+1 event carriers.")
    lines.append("- FAIL: previous mechanism relied too much on outcome-derived features.")

    (OUT / "prestate_class_verdict.txt").write_text("\n".join(lines))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--sizes", type=int, nargs="+", default=[64,128])
    p.add_argument("--densities", type=float, nargs="+", default=[0.20,0.25,0.30,0.35])
    p.add_argument("--horizons", type=int, nargs="+", default=HORIZONS)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--reuse-raw", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.sizes = [64]
        args.densities = [0.25,0.30]
        args.n_samples = min(args.n_samples, 300)

    raw_path = OUT / "prestate_class_horizon_raw.csv"

    if args.reuse_raw and raw_path.exists():
        print("Reusing:", raw_path)
        df = pd.read_csv(raw_path)
    else:
        df = simulate(args)
        df.to_csv(raw_path, index=False)
        print("Saved:", raw_path)

    cond, summary = analyze(df)
    cond.to_csv(OUT / "prestate_class_condition_summary.csv", index=False)
    summary.to_csv(OUT / "prestate_class_summary.csv", index=False)

    make_figures(summary)
    write_verdict(summary)

    print((OUT / "prestate_class_verdict.txt").read_text())
    print("\nSaved outputs to:", OUT)


if __name__ == "__main__":
    main()
