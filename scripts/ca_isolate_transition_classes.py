#!/usr/bin/env python3
"""
CA isolate transition-class decomposition.

Question:
Which local embedded-isolate context classes carry the one-step residual fine-component response?

Uses the already generated isolate-fate raw data if possible? No: we need local pattern classes,
so we rerun one-step simulations and record per-isolate class events.

Definitions:
- periodic torus
- 4-connected components
- embedded isolate = live, orthogonally 4-isolated, at least one diagonal live neighbor

Outputs:
outputs/isolate_transition_classes/
"""

import sys
from pathlib import Path
import argparse
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

OUT = ROOT / "outputs" / "isolate_transition_classes"
OUT.mkdir(parents=True, exist_ok=True)


def step_ca(grid, birth, survive):
    n = np.zeros_like(grid, dtype=np.int16)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            n += np.roll(np.roll(grid, dx, axis=0), dy, axis=1)
    return ((~grid) & np.isin(n, list(birth))) | (grid & np.isin(n, list(survive)))


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
    return orth, diag, orth + diag


def embedded_isolate_mask(grid):
    orth, diag, _ = neighbor_counts(grid)
    return grid & (orth == 0) & (diag > 0)


def count_components_4(grid):
    if grid.sum() == 0:
        return 0
    struct = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=int)
    _, nlab = ndimage.label(grid, structure=struct)
    return int(nlab)


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


def window(grid, i, j, radius):
    L = grid.shape[0]
    ii = [(i + di) % L for di in range(-radius, radius+1)]
    jj = [(j + dj) % L for dj in range(-radius, radius+1)]
    return grid[np.ix_(ii, jj)]


def pattern_code_3x3(grid, i, j):
    """
    Encode 3x3 neighborhood as integer bitmask in row-major order.
    Center included. For embedded isolates center is always 1.
    """
    w = window(grid, i, j, 1).astype(np.uint8).ravel()
    code = 0
    for k, bit in enumerate(w):
        code |= int(bit) << k
    return int(code)


def diag_mask_class(grid, i, j):
    """
    For embedded isolate, orthogonal cells are zero.
    Class determined by diagonal occupancy mask, 4 bits.
    order: NW, NE, SW, SE
    """
    L = grid.shape[0]
    coords = [
        ((i-1)%L, (j-1)%L),
        ((i-1)%L, (j+1)%L),
        ((i+1)%L, (j-1)%L),
        ((i+1)%L, (j+1)%L),
    ]
    code = 0
    for k, (a,b) in enumerate(coords):
        code |= int(grid[a,b]) << k
    return int(code)


def local_component_delta(grid0, grid1, i, j, radius=2):
    w0 = window(grid0, i, j, radius)
    w1 = window(grid1, i, j, radius)
    c0 = count_components_4(w0)
    c1 = count_components_4(w1)
    return int(c1 - c0), int(c0), int(c1)


def simulate(args):
    rng = np.random.default_rng(args.seed)
    sample_rows = []
    isolate_rows = []

    total = len(RULES) * len(args.sizes) * len(args.densities) * args.n_samples
    pbar = tqdm(total=total, desc="Simulating transition classes")
    sample_id = 0

    for rule_name, rule in RULES.items():
        for L in args.sizes:
            for rho in args.densities:
                condition_id = f"{rule_name}_L{L}_rho{rho:.2f}"

                for _ in range(args.n_samples):
                    g0 = rng.random((L, L)) < rho
                    g1 = step_ca(g0, rule["birth"], rule["survive"])

                    comp0 = count_components_4_periodic(g0)
                    comp1 = count_components_4_periodic(g1)
                    fine_net_1 = comp1 - comp0

                    live_count = int(g0.sum())
                    density = live_count / (L*L)

                    iso_mask = embedded_isolate_mask(g0)
                    coords = np.argwhere(iso_mask)
                    iso_count = int(len(coords))

                    sample_rows.append({
                        "sample_id": sample_id,
                        "condition_id": condition_id,
                        "rule": rule_name,
                        "L": L,
                        "rho": rho,
                        "live_count": live_count,
                        "density": density,
                        "components_t0": comp0,
                        "components_t1": comp1,
                        "fine_net_1": fine_net_1,
                        "iso_count": iso_count,
                    })

                    births = (~g0) & g1
                    orth1, diag1, moore1 = neighbor_counts(g1)

                    for i, j in coords:
                        i = int(i); j = int(j)
                        diag_class = diag_mask_class(g0, i, j)
                        code3 = pattern_code_3x3(g0, i, j)
                        dlocal, c0loc, c1loc = local_component_delta(g0, g1, i, j, radius=args.local_radius)

                        alive1 = bool(g1[i,j])
                        died = int(not alive1)
                        survived = int(alive1)
                        connected1 = int(alive1 and orth1[i,j] > 0)
                        isolated1 = int(alive1 and orth1[i,j] == 0)

                        Lg = g0.shape[0]
                        orth_neighbors = [
                            ((i-1)%Lg,j), ((i+1)%Lg,j), (i,(j-1)%Lg), (i,(j+1)%Lg)
                        ]
                        diag_neighbors = [
                            ((i-1)%Lg,(j-1)%Lg), ((i-1)%Lg,(j+1)%Lg),
                            ((i+1)%Lg,(j-1)%Lg), ((i+1)%Lg,(j+1)%Lg)
                        ]
                        orth_births = int(sum(births[a,b] for a,b in orth_neighbors))
                        diag_births = int(sum(births[a,b] for a,b in diag_neighbors))

                        isolate_rows.append({
                            "sample_id": sample_id,
                            "condition_id": condition_id,
                            "rule": rule_name,
                            "L": L,
                            "rho": rho,
                            "diag_class": diag_class,
                            "pattern3_code": code3,
                            "local_delta": dlocal,
                            "local_loss": max(0, -dlocal),
                            "local_gain": max(0, dlocal),
                            "local_components_t0": c0loc,
                            "local_components_t1": c1loc,
                            "died": died,
                            "survived": survived,
                            "survived_connected": connected1,
                            "survived_isolated": isolated1,
                            "orth_births": orth_births,
                            "diag_births": diag_births,
                        })

                    sample_id += 1
                    pbar.update(1)

    pbar.close()
    return pd.DataFrame(sample_rows), pd.DataFrame(isolate_rows)


def residualize_sample_target(samples):
    rows = []
    for condition_id, g in samples.groupby("condition_id"):
        X = g[["live_count", "density"]].to_numpy(float)
        y = g["fine_net_1"].to_numpy(float)
        m = LinearRegression().fit(X, y)
        resid = y - m.predict(X)
        tmp = g[["sample_id","condition_id","rule","L","rho","fine_net_1","iso_count"]].copy()
        tmp["fine_net_resid"] = resid
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


def class_decomposition(samples, isolates):
    """
    Aggregate isolate classes per sample, then test how class counts explain residual ΔC1.
    """
    resids = residualize_sample_target(samples)

    # Sample x class count table for diagonal classes
    counts = isolates.pivot_table(
        index="sample_id",
        columns="diag_class",
        values="local_delta",
        aggfunc="size",
        fill_value=0
    )
    counts.columns = [f"class_{int(c):02d}_count" for c in counts.columns]
    counts = counts.reset_index()

    # Also local loss sum per class
    losses = isolates.pivot_table(
        index="sample_id",
        columns="diag_class",
        values="local_loss",
        aggfunc="sum",
        fill_value=0
    )
    losses.columns = [f"class_{int(c):02d}_loss" for c in losses.columns]
    losses = losses.reset_index()

    # Mean local delta per class summary
    class_summary = isolates.groupby(["rule","diag_class"]).agg(
        n_isolates=("diag_class","size"),
        mean_local_delta=("local_delta","mean"),
        mean_local_loss=("local_loss","mean"),
        mean_local_gain=("local_gain","mean"),
        death_frac=("died","mean"),
        survive_connected_frac=("survived_connected","mean"),
        orth_births_mean=("orth_births","mean"),
        diag_births_mean=("diag_births","mean"),
    ).reset_index()

    global_class = isolates.groupby("diag_class").agg(
        n_isolates=("diag_class","size"),
        mean_local_delta=("local_delta","mean"),
        mean_local_loss=("local_loss","mean"),
        mean_local_gain=("local_gain","mean"),
        death_frac=("died","mean"),
        survive_connected_frac=("survived_connected","mean"),
        orth_births_mean=("orth_births","mean"),
        diag_births_mean=("diag_births","mean"),
    ).reset_index()
    global_class["rule"] = "ALL"
    class_summary = pd.concat([class_summary, global_class], ignore_index=True)

    data = resids.merge(counts, on="sample_id", how="left").merge(losses, on="sample_id", how="left")
    data = data.fillna(0)

    count_cols = [c for c in data.columns if c.startswith("class_") and c.endswith("_count")]
    loss_cols = [c for c in data.columns if c.startswith("class_") and c.endswith("_loss")]

    model_specs = {
        "iso_count": ["iso_count"],
        "diag_class_counts": count_cols,
        "diag_class_losses": loss_cols,
        "counts_plus_losses": count_cols + loss_cols,
    }

    model_rows = []
    for scope_name, group_keys in [("global", []), ("rule", ["rule"]), ("condition", ["condition_id"])]:
        if group_keys:
            groups = data.groupby(group_keys)
        else:
            groups = [("ALL", data)]

        for key, g in groups:
            y = g["fine_net_resid"].to_numpy(float)
            if np.std(y) == 0:
                continue
            y_std = StandardScaler().fit_transform(y.reshape(-1,1)).ravel()

            for model_name, cols in model_specs.items():
                X = g[cols].to_numpy(float)
                if X.shape[1] == 0 or np.all(np.std(X, axis=0) == 0):
                    r2 = np.nan
                else:
                    X_std = StandardScaler().fit_transform(X)
                    m = LinearRegression().fit(X_std, y_std)
                    pred = m.predict(X_std)
                    r2 = r2_score(y_std, pred)

                row = {
                    "scope": scope_name,
                    "key": key if isinstance(key, str) else "|".join(map(str,key)),
                    "model": model_name,
                    "R2_in_sample": r2,
                    "n": len(g),
                    "n_features": len(cols),
                }
                if scope_name == "rule":
                    row["rule"] = key
                model_rows.append(row)

    model_summary = pd.DataFrame(model_rows)

    # Contribution approximation: frequency * mean local delta by class
    freq = isolates.groupby("diag_class").size().rename("n").reset_index()
    freq["p"] = freq["n"] / freq["n"].sum()
    mean_delta = isolates.groupby("diag_class")["local_delta"].mean().rename("mean_local_delta").reset_index()
    contrib = freq.merge(mean_delta, on="diag_class")
    contrib["p_times_delta"] = contrib["p"] * contrib["mean_local_delta"]
    contrib = contrib.sort_values("p_times_delta")

    return class_summary, model_summary, contrib, data


def make_figures(class_summary, model_summary, contrib):
    g = class_summary[class_summary["rule"].eq("ALL")].sort_values("diag_class")
    plt.figure(figsize=(10,5))
    plt.bar(g["diag_class"].astype(str), g["mean_local_delta"])
    plt.axhline(0, linewidth=1)
    plt.xlabel("Diagonal-neighbor class")
    plt.ylabel("Mean local Δcomponents")
    plt.title("Mean local component change by embedded-isolate class")
    plt.tight_layout()
    plt.savefig(OUT / "fig_class_mean_delta.png", dpi=200)
    plt.close()

    c = contrib.sort_values("p_times_delta")
    plt.figure(figsize=(10,5))
    plt.bar(c["diag_class"].astype(str), c["p_times_delta"])
    plt.axhline(0, linewidth=1)
    plt.xlabel("Diagonal-neighbor class")
    plt.ylabel("Frequency × mean local Δ")
    plt.title("Class contribution proxy")
    plt.tight_layout()
    plt.savefig(OUT / "fig_class_contribution.png", dpi=200)
    plt.close()

    gm = model_summary[model_summary["scope"].eq("global")]
    plt.figure(figsize=(8,5))
    plt.bar(gm["model"], gm["R2_in_sample"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("In-sample R² on residual ΔC₁")
    plt.title("Transition-class decomposition models")
    plt.tight_layout()
    plt.savefig(OUT / "fig_transition_class_model_r2.png", dpi=200)
    plt.close()


def write_note(class_summary, model_summary, contrib):
    gm = model_summary[model_summary["scope"].eq("global")].set_index("model")
    lines = []
    lines.append("CA Isolate Transition-Class Decomposition")
    lines.append("")
    lines.append("Question: can the one-step isolate response be decomposed by local embedded-isolate classes?")
    lines.append("")
    lines.append("Global in-sample R2 on residual ΔC1:")
    for model in ["iso_count","diag_class_counts","diag_class_losses","counts_plus_losses"]:
        if model in gm.index:
            lines.append(f"  {model:20s}: {gm.loc[model,'R2_in_sample']:.4f}")

    lines.append("")
    lines.append("Top negative class contribution proxies p(class)*mean_local_delta:")
    for _, row in contrib.sort_values("p_times_delta").head(8).iterrows():
        lines.append(
            f"  class={int(row['diag_class']):02d}: "
            f"p={row['p']:.3f}, mean_delta={row['mean_local_delta']:.3f}, "
            f"p*delta={row['p_times_delta']:.4f}"
        )

    lines.append("")
    lines.append("Interpretation:")
    lines.append("- If class-count/loss models beat iso_count, local transition class structure refines the mechanism.")
    lines.append("- If iso_count remains comparable, the object count is already a near-sufficient summary at this resolution.")
    lines.append("- This is not yet a closed-form derivation; it is an empirical transition-class decomposition.")

    (OUT / "transition_class_note.txt").write_text("\n".join(lines))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--sizes", type=int, nargs="+", default=[64,128])
    p.add_argument("--densities", type=float, nargs="+", default=[0.20,0.25,0.30,0.35])
    p.add_argument("--local-radius", type=int, default=2)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--reuse-raw", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.sizes = [64]
        args.densities = [0.25,0.30]
        args.n_samples = min(args.n_samples, 300)

    sample_path = OUT / "transition_sample_raw.csv"
    isolate_path = OUT / "transition_isolate_raw.csv"

    if args.reuse_raw and sample_path.exists() and isolate_path.exists():
        samples = pd.read_csv(sample_path)
        isolates = pd.read_csv(isolate_path)
        print("Reusing raw transition-class files.")
    else:
        samples, isolates = simulate(args)
        samples.to_csv(sample_path, index=False)
        isolates.to_csv(isolate_path, index=False)
        print("Saved raw transition-class files.")

    class_summary, model_summary, contrib, data = class_decomposition(samples, isolates)
    class_summary.to_csv(OUT / "transition_class_summary.csv", index=False)
    model_summary.to_csv(OUT / "transition_class_model_summary.csv", index=False)
    contrib.to_csv(OUT / "transition_class_contributions.csv", index=False)
    data.to_csv(OUT / "transition_class_sample_design.csv", index=False)

    make_figures(class_summary, model_summary, contrib)
    write_note(class_summary, model_summary, contrib)

    print("\nTransition-class decomposition complete.")
    print((OUT / "transition_class_note.txt").read_text())


if __name__ == "__main__":
    main()
