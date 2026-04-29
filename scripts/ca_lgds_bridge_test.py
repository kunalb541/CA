#!/usr/bin/env python3
"""
CA ↔ LGDS bridge test.

Question:
Can we empirically see the LGDS common-projection logic inside the CA feature/target battery?

Formal analogy:
- finite CA feature space = description coordinates
- each prediction target = task
- each task induces a rank-1 predictive information matrix M_t = v_t v_t^T
  in whitened residual feature coordinates
- common rank-r optimality is approximate if task vectors v_t lie in a common r-plane

Tests:
1. Multi-target selection test:
   fine_net, density, block_var, block_entropy, future density/live count.
   Expected: no single rank-1 direction is globally optimal; targets split by description.

2. Horizon family test:
   fine_net at k={1,5,10,25,50,100,200}.
   Expected: horizon tasks are more coherent/aligned, with iso_embedded dominating.

Inputs:
- outputs/selection_principle/selection_raw.csv
- outputs/selection_principle_horizon/horizon_raw.csv

Outputs:
- outputs/ca_lgds_bridge/*
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs" / "ca_lgds_bridge"
OUT.mkdir(parents=True, exist_ok=True)

SELECTION_RAW = ROOT / "outputs" / "selection_principle" / "selection_raw.csv"
HORIZON_RAW = ROOT / "outputs" / "selection_principle_horizon" / "horizon_raw.csv"

EPS = 1e-10


def make_controls(df: pd.DataFrame) -> np.ndarray:
    """Condition dummies + live_count + density."""
    controls = []
    if "condition_id" in df.columns:
        dummies = pd.get_dummies(df["condition_id"], prefix="cond", drop_first=True)
        controls.append(dummies.to_numpy(float))
    for c in ["live_count", "density"]:
        if c in df.columns:
            controls.append(df[[c]].to_numpy(float))
    if not controls:
        return np.ones((len(df), 1))
    C = np.hstack(controls)
    return C


def residualize(A: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Residualize columns of A against controls C."""
    A = np.asarray(A, float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    model = LinearRegression()
    model.fit(C, A)
    return A - model.predict(C)


def whiten_features(X_resid: np.ndarray):
    """
    Standardize residual features, then whiten covariance.
    Returns Z whitened features and metadata.
    """
    Xs = StandardScaler().fit_transform(X_resid)
    Xs = np.nan_to_num(Xs)

    cov = (Xs.T @ Xs) / max(1, len(Xs) - 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    keep = evals > EPS
    evals_k = evals[keep]
    evecs_k = evecs[:, keep]

    Z = Xs @ evecs_k @ np.diag(1.0 / np.sqrt(evals_k))
    return Z, Xs, evals, evecs, keep


def task_vector(Z: np.ndarray, y_resid: np.ndarray):
    """Rank-1 predictive information vector in whitened coordinates."""
    y = np.asarray(y_resid, float).ravel()
    y = y - np.mean(y)
    sy = np.std(y)
    if sy < EPS:
        return None
    y = y / sy
    v = (Z.T @ y) / max(1, len(y) - 1)
    norm = np.linalg.norm(v)
    if norm < EPS:
        return v, None, 0.0
    return v, v / norm, norm ** 2


def direction_feature_correlations(Z: np.ndarray, Xs: np.ndarray, u: np.ndarray, feature_names):
    """
    Interpret a whitened direction u by correlating its score Z@u
    with standardized residual original features.
    """
    score = Z @ u
    out = {}
    for j, f in enumerate(feature_names):
        x = Xs[:, j]
        if np.std(score) < EPS or np.std(x) < EPS:
            out[f"corr_{f}"] = np.nan
        else:
            out[f"corr_{f}"] = float(np.corrcoef(score, x)[0, 1])
    return out


def analyze_task_family(df: pd.DataFrame, feature_names, target_map, label: str):
    """
    Build empirical task vectors and common-projection diagnostics.
    """
    C = make_controls(df)

    X = df[feature_names].to_numpy(float)
    X_resid = residualize(X, C)
    Z, Xs, evals, evecs, keep = whiten_features(X_resid)

    rows = []
    vectors = []
    names = []

    for task_name, y_col in tqdm(target_map.items(), desc=f"Tasks: {label}"):
        if y_col not in df.columns:
            continue

        y = df[y_col].to_numpy(float)
        y_resid = residualize(y, C).ravel()

        tv = task_vector(Z, y_resid)
        if tv is None:
            continue
        v, u, max_score = tv
        if u is None:
            continue

        interp = direction_feature_correlations(Z, Xs, u, feature_names)

        best_feature = max(
            feature_names,
            key=lambda f: abs(interp.get(f"corr_{f}", 0.0))
        )

        row = {
            "family": label,
            "task": task_name,
            "target_col": y_col,
            "n": len(df),
            "max_trace_score_rank1": max_score,
            "best_interpreting_feature": best_feature,
        }
        row.update(interp)
        rows.append(row)

        vectors.append(v)
        names.append(task_name)

    if not vectors:
        raise RuntimeError(f"No task vectors built for {label}")

    V = np.column_stack(vectors)  # whitened_dim x n_tasks
    Udirs = np.column_stack([v / (np.linalg.norm(v) + EPS) for v in vectors])

    # Pairwise absolute cosines between task directions
    cos = np.abs(Udirs.T @ Udirs)
    cos_df = pd.DataFrame(cos, index=names, columns=names)

    # Singular values / effective dimensionality
    svals = np.linalg.svd(Udirs, compute_uv=False)
    sv_df = pd.DataFrame({
        "family": label,
        "singular_index": np.arange(1, len(svals) + 1),
        "singular_value": svals,
        "normalized_energy": (svals ** 2) / np.sum(svals ** 2),
        "cumulative_energy": np.cumsum(svals ** 2) / np.sum(svals ** 2),
    })

    # Common rank-r projection from top left singular vectors of raw V
    U, S, _ = np.linalg.svd(V, full_matrices=False)

    regret_rows = []
    max_r = min(V.shape[0], V.shape[1])
    for r in range(1, max_r + 1):
        B = U[:, :r]
        for task_name, v in zip(names, vectors):
            max_score = float(np.linalg.norm(v) ** 2)
            score_r = float(np.linalg.norm(B.T @ v) ** 2)
            regret = max_score - score_r
            rel_regret = regret / (max_score + EPS)
            regret_rows.append({
                "family": label,
                "rank_r": r,
                "task": task_name,
                "max_score": max_score,
                "common_projection_score": score_r,
                "regret": regret,
                "relative_regret": rel_regret,
            })

    task_df = pd.DataFrame(rows)
    regret_df = pd.DataFrame(regret_rows)

    # Summary
    offdiag = cos[np.triu_indices_from(cos, k=1)]
    summary = {
        "family": label,
        "n_tasks": len(names),
        "n_features": len(feature_names),
        "mean_pairwise_abs_cosine": float(np.mean(offdiag)) if len(offdiag) else 1.0,
        "min_pairwise_abs_cosine": float(np.min(offdiag)) if len(offdiag) else 1.0,
        "max_pairwise_abs_cosine": float(np.max(offdiag)) if len(offdiag) else 1.0,
        "rank1_cumulative_energy": float(sv_df.loc[sv_df["singular_index"].eq(1), "cumulative_energy"].iloc[0]),
        "rank2_cumulative_energy": float(sv_df.loc[sv_df["singular_index"].eq(2), "cumulative_energy"].iloc[0]) if len(svals) >= 2 else 1.0,
    }

    return task_df, cos_df, sv_df, regret_df, pd.DataFrame([summary])


def prepare_selection_family():
    df = pd.read_csv(SELECTION_RAW)

    # Feature space: nontrivial description features after controls.
    candidates = ["iso_embedded", "block_var", "block_entropy", "perimeter_count"]
    feature_names = [c for c in candidates if c in df.columns]

    target_candidates = [
        "target_fine_net",
        "target_delta_components",
        "target_delta_density",
        "target_delta_block_var",
        "target_delta_block_entropy",
        "target_future_density",
        "target_future_live_count",
    ]
    target_map = {c: c for c in target_candidates if c in df.columns}

    return df, feature_names, target_map


def prepare_horizon_family():
    df = pd.read_csv(HORIZON_RAW)

    candidates = ["iso_embedded", "block_var", "block_entropy"]
    feature_names = [c for c in candidates if c in df.columns]

    # Convert horizon rows into wide target columns, one target per horizon.
    index_cols = [
        "sample_id", "condition_id", "rule", "L", "rho",
        "live_count", "density", "iso_embedded", "block_var", "block_entropy"
    ]
    index_cols = [c for c in index_cols if c in df.columns]

    wide = df.pivot_table(
        index=index_cols,
        columns="horizon",
        values="fine_net",
        aggfunc="first"
    ).reset_index()

    target_map = {}
    for h in sorted([c for c in wide.columns if isinstance(c, (int, np.integer))]):
        col = f"fine_net_k{int(h)}"
        wide = wide.rename(columns={h: col})
        target_map[col] = col

    return wide, feature_names, target_map


def write_verdict(summaries, regrets):
    lines = []
    lines.append("CA ↔ LGDS Bridge Test")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- High cosine / low rank means tasks share an approximate common description direction.")
    lines.append("- Low cosine / high relative regret means no single low-rank description is globally privileged.")
    lines.append("")

    for _, row in summaries.iterrows():
        fam = row["family"]
        lines.append(f"{fam}:")
        lines.append(f"  n_tasks = {int(row['n_tasks'])}")
        lines.append(f"  mean pairwise |cos| = {row['mean_pairwise_abs_cosine']:.3f}")
        lines.append(f"  min pairwise |cos| = {row['min_pairwise_abs_cosine']:.3f}")
        lines.append(f"  rank-1 energy = {row['rank1_cumulative_energy']:.3f}")
        lines.append(f"  rank-2 energy = {row['rank2_cumulative_energy']:.3f}")

        rr = regrets[(regrets["family"].eq(fam)) & (regrets["rank_r"].eq(1))]
        lines.append(f"  mean rank-1 relative regret = {rr['relative_regret'].mean():.3f}")
        lines.append(f"  max rank-1 relative regret = {rr['relative_regret'].max():.3f}")
        lines.append("")

    # Heuristic verdict
    sel = summaries[summaries["family"].eq("selection_multi_target")]
    hor = summaries[summaries["family"].eq("horizon_fine_net")]
    if len(sel) and len(hor):
        sel_r1 = float(sel["rank1_cumulative_energy"].iloc[0])
        hor_r1 = float(hor["rank1_cumulative_energy"].iloc[0])
        if hor_r1 > sel_r1 + 0.2:
            lines.append("VERDICT: PASS")
            lines.append("Fine-net horizon tasks are substantially more coherent than heterogeneous targets.")
            lines.append("This supports task-indexed description privilege: one description direction aligns within a target family but not globally across targets.")
        else:
            lines.append("VERDICT: CHECK")
            lines.append("Horizon tasks are not clearly more coherent than heterogeneous targets; inspect matrices.")

    (OUT / "bridge_verdict.txt").write_text("\n".join(lines))


def main():
    all_task = []
    all_sv = []
    all_regret = []
    all_summary = []

    # 1. Multi-target family
    if SELECTION_RAW.exists():
        df, feature_names, target_map = prepare_selection_family()
        task_df, cos_df, sv_df, regret_df, summary_df = analyze_task_family(
            df, feature_names, target_map, "selection_multi_target"
        )
        task_df.to_csv(OUT / "selection_task_directions.csv", index=False)
        cos_df.to_csv(OUT / "selection_pairwise_abs_cosines.csv")
        sv_df.to_csv(OUT / "selection_singular_values.csv", index=False)
        regret_df.to_csv(OUT / "selection_common_projection_regret.csv", index=False)

        all_task.append(task_df)
        all_sv.append(sv_df)
        all_regret.append(regret_df)
        all_summary.append(summary_df)

    # 2. Fine-net horizon family
    if HORIZON_RAW.exists():
        df, feature_names, target_map = prepare_horizon_family()
        task_df, cos_df, sv_df, regret_df, summary_df = analyze_task_family(
            df, feature_names, target_map, "horizon_fine_net"
        )
        task_df.to_csv(OUT / "horizon_task_directions.csv", index=False)
        cos_df.to_csv(OUT / "horizon_pairwise_abs_cosines.csv")
        sv_df.to_csv(OUT / "horizon_singular_values.csv", index=False)
        regret_df.to_csv(OUT / "horizon_common_projection_regret.csv", index=False)

        all_task.append(task_df)
        all_sv.append(sv_df)
        all_regret.append(regret_df)
        all_summary.append(summary_df)

    task_all = pd.concat(all_task, ignore_index=True)
    sv_all = pd.concat(all_sv, ignore_index=True)
    regret_all = pd.concat(all_regret, ignore_index=True)
    summary_all = pd.concat(all_summary, ignore_index=True)

    task_all.to_csv(OUT / "bridge_task_directions_all.csv", index=False)
    sv_all.to_csv(OUT / "bridge_singular_values_all.csv", index=False)
    regret_all.to_csv(OUT / "bridge_common_projection_regret_all.csv", index=False)
    summary_all.to_csv(OUT / "bridge_summary.csv", index=False)

    write_verdict(summary_all, regret_all)

    print("\nCA ↔ LGDS bridge test complete.")
    print("Outputs in:", OUT)
    print("\n=== SUMMARY ===")
    print(summary_all.to_string(index=False))
    print("\n=== VERDICT ===")
    print((OUT / "bridge_verdict.txt").read_text())


if __name__ == "__main__":
    main()
