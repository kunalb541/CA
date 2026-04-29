#!/usr/bin/env python3
"""
CA mechanism transfer test with condition-standardized targets/features.

Purpose:
The raw transfer test passed across density/rule/condition but failed across size
with huge negative R2 despite positive correlations. That indicates scale
miscalibration. This script standardizes target and features within each condition
before transfer.

Inputs:
- outputs/isolate_fate/fate_raw.csv
- outputs/isolate_transition_classes/transition_class_sample_design.csv

Outputs:
- outputs/mechanism_transfer_standardized/
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs" / "mechanism_transfer_standardized"
OUT.mkdir(parents=True, exist_ok=True)

FATE_RAW = ROOT / "outputs" / "isolate_fate" / "fate_raw.csv"
TRANSITION_DESIGN = ROOT / "outputs" / "isolate_transition_classes" / "transition_class_sample_design.csv"


def residualize_within_condition(df, target_col="fine_net_1"):
    rows = []
    for cid, g in df.groupby("condition_id"):
        g = g.copy()
        X = g[["live_count", "density"]].to_numpy(float)
        y = g[target_col].to_numpy(float)
        m = LinearRegression().fit(X, y)
        resid = y - m.predict(X)
        g["fine_net_resid"] = resid
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def load_design():
    fate = pd.read_csv(FATE_RAW)
    fate = residualize_within_condition(fate, "fine_net_1")

    trans = pd.read_csv(TRANSITION_DESIGN)
    trans_cols = ["sample_id"] + [
        c for c in trans.columns
        if c.startswith("class_") and (c.endswith("_count") or c.endswith("_loss"))
    ]
    merged = fate.merge(trans[trans_cols], on="sample_id", how="left")
    class_cols = [c for c in merged.columns if c.startswith("class_")]
    merged[class_cols] = merged[class_cols].fillna(0)
    return merged


def model_specs(df):
    class_count_cols = sorted([c for c in df.columns if c.startswith("class_") and c.endswith("_count")])
    class_loss_cols = sorted([c for c in df.columns if c.startswith("class_") and c.endswith("_loss")])

    specs = {
        "iso_count": ["iso_count"],
        "local_window_loss": ["iso_local_window_loss_sum"],
        "local_window_delta_loss_gain": [
            "iso_local_window_delta_sum",
            "iso_local_window_loss_sum",
            "iso_local_window_gain_sum",
        ],
        "fate_core": [
            "iso_die",
            "iso_survive_connected",
            "iso_orth_birth_any",
            "iso_diag_birth_any",
        ],
        "fate_all": [
            "iso_die",
            "iso_survive_isolated",
            "iso_survive_connected",
            "iso_orth_birth_any",
            "iso_diag_birth_any",
            "iso_local_window_delta_sum",
            "iso_local_window_loss_sum",
            "iso_local_window_gain_sum",
        ],
        "class_counts": class_count_cols,
        "class_losses": class_loss_cols,
        "class_counts_plus_losses": class_count_cols + class_loss_cols,
    }
    return {k: [c for c in v if c in df.columns] for k, v in specs.items() if len([c for c in v if c in df.columns]) > 0}


def add_condition_standardized_columns(df, features):
    """
    For each condition:
    - y_z = zscore(fine_net_resid)
    - each feature_z = zscore(feature)
    """
    rows = []
    for cid, g in df.groupby("condition_id"):
        g = g.copy()

        y = g["fine_net_resid"].to_numpy(float)
        y_sd = np.std(y)
        if y_sd == 0:
            g["fine_net_resid_z"] = 0.0
        else:
            g["fine_net_resid_z"] = (y - np.mean(y)) / y_sd

        for f in features:
            x = g[f].to_numpy(float)
            x_sd = np.std(x)
            if x_sd == 0:
                g[f"{f}_z"] = 0.0
            else:
                g[f"{f}_z"] = (x - np.mean(x)) / x_sd

        rows.append(g)

    return pd.concat(rows, ignore_index=True)


def make_splits(df):
    splits = []

    for rho in sorted(df["rho"].unique()):
        splits.append(("leave_density", f"rho={rho}", df["rho"] != rho, df["rho"] == rho))

    for L in sorted(df["L"].unique()):
        splits.append(("leave_size", f"L={L}", df["L"] != L, df["L"] == L))

    for rule in sorted(df["rule"].unique()):
        splits.append(("leave_rule", f"rule={rule}", df["rule"] != rule, df["rule"] == rule))

    for cid in sorted(df["condition_id"].unique()):
        splits.append(("leave_condition", cid, df["condition_id"] != cid, df["condition_id"] == cid))

    return splits


def fit_predict(train, test, zfeatures):
    Xtr = train[zfeatures].to_numpy(float)
    ytr = train["fine_net_resid_z"].to_numpy(float)
    Xte = test[zfeatures].to_numpy(float)
    yte = test["fine_net_resid_z"].to_numpy(float)

    if len(train) < 5 or len(test) < 5:
        return np.nan, np.nan
    if np.std(ytr) == 0 or np.std(yte) == 0:
        return np.nan, np.nan
    if Xtr.shape[1] == 0 or np.all(np.std(Xtr, axis=0) == 0):
        return np.nan, np.nan

    model = RidgeCV(alphas=np.logspace(-6, 3, 20))
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    r2 = float(r2_score(yte, pred))
    corr = float(np.corrcoef(pred, yte)[0, 1]) if np.std(pred) > 0 and np.std(yte) > 0 else np.nan
    return r2, corr


def run():
    df = load_design()
    specs = model_specs(df)
    all_features = sorted(set(sum(specs.values(), [])))
    dfz = add_condition_standardized_columns(df, all_features)

    rows = []
    for split_type, heldout, train_mask, test_mask in make_splits(dfz):
        train = dfz[train_mask].copy()
        test = dfz[test_mask].copy()

        for model_name, feats in specs.items():
            zfeats = [f"{f}_z" for f in feats]
            r2, corr = fit_predict(train, test, zfeats)
            rows.append({
                "split_type": split_type,
                "heldout": heldout,
                "model": model_name,
                "n_train": len(train),
                "n_test": len(test),
                "n_features": len(feats),
                "test_R2_z": r2,
                "test_corr_z": corr,
                "features": ",".join(feats),
            })

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "transfer_standardized_results.csv", index=False)

    summary_rows = []
    for (split_type, model), g in res.groupby(["split_type", "model"]):
        vals = g["test_R2_z"].dropna().to_numpy(float)
        corrs = g["test_corr_z"].dropna().to_numpy(float)
        summary_rows.append({
            "split_type": split_type,
            "model": model,
            "n_splits": len(vals),
            "mean_test_R2_z": float(np.mean(vals)),
            "median_test_R2_z": float(np.median(vals)),
            "min_test_R2_z": float(np.min(vals)),
            "max_test_R2_z": float(np.max(vals)),
            "frac_R2_positive": float(np.mean(vals > 0)),
            "mean_corr_z": float(np.mean(corrs)) if len(corrs) else np.nan,
            "frac_corr_positive": float(np.mean(corrs > 0)) if len(corrs) else np.nan,
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "transfer_standardized_summary.csv", index=False)

    # Figure
    models_order = [
        "iso_count",
        "local_window_loss",
        "local_window_delta_loss_gain",
        "fate_core",
        "fate_all",
        "class_counts",
        "class_losses",
        "class_counts_plus_losses",
    ]
    split_order = ["leave_density", "leave_size", "leave_rule", "leave_condition"]
    pivot = summary.pivot(index="model", columns="split_type", values="mean_test_R2_z")
    pivot = pivot.reindex(models_order)
    pivot = pivot[[c for c in split_order if c in pivot.columns]]
    ax = pivot.plot(kind="bar", figsize=(11, 5))
    ax.axhline(0, linewidth=1)
    ax.set_ylabel("Mean held-out R² after condition z-scoring")
    ax.set_title("Condition-standardized mechanism transfer")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / "fig_transfer_standardized_r2.png", dpi=200)
    plt.close()

    # Verdict
    lines = []
    lines.append("CA Mechanism Transfer Test — Condition-Standardized")
    lines.append("")
    lines.append("All targets and mechanism features are z-scored within condition before transfer.")
    lines.append("This tests directional/mechanism transport after removing size/amplitude calibration.")
    lines.append("")

    for split_type in split_order:
        lines.append(f"\n{split_type}:")
        sub = summary[summary["split_type"].eq(split_type)].sort_values("mean_test_R2_z", ascending=False)
        for _, row in sub.iterrows():
            lines.append(
                f"  {row['model']:28s} "
                f"mean_R2_z={row['mean_test_R2_z']:.4f}, "
                f"median_R2_z={row['median_test_R2_z']:.4f}, "
                f"min_R2_z={row['min_test_R2_z']:.4f}, "
                f"frac_R2_pos={row['frac_R2_positive']:.2f}, "
                f"mean_corr_z={row['mean_corr_z']:.3f}"
            )

    key = summary[summary["model"].isin(["fate_all", "class_counts_plus_losses"])]
    best_by_split = key.groupby("split_type")["mean_test_R2_z"].max()
    all_positive = bool((best_by_split > 0).all())
    size_positive = bool(best_by_split.get("leave_size", -999) > 0)

    lines.append("")
    if all_positive and size_positive:
        lines.append("VERDICT: PASS")
        lines.append("After condition-standardization, mechanism representations transfer across density, size, rule, and condition.")
    else:
        lines.append("VERDICT: CHECK")
        lines.append("Even after condition-standardization, at least one split type remains non-positive.")

    (OUT / "transfer_standardized_verdict.txt").write_text("\n".join(lines))

    print("\n".join(lines))
    print("\nSaved outputs to:", OUT)


if __name__ == "__main__":
    run()
