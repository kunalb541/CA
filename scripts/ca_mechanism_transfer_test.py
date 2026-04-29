#!/usr/bin/env python3
"""
CA mechanism transfer test.

Question:
Do isolate/fate/transition-class mechanisms learned in one condition transfer
to held-out rule, size, density, or condition?

Inputs:
- outputs/isolate_fate/fate_raw.csv
- outputs/isolate_transition_classes/transition_class_sample_design.csv

Outputs:
- outputs/mechanism_transfer/transfer_results.csv
- outputs/mechanism_transfer/transfer_summary.csv
- outputs/mechanism_transfer/transfer_verdict.txt
- outputs/mechanism_transfer/fig_transfer_r2.png
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

OUT = ROOT / "outputs" / "mechanism_transfer"
OUT.mkdir(parents=True, exist_ok=True)

FATE_RAW = ROOT / "outputs" / "isolate_fate" / "fate_raw.csv"
TRANSITION_DESIGN = ROOT / "outputs" / "isolate_transition_classes" / "transition_class_sample_design.csv"


def residualize_within_condition(df: pd.DataFrame, target_col: str = "fine_net_1") -> pd.DataFrame:
    """Add fine_net_resid by regressing target on live_count+density within each condition."""
    rows = []
    for condition_id, g in df.groupby("condition_id"):
        g = g.copy()
        X = g[["live_count", "density"]].to_numpy(float)
        y = g[target_col].to_numpy(float)
        model = LinearRegression().fit(X, y)
        g["fine_net_resid"] = y - model.predict(X)
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def load_design() -> pd.DataFrame:
    if not FATE_RAW.exists():
        raise FileNotFoundError(f"Missing {FATE_RAW}")
    if not TRANSITION_DESIGN.exists():
        raise FileNotFoundError(f"Missing {TRANSITION_DESIGN}")

    fate = pd.read_csv(FATE_RAW)
    fate = residualize_within_condition(fate, "fine_net_1")

    trans = pd.read_csv(TRANSITION_DESIGN)

    # transition_class_sample_design already has fine_net_resid and class columns.
    # Keep only transition columns and identifiers, then merge into fate.
    trans_cols = ["sample_id"] + [
        c for c in trans.columns
        if (c.startswith("class_") and (c.endswith("_count") or c.endswith("_loss")))
    ]

    merged = fate.merge(trans[trans_cols], on="sample_id", how="left")
    class_cols = [c for c in merged.columns if c.startswith("class_")]
    merged[class_cols] = merged[class_cols].fillna(0)

    return merged


def model_specs(df: pd.DataFrame):
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

    # Keep only models whose columns exist and are nonempty
    clean = {}
    for name, cols in specs.items():
        cols = [c for c in cols if c in df.columns]
        if cols:
            clean[name] = cols
    return clean


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    Xtr = train[features].to_numpy(float)
    ytr = train["fine_net_resid"].to_numpy(float)
    Xte = test[features].to_numpy(float)
    yte = test["fine_net_resid"].to_numpy(float)

    if len(train) < 5 or len(test) < 5:
        return np.nan, np.nan, np.nan
    if np.std(ytr) == 0 or np.std(yte) == 0:
        return np.nan, np.nan, np.nan
    if Xtr.shape[1] == 0:
        return np.nan, np.nan, np.nan
    if np.all(np.std(Xtr, axis=0) == 0):
        return np.nan, np.nan, np.nan

    model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-6, 3, 20))
    )
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    r2 = float(r2_score(yte, pred))

    # Fit a single-feature signed slope proxy for iso/local models if possible.
    # For multivariate models, compute corr(pred, yte) sign as transfer direction.
    corr = np.corrcoef(pred, yte)[0, 1] if np.std(pred) > 0 and np.std(yte) > 0 else np.nan

    return r2, corr, float(np.mean(pred))


def make_splits(df: pd.DataFrame):
    splits = []

    # Leave-one-density-out
    for rho in sorted(df["rho"].unique()):
        splits.append({
            "split_type": "leave_density",
            "heldout": f"rho={rho}",
            "train_idx": df["rho"] != rho,
            "test_idx": df["rho"] == rho,
        })

    # Leave-one-size-out
    for L in sorted(df["L"].unique()):
        splits.append({
            "split_type": "leave_size",
            "heldout": f"L={L}",
            "train_idx": df["L"] != L,
            "test_idx": df["L"] == L,
        })

    # Leave-one-rule-out
    for rule in sorted(df["rule"].unique()):
        splits.append({
            "split_type": "leave_rule",
            "heldout": f"rule={rule}",
            "train_idx": df["rule"] != rule,
            "test_idx": df["rule"] == rule,
        })

    # Leave-one-condition-out
    for cid in sorted(df["condition_id"].unique()):
        splits.append({
            "split_type": "leave_condition",
            "heldout": cid,
            "train_idx": df["condition_id"] != cid,
            "test_idx": df["condition_id"] == cid,
        })

    return splits


def run_transfer(df: pd.DataFrame) -> pd.DataFrame:
    specs = model_specs(df)
    splits = make_splits(df)
    rows = []

    for split in splits:
        train = df[split["train_idx"]].copy()
        test = df[split["test_idx"]].copy()

        for model_name, features in specs.items():
            r2, corr, mean_pred = fit_predict(train, test, features)
            rows.append({
                "split_type": split["split_type"],
                "heldout": split["heldout"],
                "model": model_name,
                "n_train": len(train),
                "n_test": len(test),
                "n_features": len(features),
                "test_R2": r2,
                "test_corr_pred_target": corr,
                "mean_prediction": mean_pred,
                "features": ",".join(features),
            })

    return pd.DataFrame(rows)


def summarize(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split_type, model), g in res.groupby(["split_type", "model"]):
        vals = g["test_R2"].dropna().to_numpy(float)
        corrs = g["test_corr_pred_target"].dropna().to_numpy(float)

        if len(vals) == 0:
            continue

        rows.append({
            "split_type": split_type,
            "model": model,
            "n_splits": len(vals),
            "mean_test_R2": float(np.mean(vals)),
            "median_test_R2": float(np.median(vals)),
            "min_test_R2": float(np.min(vals)),
            "max_test_R2": float(np.max(vals)),
            "frac_R2_positive": float(np.mean(vals > 0)),
            "mean_corr": float(np.mean(corrs)) if len(corrs) else np.nan,
            "frac_corr_positive": float(np.mean(corrs > 0)) if len(corrs) else np.nan,
        })

    return pd.DataFrame(rows)


def make_fig(summary: pd.DataFrame):
    # Plot mean transfer R2 by split/model
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

    pivot = summary.pivot(index="model", columns="split_type", values="mean_test_R2")
    pivot = pivot.reindex(models_order)
    pivot = pivot[[c for c in split_order if c in pivot.columns]]

    ax = pivot.plot(kind="bar", figsize=(11, 5))
    ax.axhline(0, linewidth=1)
    ax.set_ylabel("Mean held-out R²")
    ax.set_title("Mechanism transfer across held-out regimes")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / "fig_transfer_r2.png", dpi=200)
    plt.close()


def write_verdict(summary: pd.DataFrame, res: pd.DataFrame):
    lines = []
    lines.append("CA Mechanism Transfer Test")
    lines.append("")
    lines.append("Question: do isolate/fate/class mechanisms transfer across held-out density, size, rule, and condition?")
    lines.append("")
    lines.append("Summary by split/model:")
    for split_type in ["leave_density", "leave_size", "leave_rule", "leave_condition"]:
        lines.append(f"\n{split_type}:")
        sub = summary[summary["split_type"].eq(split_type)].sort_values("mean_test_R2", ascending=False)
        for _, row in sub.iterrows():
            lines.append(
                f"  {row['model']:28s} "
                f"mean_R2={row['mean_test_R2']:.4f}, "
                f"median_R2={row['median_test_R2']:.4f}, "
                f"min_R2={row['min_test_R2']:.4f}, "
                f"frac_R2_pos={row['frac_R2_positive']:.2f}, "
                f"mean_corr={row['mean_corr']:.3f}"
            )

    # Verdict logic
    key_models = ["iso_count", "local_window_loss", "fate_all", "class_losses", "class_counts_plus_losses"]
    verdict_rows = summary[summary["model"].isin(key_models)]
    pos_by_split = verdict_rows.groupby("split_type")["mean_test_R2"].max()

    all_splits_positive = bool((pos_by_split > 0).all())
    rule_positive = bool(pos_by_split.get("leave_rule", -999) > 0)
    condition_positive = bool(pos_by_split.get("leave_condition", -999) > 0)

    # Does a mechanism model beat iso_count?
    beats = []
    for split_type, g in summary.groupby("split_type"):
        iso = g[g["model"].eq("iso_count")]["mean_test_R2"]
        if len(iso) == 0:
            continue
        iso_val = float(iso.iloc[0])
        best_mech = g[g["model"].isin(["local_window_loss", "fate_all", "class_losses", "class_counts_plus_losses"])]["mean_test_R2"].max()
        beats.append(best_mech > iso_val)

    lines.append("")
    if all_splits_positive and rule_positive and condition_positive and any(beats):
        lines.append("VERDICT: PASS")
        lines.append("At least one mechanism representation transfers with positive held-out R² across all split types, including rule and condition, and mechanism features beat iso_count in at least one split.")
    elif all_splits_positive:
        lines.append("VERDICT: WEAK PASS")
        lines.append("Mechanism transfers with positive held-out R², but detailed mechanism features do not clearly beat iso_count.")
    else:
        lines.append("VERDICT: FAIL/CHECK")
        lines.append("Transfer is not consistently positive across split types.")

    lines.append("")
    lines.append("Interpretation guide:")
    lines.append("- PASS: mechanism travels across held-out regimes.")
    lines.append("- WEAK PASS: robust object count transfers, detailed mechanism may overfit.")
    lines.append("- FAIL/CHECK: within-condition mechanism may not transport.")

    (OUT / "transfer_verdict.txt").write_text("\n".join(lines))


def main():
    df = load_design()
    res = run_transfer(df)
    summary = summarize(res)

    res.to_csv(OUT / "transfer_results.csv", index=False)
    summary.to_csv(OUT / "transfer_summary.csv", index=False)

    make_fig(summary)
    write_verdict(summary, res)

    print("\nCA mechanism transfer test complete.")
    print("Saved outputs to:", OUT)
    print("\n" + (OUT / "transfer_verdict.txt").read_text())


if __name__ == "__main__":
    main()
