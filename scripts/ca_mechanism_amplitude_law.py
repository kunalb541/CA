#!/usr/bin/env python3
"""
CA mechanism amplitude-law test.

Question:
The isolate/fate mechanism transfers after condition-standardization.
Can the raw amplitude/scale factor be predicted from rule, L, and rho?

Interpretation:
If standardized mechanism M_z transfers but raw transfer fails across size,
then raw residual response may have the form

    y_resid = a_condition * M_z + noise

This script estimates a_condition for each condition and tests whether
a_condition is predictable from rule, L, rho.

Inputs:
- outputs/isolate_fate/fate_raw.csv

Outputs:
- outputs/mechanism_amplitude_law/amplitude_condition_table.csv
- outputs/mechanism_amplitude_law/amplitude_model_summary.csv
- outputs/mechanism_amplitude_law/amplitude_verdict.txt
- outputs/mechanism_amplitude_law/fig_amplitude_by_condition.png
- outputs/mechanism_amplitude_law/fig_amplitude_vs_density.png
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs" / "mechanism_amplitude_law"
OUT.mkdir(parents=True, exist_ok=True)

FATE_RAW = ROOT / "outputs" / "isolate_fate" / "fate_raw.csv"


MECH_MODELS = {
    "iso_count": ["iso_count"],
    "local_window_loss": ["iso_local_window_loss_sum"],
    "local_window_delta_loss_gain": [
        "iso_local_window_delta_sum",
        "iso_local_window_loss_sum",
        "iso_local_window_gain_sum",
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
}


def residualize_within_condition(df):
    rows = []
    for cid, g in df.groupby("condition_id"):
        g = g.copy()
        X = g[["live_count", "density"]].to_numpy(float)
        y = g["fine_net_1"].to_numpy(float)
        m = LinearRegression().fit(X, y)
        g["fine_net_resid"] = y - m.predict(X)
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def condition_mechanism_score(g, features):
    """
    Build standardized mechanism score inside condition:
    fit y_z ~ X_z, return predicted score M_z.
    This estimates the condition's best mechanism coordinate for that representation.
    """
    y = g["fine_net_resid"].to_numpy(float)
    X = g[features].to_numpy(float)

    if np.std(y) == 0 or X.shape[1] == 0:
        return None

    yz = (y - y.mean()) / y.std()

    Xz = X.copy().astype(float)
    for j in range(Xz.shape[1]):
        sd = Xz[:, j].std()
        if sd == 0:
            Xz[:, j] = 0.0
        else:
            Xz[:, j] = (Xz[:, j] - Xz[:, j].mean()) / sd

    model = RidgeCV(alphas=np.logspace(-6, 3, 20))
    model.fit(Xz, yz)
    mz = model.predict(Xz)

    # Orient so mechanism score positively tracks y residual.
    # Since y_resid is usually negative with isolate loss features, the model prediction
    # already has the correct sign for y. Keep as predicted y_z mechanism score.
    if np.std(mz) == 0:
        return None

    return yz, mz


def estimate_condition_amplitudes(df):
    rows = []

    for cid, g in df.groupby("condition_id"):
        rule = g["rule"].iloc[0]
        L = int(g["L"].iloc[0])
        rho = float(g["rho"].iloc[0])

        y_raw = g["fine_net_resid"].to_numpy(float)
        y_sd = float(np.std(y_raw))

        for mech_name, features in MECH_MODELS.items():
            features = [f for f in features if f in g.columns]
            out = condition_mechanism_score(g, features)
            if out is None:
                continue

            yz, mz = out

            # Standardized strength inside condition
            r2_z = float(r2_score(yz, mz))
            corr_z = float(np.corrcoef(yz, mz)[0, 1])

            # Amplitude law:
            # y_raw ≈ a * mz + b
            # mz is dimensionless standardized mechanism coordinate.
            reg = LinearRegression().fit(mz.reshape(-1, 1), y_raw)
            pred_raw = reg.predict(mz.reshape(-1, 1))
            amp = float(reg.coef_[0])
            intercept = float(reg.intercept_)
            raw_r2 = float(r2_score(y_raw, pred_raw))

            rows.append({
                "condition_id": cid,
                "rule": rule,
                "L": L,
                "rho": rho,
                "mechanism": mech_name,
                "n": len(g),
                "y_resid_sd": y_sd,
                "amplitude": amp,
                "intercept": intercept,
                "within_condition_R2_z": r2_z,
                "within_condition_corr_z": corr_z,
                "raw_R2_from_mechanism_score": raw_r2,
            })

    return pd.DataFrame(rows)


def design_matrix(tab):
    """
    Predict amplitude from rule, log2L, rho, and simple interactions.
    """
    d = tab.copy()
    d["is_highlife"] = (d["rule"] == "HighLife").astype(float)
    d["log2L"] = np.log2(d["L"].astype(float))
    d["rho_centered"] = d["rho"] - d["rho"].mean()
    d["log2L_centered"] = d["log2L"] - d["log2L"].mean()
    d["rho_x_log2L"] = d["rho_centered"] * d["log2L_centered"]
    d["rho_x_rule"] = d["rho_centered"] * d["is_highlife"]
    d["L_x_rule"] = d["log2L_centered"] * d["is_highlife"]

    features = [
        "is_highlife",
        "log2L_centered",
        "rho_centered",
        "rho_x_log2L",
        "rho_x_rule",
        "L_x_rule",
    ]
    return d, features


def fit_amplitude_law(tab):
    rows = []

    for mech, g in tab.groupby("mechanism"):
        d, feats = design_matrix(g)
        X = d[feats].to_numpy(float)
        y = d["amplitude"].to_numpy(float)

        # In-sample OLS
        ols = LinearRegression().fit(X, y)
        pred = ols.predict(X)
        r2_in = float(r2_score(y, pred))

        # Leave-one-condition-out CV
        loo = LeaveOneOut()
        preds = np.zeros_like(y, dtype=float)
        for train_idx, test_idx in loo.split(X):
            model = RidgeCV(alphas=np.logspace(-6, 3, 20))
            model.fit(X[train_idx], y[train_idx])
            preds[test_idx] = model.predict(X[test_idx])
        r2_loo = float(r2_score(y, preds))
        corr_loo = float(np.corrcoef(y, preds)[0, 1]) if np.std(preds) > 0 and np.std(y) > 0 else np.nan

        # Simpler models for interpretability
        simple_specs = {
            "rule_only": ["is_highlife"],
            "size_only": ["log2L_centered"],
            "rho_only": ["rho_centered"],
            "size_rho": ["log2L_centered", "rho_centered"],
            "rule_size_rho": ["is_highlife", "log2L_centered", "rho_centered"],
            "full": feats,
        }

        for spec_name, spec_feats in simple_specs.items():
            Xs = d[spec_feats].to_numpy(float)
            pred_cv = np.zeros_like(y, dtype=float)
            for train_idx, test_idx in loo.split(Xs):
                model = RidgeCV(alphas=np.logspace(-6, 3, 20))
                model.fit(Xs[train_idx], y[train_idx])
                pred_cv[test_idx] = model.predict(Xs[test_idx])
            rows.append({
                "mechanism": mech,
                "amplitude_model": spec_name,
                "n_conditions": len(g),
                "amplitude_mean": float(np.mean(y)),
                "amplitude_sd": float(np.std(y, ddof=1)),
                "amplitude_cv": float(np.std(y, ddof=1) / (abs(np.mean(y)) + 1e-12)),
                "R2_in_sample_full": r2_in if spec_name == "full" else np.nan,
                "R2_LOO": float(r2_score(y, pred_cv)),
                "corr_LOO": float(np.corrcoef(y, pred_cv)[0, 1]) if np.std(pred_cv) > 0 else np.nan,
                "features": ",".join(spec_feats),
            })

    return pd.DataFrame(rows)


def make_figures(tab, summary):
    # Amplitude by condition for best mechanism
    best = "fate_all" if "fate_all" in tab["mechanism"].unique() else tab["mechanism"].iloc[0]
    g = tab[tab["mechanism"].eq(best)].sort_values(["rule", "L", "rho"])

    labels = [f"{r}\nL{L}\nρ{rho:.2f}" for r, L, rho in zip(g["rule"], g["L"], g["rho"])]

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(g)), g["amplitude"])
    plt.xticks(range(len(g)), labels, rotation=45, ha="right")
    plt.ylabel("Amplitude a_condition")
    plt.title(f"Condition amplitude for mechanism: {best}")
    plt.tight_layout()
    plt.savefig(OUT / "fig_amplitude_by_condition.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    for (rule, L), sub in g.groupby(["rule", "L"]):
        sub = sub.sort_values("rho")
        plt.plot(sub["rho"], sub["amplitude"], marker="o", label=f"{rule} L={L}")
    plt.xlabel("Density rho")
    plt.ylabel("Amplitude a_condition")
    plt.title(f"Amplitude vs density: {best}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "fig_amplitude_vs_density.png", dpi=200)
    plt.close()


def write_verdict(tab, summary):
    lines = []
    lines.append("CA Mechanism Amplitude-Law Test")
    lines.append("")
    lines.append("Question: can condition-dependent response amplitude be predicted from rule, L, and rho?")
    lines.append("")
    lines.append("Condition amplitude table summary:")
    for mech, g in tab.groupby("mechanism"):
        lines.append(
            f"  {mech:28s} mean_amp={g['amplitude'].mean():.4f}, "
            f"sd={g['amplitude'].std(ddof=1):.4f}, "
            f"CV={g['amplitude'].std(ddof=1)/(abs(g['amplitude'].mean())+1e-12):.3f}, "
            f"mean_within_R2_z={g['within_condition_R2_z'].mean():.4f}"
        )

    lines.append("")
    lines.append("Amplitude prediction models:")
    for _, row in summary.sort_values(["mechanism", "R2_LOO"], ascending=[True, False]).iterrows():
        lines.append(
            f"  {row['mechanism']:28s} {row['amplitude_model']:16s} "
            f"R2_LOO={row['R2_LOO']:.4f}, corr_LOO={row['corr_LOO']:.3f}, "
            f"amp_CV={row['amplitude_cv']:.3f}"
        )

    # Best per mechanism
    best = summary.sort_values("R2_LOO", ascending=False).groupby("mechanism").head(1)
    best_overall = best.iloc[0]

    lines.append("")
    if best_overall["R2_LOO"] > 0.5:
        lines.append("VERDICT: PASS")
        lines.append("Condition amplitude is substantially predictable from rule/L/rho covariates.")
    elif best_overall["R2_LOO"] > 0.2:
        lines.append("VERDICT: WEAK PASS")
        lines.append("Condition amplitude is partly predictable, but substantial calibration variance remains.")
    else:
        lines.append("VERDICT: CHECK/FAIL")
        lines.append("Condition amplitude is not well predicted by simple rule/L/rho covariates.")

    lines.append("")
    lines.append("Interpretation:")
    lines.append("The standardized mechanism already transfers. This test asks whether the raw scale factor also obeys a simple law.")
    lines.append("If this passes, ODD gets a two-layer structure: transferable mechanism + predictable condition amplitude.")
    lines.append("If this fails, the mechanism still transfers directionally, but raw amplitude calibration remains open.")

    (OUT / "amplitude_verdict.txt").write_text("\n".join(lines))


def main():
    if not FATE_RAW.exists():
        raise FileNotFoundError(FATE_RAW)

    df = pd.read_csv(FATE_RAW)
    df = residualize_within_condition(df)

    tab = estimate_condition_amplitudes(df)
    tab.to_csv(OUT / "amplitude_condition_table.csv", index=False)

    summary = fit_amplitude_law(tab)
    summary.to_csv(OUT / "amplitude_model_summary.csv", index=False)

    make_figures(tab, summary)
    write_verdict(tab, summary)

    print((OUT / "amplitude_verdict.txt").read_text())
    print("\nSaved outputs to:", OUT)


if __name__ == "__main__":
    main()
