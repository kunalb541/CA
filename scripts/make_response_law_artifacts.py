#!/usr/bin/env python3
"""
Artifact generator for the response-law paper.

Reads existing outputs/ and writes:
  paper/macros.tex   -- LaTeX macros with exact numbers
  paper/tables/      -- compact LaTeX tables
  paper/figures/     -- copies of key figures

Does NOT rerun simulations.

All macro names use only letters (no digits) to be valid LaTeX command names.
"""
from pathlib import Path
import shutil
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_PAPER = ROOT / "paper"
OUT_TABLES = OUT_PAPER / "tables"
OUT_FIGURES = OUT_PAPER / "figures"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGURES.mkdir(parents=True, exist_ok=True)


def copy_figures():
    fig_map = {
        "outputs/selection_principle_horizon/fig_beta_iso_vs_horizon.png": "fig_beta_iso_vs_horizon.png",
        "outputs/selection_principle_horizon/fig_residR2_vs_horizon.png": "fig_residR2_vs_horizon.png",
        "outputs/isolate_fate/fig_fate_r2.png": "fig_fate_r2.png",
        "outputs/isolate_fate/fig_fate_slopes.png": "fig_fate_slopes.png",
        "outputs/mechanism_transfer_standardized/fig_transfer_standardized_r2.png": "fig_transfer_standardized_r2.png",
        "outputs/mechanism_amplitude_law/fig_amplitude_by_condition.png": "fig_amplitude_by_condition.png",
        "outputs/mechanism_amplitude_law/fig_amplitude_vs_density.png": "fig_amplitude_vs_density.png",
        "outputs/selection_principle/fig_deltaR2_by_target.png": "fig_deltaR2_by_target.png",
        "outputs/prestate_class_horizon/fig_prestate_class_R2_vs_horizon.png": "fig_prestate_class_R2_vs_horizon.png",
    }
    for src_rel, dst_name in fig_map.items():
        src = ROOT / src_rel
        dst = OUT_FIGURES / dst_name
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  copied {src_rel} -> paper/figures/{dst_name}")
        else:
            print(f"  MISSING: {src_rel}")


def _get_horizon_row(df, k):
    """Return the global summary row for a given horizon k."""
    k_col = "k" if "k" in df.columns else "horizon"
    rows = df[df[k_col].astype(int) == k]
    return rows.iloc[0] if len(rows) else None


def write_macros():
    """Write paper/macros.tex with all exact numbers used in paper.tex.

    Macro naming convention: letters only, no digits.
      k=1   -> kone
      k=5   -> kfive
      k=10  -> kten
      k=25  -> ktwentyfive
      k=50  -> kfifty
      k=100 -> khundred
      k=200 -> ktwohundred
    """
    macros = {}

    # ------------------------------------------------------------------
    # Horizon global summary (mean slope, CV, R², null R² per horizon)
    # ------------------------------------------------------------------
    hg = ROOT / "outputs/selection_principle_horizon/horizon_global_summary.csv"
    if hg.exists():
        df = pd.read_csv(hg)
        slope_col = "mean_slope" if "mean_slope" in df.columns else "mean_raw_slope"
        cv_col    = "CV"         if "CV"         in df.columns else "cv_raw_slope"
        r2_col    = "mean_R2"    if "mean_R2"    in df.columns else "mean_resid_R2"
        null_col  = "null_R2"    if "null_R2"    in df.columns else "mean_null_R2"
        frac_col  = "frac_CI_neg" if "frac_CI_neg" in df.columns else "frac_slope_ci_negative"

        k_name = {1: "kone", 5: "kfive", 10: "kten", 25: "ktwentyfive",
                  50: "kfifty", 100: "khundred", 200: "ktwohundred"}
        for k, name in k_name.items():
            row = _get_horizon_row(df, k)
            if row is not None:
                macros[f"horizon{name}"]  = f"{row[slope_col]:.3f}"
                macros[f"horizonR{name}"] = f"{row[r2_col]:.3f}"

        # Fraction CI negative at k=1 (representative; is 1.0 everywhere)
        row1 = _get_horizon_row(df, 1)
        if row1 is not None and frac_col in df.columns:
            macros["horizonnegfrac"] = f"{row1[frac_col]:.2f}"
        else:
            macros["horizonnegfrac"] = "1.00"

    # ------------------------------------------------------------------
    # Horizon rule summary (GoL / HighLife per-rule averages)
    # ------------------------------------------------------------------
    hr = ROOT / "outputs/selection_principle_horizon/horizon_rule_summary.csv"
    if hr.exists():
        df = pd.read_csv(hr)
        slope_col = "mean_raw_slope" if "mean_raw_slope" in df.columns else "mean_slope"
        cv_col    = "cv_raw_slope"   if "cv_raw_slope"   in df.columns else "CV"
        r2_col    = "mean_resid_R2"  if "mean_resid_R2"  in df.columns else "mean_R2"
        null_col  = "mean_null_R2"   if "mean_null_R2"   in df.columns else "null_R2"

        for rule, prefix in [("GoL", "gol"), ("HighLife", "hl")]:
            sub = df[df["rule"] == rule]
            if len(sub):
                macros[f"{prefix}resiR"]    = f"{sub[r2_col].mean():.3f}"
                macros[f"{prefix}slopeCV"]  = f"{sub[cv_col].mean():.3f}"
                macros[f"{prefix}slopemean"] = f"{sub[slope_col].mean():.3f}"
                macros[f"{prefix}nullresiR"] = f"{sub[null_col].mean():.4f}"

        # Global averages
        macros["globalresiR"] = f"{df[r2_col].mean():.3f}"
        macros["nullresiR"]   = f"{df[null_col].mean():.4f}"

    # ------------------------------------------------------------------
    # Fate mechanism: all_fates CV R²
    # ------------------------------------------------------------------
    fg = ROOT / "outputs/isolate_fate/fate_global_summary.csv"
    if fg.exists():
        df = pd.read_csv(fg)
        model_col = "model" if "model" in df.columns else df.columns[1]
        cv_col    = "mean_cv_R2" if "mean_cv_R2" in df.columns else "cv_r2"
        if cv_col in df.columns:
            for mname, key in [("all_fates",    "allfatesR"),
                                ("iso_count",    "isocountR"),
                                ("local_window", "localwindowR"),
                                ("local_window", "localwinR")]:
                rows = df[(df[model_col] == mname) & df[cv_col].notna()]
                if len(rows):
                    macros[key] = f"{rows.iloc[0][cv_col]:.3f}"

    # ------------------------------------------------------------------
    # Amplitude law
    # ------------------------------------------------------------------
    am = ROOT / "outputs/mechanism_amplitude_law/amplitude_model_summary.csv"
    if am.exists():
        df = pd.read_csv(am)
        mech_col = "mechanism"      if "mechanism"      in df.columns else "feature"
        mod_col  = "amplitude_model" if "amplitude_model" in df.columns else "model"
        cv_col   = "amplitude_cv"   if "amplitude_cv"   in df.columns else None

        for mod, key in [("full",      "ampLOOfull"),
                         ("size_rho",  "ampLOOsizerho"),
                         ("size_only", "ampLOOsizeonly"),
                         ("rule_only", "ampLOOruleonly")]:
            mask = (df[mech_col] == "iso_count") & (df[mod_col] == mod)
            row = df[mask]
            if len(row):
                macros[key] = f"{row.iloc[0]['R2_LOO']:.3f}"

        # Amplitude CV (condition-to-condition spread)
        if cv_col and cv_col in df.columns:
            mask_full = (df[mech_col] == "iso_count") & (df[mod_col] == "full")
            row_full = df[mask_full]
            if len(row_full):
                macros["ampCV"] = f"{row_full.iloc[0][cv_col]:.3f}"
        else:
            # amplitude_cv is constant across all rows for iso_count — read it
            mask_any = df[mech_col] == "iso_count"
            row_any = df[mask_any]
            if len(row_any) and "amplitude_cv" in df.columns:
                macros["ampCV"] = f"{row_any.iloc[0]['amplitude_cv']:.3f}"

    # ------------------------------------------------------------------
    # LGDS bridge
    # ------------------------------------------------------------------
    bs = ROOT / "outputs/ca_lgds_bridge/bridge_summary.csv"
    if bs.exists():
        df = pd.read_csv(bs)
        for fam, cos_key, r1_key in [
            ("horizon_fine_net",       "horizonCos", "horizonRankone"),
            ("selection_multi_target", "selCos",     "selRankone"),
        ]:
            row = df[df["family"] == fam]
            if len(row):
                r = row.iloc[0]
                macros[cos_key] = f"{r['mean_pairwise_abs_cosine']:.3f}"
                macros[r1_key]  = f"{r['rank1_cumulative_energy']:.3f}"

    # Regret from verdict file (already parsed at top of session)
    br = ROOT / "outputs/ca_lgds_bridge/bridge_verdict.txt"
    if br.exists():
        text = br.read_text()
        import re
        # horizon fine_net: mean rank-1 relative regret = 0.001
        m = re.search(r"horizon_fine_net:.*?mean rank-1 relative regret = ([\d.]+)", text, re.DOTALL)
        if m:
            macros["horizonRegret"] = m.group(1)
        # selection multi-target: mean rank-1 relative regret = 0.596
        m = re.search(r"selection_multi_target:.*?mean rank-1 relative regret = ([\d.]+)", text, re.DOTALL)
        if m:
            macros["selRegret"] = m.group(1)

    # ------------------------------------------------------------------
    # Prestate class horizon
    # ------------------------------------------------------------------
    ps = ROOT / "outputs/prestate_class_horizon/prestate_class_summary.csv"
    if ps.exists():
        df = pd.read_csv(ps)
        iso = df[df["model"] == "iso_count"]
        if len(iso):
            macros["prestateIsoMean"]    = f"{iso['mean_R2'].mean():.3f}"
            macros["prestateIsoMin"]     = f"{iso['mean_R2'].min():.3f}"
            macros["prestateIsoShuffle"] = f"{iso['mean_shuffle_R2'].mean():.4f}"

    # ------------------------------------------------------------------
    # Write macros.tex
    # ------------------------------------------------------------------
    out = OUT_PAPER / "macros.tex"
    lines = ["% Auto-generated by scripts/make_response_law_artifacts.py\n",
             "% All command names use letters only (no digits) — valid LaTeX.\n"]
    for k, v in macros.items():
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    out.write_text("".join(lines))
    print(f"  wrote {len(macros)} macros to paper/macros.tex")


def write_horizon_table():
    hg = ROOT / "outputs/selection_principle_horizon/horizon_global_summary.csv"
    if not hg.exists():
        print("  MISSING horizon_global_summary.csv"); return
    df = pd.read_csv(hg)
    slope_col = "mean_slope"    if "mean_slope"    in df.columns else "mean_raw_slope"
    cv_col    = "CV"            if "CV"            in df.columns else "cv_raw_slope"
    r2_col    = "mean_R2"       if "mean_R2"       in df.columns else "mean_resid_R2"
    null_col  = "null_R2"       if "null_R2"       in df.columns else "mean_null_R2"
    k_col     = "k"             if "k"             in df.columns else "horizon"

    lines = [
        "% Auto-generated horizon table\n",
        "\\begin{tabular}{rrrrr}\n",
        "\\toprule\n",
        "$k$ & $\\bar{\\beta}_{\\rm iso}$ & CV & $\\bar{R}^2$ & Null $R^2$ \\\\\n",
        "\\midrule\n",
    ]
    for _, row in df.iterrows():
        k     = int(row[k_col])
        slope = row.get(slope_col, float("nan"))
        cv    = row.get(cv_col,    float("nan"))
        r2    = row.get(r2_col,    float("nan"))
        null  = row.get(null_col,  float("nan"))
        lines.append(f"  {k} & {slope:.3f} & {cv:.3f} & {r2:.3f} & {null:.4f} \\\\\n")
    lines += ["\\bottomrule\n", "\\end{tabular}\n"]
    (OUT_TABLES / "tab_horizon.tex").write_text("".join(lines))
    print("  wrote paper/tables/tab_horizon.tex")


def write_mechanism_table():
    fg = ROOT / "outputs/isolate_fate/fate_global_summary.csv"
    if not fg.exists():
        print("  MISSING fate_global_summary.csv"); return
    df = pd.read_csv(fg)
    model_col = "model" if "model" in df.columns else df.columns[1]
    cv_col    = "mean_cv_R2" if "mean_cv_R2" in df.columns else "cv_r2"
    slope_col = "mean_slope_raw" if "mean_slope_raw" in df.columns else "mean_slope"

    lines = [
        "% Auto-generated mechanism table\n",
        "\\begin{tabular}{lrr}\n",
        "\\toprule\n",
        "Feature & CV $R^2$ & Slope \\\\\n",
        "\\midrule\n",
    ]
    for _, row in df.iterrows():
        feat   = str(row[model_col])
        cv_r2  = row.get(cv_col,    float("nan"))
        slope  = row.get(slope_col, float("nan"))
        if pd.isna(cv_r2) and pd.isna(slope):
            continue  # skip blank rows
        cv_str    = f"{cv_r2:.3f}"  if not pd.isna(cv_r2)  else "---"
        slope_str = f"{slope:.3f}"  if not pd.isna(slope)   else "---"
        lines.append(f"  {feat} & {cv_str} & {slope_str} \\\\\n")
    lines += ["\\bottomrule\n", "\\end{tabular}\n"]
    (OUT_TABLES / "tab_mechanism.tex").write_text("".join(lines))
    print("  wrote paper/tables/tab_mechanism.tex")


if __name__ == "__main__":
    print("Generating paper artifacts...")
    copy_figures()
    write_macros()
    write_horizon_table()
    write_mechanism_table()
    print("Done.")
