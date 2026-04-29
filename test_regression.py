"""
test_regression.py
==================
Regression tests for ca.py core computations and stored analysis outputs.

Run with:
    pytest test_regression.py -v

Tests cover:
  1. Periodic (torus) BFS component counting
  2. Embedded / alone cell classification
  3. Narrative gap (G) computation
  4. GoL step correctness on known patterns
  5. Stored study outputs (study A/B/C/D)
  6. Response-law analysis outputs (selection, horizon, mechanism, amplitude,
     transfer-standardized, LGDS bridge, prestate class)
  7. Artifact generator correctness
"""

import json
import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Make sure ca.py is importable from the same directory as this test file
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from ca import (
    comp_count_periodic,
    comp_count_nonperiodic_4,
    block_count,
    embedded_isolated_coords,
    isolated_counts,
    gol_step,
)

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = str(REPO_ROOT / "outputs" / "data")
STATS_EXIST = os.path.isdir(DATA_DIR)
OUTPUTS_ROOT = REPO_ROOT / "outputs"


# ===========================================================================
# 1. Periodic BFS component counting
# ===========================================================================

class TestCompCountPeriodic:
    def test_empty_grid(self):
        g = np.zeros((8, 8), dtype=np.uint8)
        assert comp_count_periodic(g) == 0

    def test_single_cell(self):
        g = np.zeros((8, 8), dtype=np.uint8)
        g[3, 3] = 1
        assert comp_count_periodic(g) == 1

    def test_two_disconnected_cells(self):
        g = np.zeros((8, 8), dtype=np.uint8)
        g[0, 0] = 1
        g[4, 4] = 1
        assert comp_count_periodic(g) == 2

    def test_horizontal_strip_one_component(self):
        g = np.zeros((8, 8), dtype=np.uint8)
        g[0, :] = 1  # full row
        assert comp_count_periodic(g) == 1

    def test_periodic_wrap_merges_components(self):
        """
        Two cells on opposite edges of the same row are one component on
        the torus but two components on a flat grid.
        """
        g = np.zeros((8, 8), dtype=np.uint8)
        g[4, 0] = 1  # left edge
        g[4, 7] = 1  # right edge — 4-connected neighbours via wrap
        assert comp_count_periodic(g) == 1, (
            "Cells at column 0 and column 7 on the same row should merge "
            "into one component on the torus"
        )

    def test_periodic_wrap_merges_top_bottom(self):
        g = np.zeros((8, 8), dtype=np.uint8)
        g[0, 3] = 1
        g[7, 3] = 1
        assert comp_count_periodic(g) == 1

    def test_nonperiodic_keeps_two_components(self):
        """
        The same pair of cells that merge periodically stay separate
        under the non-periodic counter.
        """
        g = np.zeros((8, 8), dtype=np.uint8)
        g[4, 0] = 1
        g[4, 7] = 1
        assert comp_count_nonperiodic_4(g) == 2

    def test_diagonal_not_4connected(self):
        """Diagonal adjacency does not count as 4-connectivity."""
        g = np.zeros((4, 4), dtype=np.uint8)
        g[0, 0] = 1
        g[1, 1] = 1
        assert comp_count_periodic(g) == 2

    def test_known_glider_component_count(self):
        """
        The canonical GoL glider (SE-moving) has exactly 2 components under
        4-connectivity: the top cell is only diagonally adjacent to the rest.
        Standard glider pattern at rows 1-3:
          .X.   <- (1,2) is NOT 4-connected to (2,3)
          ..X
          XXX
        """
        g = np.zeros((10, 10), dtype=np.uint8)
        g[1, 2] = 1
        g[2, 3] = 1
        g[3, 1] = 1
        g[3, 2] = 1
        g[3, 3] = 1
        # (1,2) is isolated from the rest under 4-connectivity → 2 components
        assert comp_count_periodic(g) == 2

    def test_i_tetromino_one_component(self):
        """Four cells in a row form a single 4-connected component."""
        g = np.zeros((8, 8), dtype=np.uint8)
        g[4, 1] = g[4, 2] = g[4, 3] = g[4, 4] = 1
        assert comp_count_periodic(g) == 1

    def test_two_separate_still_lifes(self):
        """
        Two 2×2 blocks separated by at least one cell gap are
        two components.
        """
        g = np.zeros((10, 10), dtype=np.uint8)
        g[1, 1] = g[1, 2] = g[2, 1] = g[2, 2] = 1  # block 1
        g[1, 6] = g[1, 7] = g[2, 6] = g[2, 7] = 1  # block 2
        assert comp_count_periodic(g) == 2


# ===========================================================================
# 2. Embedded / alone cell classification
# ===========================================================================

class TestIsolatedCounts:
    def _grid(self, shape=(8, 8)):
        return np.zeros(shape, dtype=np.uint8)

    def test_alone_cell_no_neighbours(self):
        """A cell with no 8-connected neighbours is 'alone'."""
        g = self._grid()
        g[4, 4] = 1
        counts = isolated_counts(g)
        assert counts["alone"] == 1
        assert counts["embedded"] == 0

    def test_embedded_cell_has_diagonal_only(self):
        """A cell with no 4-connected but ≥1 diagonal neighbour is 'embedded'."""
        g = self._grid()
        g[4, 4] = 1   # focal cell
        g[3, 3] = 1   # diagonal neighbour
        counts = isolated_counts(g)
        # g[4,4] is embedded (diagonal only); g[3,3] also has only a diagonal
        # neighbour, so it too is embedded
        assert counts["embedded"] == 2
        assert counts["alone"] == 0

    def test_cell_with_4connected_not_isolated(self):
        """A cell with a 4-connected live neighbour is neither embedded nor alone."""
        g = self._grid()
        g[4, 4] = 1
        g[4, 5] = 1   # 4-connected
        counts = isolated_counts(g)
        assert counts["embedded"] == 0
        assert counts["alone"] == 0

    def test_embedded_coords_matches_counts(self):
        g = self._grid()
        g[4, 4] = 1
        g[3, 3] = 1
        coords = embedded_isolated_coords(g)
        counts = isolated_counts(g)
        assert len(coords) == counts["embedded"]

    def test_periodic_diagonal_detection(self):
        """
        A cell at corner (0,0) with its diagonal neighbour at (7,7)
        (i.e. via torus wrap) should be classified as embedded.
        """
        g = self._grid((8, 8))
        g[0, 0] = 1
        g[7, 7] = 1   # diagonal via torus (row -1, col -1)
        counts = isolated_counts(g)
        assert counts["embedded"] == 2
        assert counts["alone"] == 0

    def test_periodic_4connected_not_isolated(self):
        """
        A cell at (0, 0) with a 4-connected neighbour at (0, 7) via wrap
        should NOT be classified as isolated.
        """
        g = self._grid((8, 8))
        g[0, 0] = 1
        g[0, 7] = 1   # same row, opposite edge — 4-connected via wrap
        counts = isolated_counts(g)
        assert counts["embedded"] == 0
        assert counts["alone"] == 0


# ===========================================================================
# 3. Narrative gap computation
# ===========================================================================

class TestNarrativeGap:
    def test_identical_observers_zero_gap(self):
        """
        If fine and coarse normalised net changes are equal, G = 0.
        Construct a case where net_F / c0_F == net_C / c0_C.
        """
        # Use a tiny hand-crafted example: G = rho_C - rho_F
        net_F, c0_F = 10, 100     # rho_F = 0.10
        net_C, c0_C = 5, 50       # rho_C = 0.10
        G = net_C / max(c0_C, 1) - net_F / max(c0_F, 1)
        assert abs(G) < 1e-12

    def test_positive_gap_coarse_creates_more(self):
        net_F, c0_F = -10, 100    # rho_F = -0.10
        net_C, c0_C = 5, 50       # rho_C = +0.10
        G = net_C / max(c0_C, 1) - net_F / max(c0_F, 1)
        assert G > 0

    def test_negative_gap_fine_creates_more(self):
        net_F, c0_F = 10, 100     # rho_F = +0.10
        net_C, c0_C = -5, 50      # rho_C = -0.10
        G = net_C / max(c0_C, 1) - net_F / max(c0_F, 1)
        assert G < 0

    def test_empty_world_zero_gap(self):
        """
        An all-dead grid remains all-dead. Both observers record zero change.
        G = 0.
        """
        g = np.zeros((16, 16), dtype=np.uint8)
        g_next = gol_step(g)
        assert g_next.sum() == 0
        c_fine_t0 = comp_count_periodic(g)
        c_fine_t1 = comp_count_periodic(g_next)
        c_coarse_t0 = block_count(g, 8)
        c_coarse_t1 = block_count(g_next, 8)
        G = (c_coarse_t1 - c_coarse_t0) / max(c_coarse_t0, 1) - \
            (c_fine_t1 - c_fine_t0) / max(c_fine_t0, 1)
        assert G == 0.0

    def test_still_life_zero_gap(self):
        """
        A 2×2 block is a GoL still life. After one step it is unchanged.
        Both observers record zero net change; G = 0.
        """
        g = np.zeros((8, 8), dtype=np.uint8)
        g[3, 3] = g[3, 4] = g[4, 3] = g[4, 4] = 1
        g2 = gol_step(g)
        np.testing.assert_array_equal(g, g2)
        c_fine_t0 = comp_count_periodic(g)
        c_fine_t1 = comp_count_periodic(g2)
        c_coarse_t0 = block_count(g, 8)
        c_coarse_t1 = block_count(g2, 8)
        G = (c_coarse_t1 - c_coarse_t0) / max(c_coarse_t0, 1) - \
            (c_fine_t1 - c_fine_t0) / max(c_fine_t0, 1)
        assert G == 0.0


# ===========================================================================
# 4. GoL step correctness on known patterns
# ===========================================================================

class TestGolStep:
    def test_still_life_block(self):
        g = np.zeros((6, 6), dtype=np.uint8)
        g[2, 2] = g[2, 3] = g[3, 2] = g[3, 3] = 1
        np.testing.assert_array_equal(gol_step(g), g)

    def test_blinker_period_2(self):
        """A period-2 oscillator returns to original after two steps."""
        g = np.zeros((5, 5), dtype=np.uint8)
        g[2, 1] = g[2, 2] = g[2, 3] = 1
        g2 = gol_step(gol_step(g))
        np.testing.assert_array_equal(g, g2)

    def test_birth_rule(self):
        """Dead cell with exactly 3 live neighbours is born."""
        g = np.zeros((5, 5), dtype=np.uint8)
        # Three live cells around (2, 2): above, left, below
        g[1, 2] = g[3, 2] = g[2, 1] = 1
        g2 = gol_step(g)
        assert g2[2, 2] == 1

    def test_overpopulation(self):
        """Live cell with 4 live neighbours dies."""
        g = np.zeros((5, 5), dtype=np.uint8)
        g[2, 2] = 1   # focal
        g[1, 2] = g[3, 2] = g[2, 1] = g[2, 3] = 1  # 4 neighbours
        g2 = gol_step(g)
        assert g2[2, 2] == 0


# ===========================================================================
# 5. Regression tests against stored outputs
# ===========================================================================

@pytest.mark.skipif(not STATS_EXIST, reason="outputs/data not found — run ca.py first")
class TestStoredStudyA:
    def setup_method(self):
        with open(os.path.join(DATA_DIR, "study_A_stats.json")) as f:
            self.stats = json.load(f)

    def test_partial_r_magnitude(self):
        pr = abs(self.stats["partial_r"])
        assert pr >= 0.80, f"partial r magnitude {pr:.3f} unexpectedly low"

    def test_partial_r_sign(self):
        assert self.stats["partial_r"] < 0, "partial r should be negative"

    def test_ci_excludes_zero(self):
        lo, hi = self.stats["partial_r_boot_ci95"]
        assert hi < 0, f"CI upper bound {hi:.3f} should be < 0"

    def test_raw_r_sign(self):
        assert self.stats["r_main"] < 0


@pytest.mark.skipif(not STATS_EXIST, reason="outputs/data not found — run ca.py first")
class TestStoredStudyB:
    def setup_method(self):
        with open(os.path.join(DATA_DIR, "study_B_stats.json")) as f:
            self.stats = json.load(f)

    def test_negative_controls_clean(self):
        n_pos = self.stats["negative_control_n_positive"]
        assert n_pos == 0, (
            f"{n_pos} primary-target static/early conditions exceeded R²=0.04"
        )

    def test_nlive_peak_at_fine_scale(self):
        assert self.stats["peak_B"]["Nlive"]["point"] <= 2

    def test_nocc8_peak_at_coarse_scale(self):
        assert self.stats["peak_B"]["Nocc8"]["point"] >= 4

    def test_ci_no_overlap(self):
        nlive_hi = self.stats["peak_B"]["Nlive"]["ci_hi"]
        nocc8_lo = self.stats["peak_B"]["Nocc8"]["ci_lo"]
        assert nlive_hi < nocc8_lo, (
            f"Bootstrap CIs overlap: Nlive hi={nlive_hi}, Nocc8 lo={nocc8_lo}"
        )


@pytest.mark.skipif(not STATS_EXIST, reason="outputs/data not found — run ca.py first")
class TestStoredStudyD:
    def setup_method(self):
        with open(os.path.join(DATA_DIR, "study_D_stats.json")) as f:
            self.stats = json.load(f)

    def test_cv_below_threshold(self):
        cv = self.stats["beta_emp_cv"]
        assert cv <= 0.20, f"CV = {cv:.3f} exceeds stability threshold 0.20"

    def test_mean_slope_negative(self):
        assert self.stats["beta_emp_mean"] < 0

    def test_mean_slope_approx(self):
        slope = self.stats["beta_emp_mean"]
        assert -2.0 <= slope <= -1.0, (
            f"Mean slope {slope:.3f} outside expected range [-2.0, -1.0]"
        )

    def test_chi_all_negative(self):
        """χ < 0 in all conditions: residual coupling claim fails."""
        assert self.stats["chi_mean"] < 0
        assert self.stats["chi_min"] < 0


@pytest.mark.skipif(not STATS_EXIST, reason="outputs/data not found — run ca.py first")
class TestStoredStudyC:
    def setup_method(self):
        with open(os.path.join(DATA_DIR, "study_C_stats.json")) as f:
            self.stats = json.load(f)

    def test_mediation_fails(self):
        med = self.stats["mediation"]
        frac = med["mediation_fraction"]
        assert frac < 0.50, f"Mediation fraction {frac:.3f} should be < 0.50"

    def test_mediation_ci_below_zero(self):
        med = self.stats["mediation"]
        assert med["boot_ci95_hi"] < 0, (
            "Mediation CI upper bound should be negative"
        )

    def test_embedded_stronger_than_alone(self):
        pr_emb = abs(self.stats["pairwise"]["partial_r_embedded_G"])
        pr_aln = abs(self.stats["pairwise"]["partial_r_alone_G"])
        assert pr_emb > pr_aln, (
            "Embedded partial r should exceed alone partial r"
        )


# ===========================================================================
# 6. Response-law analysis outputs
# ===========================================================================

SELECTION_VERDICT = OUTPUTS_ROOT / "selection_principle" / "selection_verdict.txt"
HORIZON_VERDICT = OUTPUTS_ROOT / "selection_principle_horizon" / "horizon_verdict.txt"
FATE_VERDICT = OUTPUTS_ROOT / "isolate_fate" / "fate_verdict.txt"
TRANSFER_STD_VERDICT = OUTPUTS_ROOT / "mechanism_transfer_standardized" / "transfer_standardized_verdict.txt"
AMPLITUDE_VERDICT = OUTPUTS_ROOT / "mechanism_amplitude_law" / "amplitude_verdict.txt"
BRIDGE_VERDICT = OUTPUTS_ROOT / "ca_lgds_bridge" / "bridge_verdict.txt"
PRESTATE_VERDICT = OUTPUTS_ROOT / "prestate_class_horizon" / "prestate_class_verdict.txt"

HORIZON_GLOBAL_CSV = OUTPUTS_ROOT / "selection_principle_horizon" / "horizon_global_summary.csv"
FATE_GLOBAL_CSV = OUTPUTS_ROOT / "isolate_fate" / "fate_global_summary.csv"
AMPLITUDE_MODEL_CSV = OUTPUTS_ROOT / "mechanism_amplitude_law" / "amplitude_model_summary.csv"
TRANSFER_STD_CSV = OUTPUTS_ROOT / "mechanism_transfer_standardized" / "transfer_standardized_summary.csv"
BRIDGE_SUMMARY_CSV = OUTPUTS_ROOT / "ca_lgds_bridge" / "bridge_summary.csv"
PRESTATE_SUMMARY_CSV = OUTPUTS_ROOT / "prestate_class_horizon" / "prestate_class_summary.csv"
SELECTION_SUMMARY_CSV = OUTPUTS_ROOT / "selection_principle" / "selection_summary.csv"


def _outputs_exist(*paths):
    return all(p.exists() for p in paths)


@pytest.mark.skipif(
    not _outputs_exist(SELECTION_VERDICT, SELECTION_SUMMARY_CSV),
    reason="selection_principle outputs not found",
)
class TestSelectionPrinciple:
    """Target-specific selection: iso_count adds ΔR² to fine-net only."""

    def test_gol_verdict_pass(self):
        text = SELECTION_VERDICT.read_text()
        assert "GoL pooled: PASS" in text, "GoL selection verdict should be PASS"

    def test_global_verdict_pass(self):
        text = SELECTION_VERDICT.read_text()
        assert "Global pooled: PASS" in text, "Global selection verdict should be PASS"

    def test_fine_net_deltaR2_positive(self):
        """iso_count adds positive ΔR² for the fine-net target."""
        df = pd.read_csv(SELECTION_SUMMARY_CSV)
        fine = df[df["target"] == "target_fine_net"]
        assert len(fine) > 0, "No fine_net rows in selection_summary.csv"
        assert (fine["deltaR2_iso"] > 0).all(), (
            f"All fine_net deltaR2_iso should be positive; got {fine['deltaR2_iso'].tolist()}"
        )

    def test_iso_slope_negative_for_fine_net(self):
        """Mean slope of iso_count on fine-net residual is strongly negative;
        at least 80% of individual conditions also show a negative slope."""
        df = pd.read_csv(SELECTION_SUMMARY_CSV)
        fine = df[df["target"] == "target_fine_net"]
        mean_slope = fine["iso_slope_standardized"].mean()
        frac_neg = (fine["iso_slope_standardized"] < 0).mean()
        assert mean_slope < -0.05, (
            f"Mean iso_slope for fine_net = {mean_slope:.4f} should be < -0.05"
        )
        assert frac_neg >= 0.80, (
            f"Only {frac_neg:.0%} of fine_net conditions have negative iso_slope "
            "(expected ≥ 80%)"
        )

    def test_null_deltaR2_near_zero(self):
        """Shuffled-iso null ΔR² is near zero — no spurious signal."""
        df = pd.read_csv(SELECTION_SUMMARY_CSV)
        fine = df[df["target"] == "target_fine_net"]
        mean_null = fine["deltaR2_null_iso_shuffle"].mean()
        assert abs(mean_null) < 0.01, (
            f"Mean null ΔR² = {mean_null:.4f} should be near zero"
        )


@pytest.mark.skipif(
    not _outputs_exist(HORIZON_VERDICT, HORIZON_GLOBAL_CSV),
    reason="selection_principle_horizon outputs not found",
)
class TestHorizonResponse:
    """Temporal response law: β_iso(k) < 0 stable across all horizons k."""

    def setup_method(self):
        self.df = pd.read_csv(HORIZON_GLOBAL_CSV)
        self.text = HORIZON_VERDICT.read_text()

    def test_gol_verdict_pass(self):
        assert "GoL: PASS" in self.text

    def test_highlife_verdict_pass(self):
        assert "HighLife: PASS" in self.text

    def test_all_slopes_negative(self):
        """Every horizon row should have a negative mean slope."""
        slope_col = "mean_slope" if "mean_slope" in self.df.columns else "mean_raw_slope"
        assert (self.df[slope_col] < 0).all(), (
            f"All horizon slopes should be negative; got {self.df[slope_col].tolist()}"
        )

    def test_slope_range(self):
        """Mean slopes across horizons should be in [−0.95, −0.60]."""
        slope_col = "mean_slope" if "mean_slope" in self.df.columns else "mean_raw_slope"
        for _, row in self.df.iterrows():
            k = int(row.get("k", row.get("horizon", -1)))
            s = row[slope_col]
            assert -0.95 <= s <= -0.60, (
                f"k={k}: slope {s:.3f} outside expected range [−0.95, −0.60]"
            )

    def test_cv_below_threshold(self):
        """CV of slopes across conditions < 0.25 at every horizon."""
        cv_col = "CV" if "CV" in self.df.columns else "cv_raw_slope"
        for _, row in self.df.iterrows():
            k = int(row.get("k", row.get("horizon", -1)))
            cv = row[cv_col]
            assert cv < 0.25, (
                f"k={k}: CV = {cv:.3f} exceeds 0.25 — slopes not stable across conditions"
            )

    def test_all_horizons_present(self):
        """All 7 canonical horizons [1,5,10,25,50,100,200] are represented."""
        k_col = "k" if "k" in self.df.columns else "horizon"
        found = set(self.df[k_col].astype(int).tolist())
        expected = {1, 5, 10, 25, 50, 100, 200}
        assert expected.issubset(found), (
            f"Missing horizons: {expected - found}"
        )

    def test_R2_above_floor(self):
        """Mean residual R² > 0.15 at every horizon (robust predictive signal)."""
        r2_col = "mean_R2" if "mean_R2" in self.df.columns else "mean_resid_R2"
        for _, row in self.df.iterrows():
            k = int(row.get("k", row.get("horizon", -1)))
            r2 = row[r2_col]
            assert r2 > 0.15, (
                f"k={k}: mean R² = {r2:.3f} below floor 0.15"
            )

    def test_null_R2_near_zero(self):
        """Shuffled null R² < 0.005 at every horizon."""
        null_col = "null_R2" if "null_R2" in self.df.columns else "mean_null_R2"
        for _, row in self.df.iterrows():
            k = int(row.get("k", row.get("horizon", -1)))
            null = row[null_col]
            assert null < 0.005, (
                f"k={k}: null R² = {null:.4f} suspiciously high"
            )


@pytest.mark.skipif(
    not _outputs_exist(FATE_VERDICT, FATE_GLOBAL_CSV),
    reason="isolate_fate outputs not found",
)
class TestFateMechanism:
    """Local component-context loss mechanism (not simple cell death)."""

    def setup_method(self):
        self.df = pd.read_csv(FATE_GLOBAL_CSV)
        self.text = FATE_VERDICT.read_text()
        # Only the global_model rows have CV R²
        cv_col = "mean_cv_R2" if "mean_cv_R2" in self.df.columns else "cv_r2"
        model_col = "model" if "model" in self.df.columns else "predictor"
        self.model_df = self.df[self.df[cv_col].notna()].copy()
        self.cv_col = cv_col
        self.model_col = model_col

    def _r2(self, name):
        row = self.model_df[self.model_df[self.model_col] == name]
        assert len(row) == 1, f"Model '{name}' not found in fate_global_summary"
        return float(row.iloc[0][self.cv_col])

    def test_verdict_pass(self):
        assert "VERDICT: PASS" in self.text

    def test_local_window_beats_iso_count(self):
        """Local-window loss R² > iso_count R²: mechanism is contextual."""
        assert self._r2("local_window") > self._r2("iso_count"), (
            "local_window CV R² should exceed iso_count"
        )

    def test_iso_count_beats_coarse(self):
        """iso_count R² >> coarse baseline: signal is object-specific."""
        assert self._r2("iso_count") > 5 * self._r2("coarse"), (
            "iso_count should be >> coarse baseline"
        )

    def test_iso_count_r2_range(self):
        r2 = self._r2("iso_count")
        assert 0.30 <= r2 <= 0.45, f"iso_count CV R² = {r2:.3f} outside [0.30, 0.45]"

    def test_local_window_r2_range(self):
        r2 = self._r2("local_window")
        assert 0.48 <= r2 <= 0.62, f"local_window CV R² = {r2:.3f} outside [0.48, 0.62]"

    def test_all_fates_r2_range(self):
        r2 = self._r2("all_fates")
        assert 0.50 <= r2 <= 0.62, f"all_fates CV R² = {r2:.3f} outside [0.50, 0.62]"

    def test_entropy_baseline_low(self):
        r2 = self._r2("entropy")
        assert r2 < 0.02, f"entropy baseline R² = {r2:.4f} should be < 0.02"


@pytest.mark.skipif(
    not _outputs_exist(TRANSFER_STD_VERDICT, TRANSFER_STD_CSV),
    reason="mechanism_transfer_standardized outputs not found",
)
class TestMechanismTransferStandardized:
    """Standardized mechanism transfers across density, size, rule, condition."""

    def setup_method(self):
        self.df = pd.read_csv(TRANSFER_STD_CSV)
        self.text = TRANSFER_STD_VERDICT.read_text()

    def test_verdict_pass(self):
        assert "VERDICT: PASS" in self.text

    def test_all_frac_R2_positive_unity(self):
        """frac_R2_positive = 1.0 for every split/model combo."""
        frac_col = "frac_R2_positive"
        if frac_col in self.df.columns:
            bad = self.df[self.df[frac_col] < 1.0]
            assert len(bad) == 0, (
                f"{len(bad)} split/model combos have frac_R2_positive < 1.0:\n{bad[['split_type','model',frac_col]].to_string()}"
            )

    def test_fate_all_mean_R2_z_above_floor(self):
        """fate_all mean_R2_z > 0.50 for every leave-one-out split."""
        sub = self.df[self.df["model"] == "fate_all"]
        r2_col = "mean_test_R2_z" if "mean_test_R2_z" in self.df.columns else "mean_R2_z"
        for _, row in sub.iterrows():
            split = row.get("split_type", "?")
            r2z = row[r2_col]
            assert r2z > 0.50, (
                f"fate_all mean_R2_z = {r2z:.3f} < 0.50 for split={split}"
            )

    def test_iso_count_mean_R2_z_positive(self):
        """Even the scalar iso_count has positive mean_R2_z on every split."""
        sub = self.df[self.df["model"] == "iso_count"]
        r2_col = "mean_test_R2_z" if "mean_test_R2_z" in self.df.columns else "mean_R2_z"
        for _, row in sub.iterrows():
            split = row.get("split_type", "?")
            r2z = row[r2_col]
            assert r2z > 0.30, (
                f"iso_count mean_R2_z = {r2z:.3f} < 0.30 for split={split}"
            )


@pytest.mark.skipif(
    not _outputs_exist(AMPLITUDE_VERDICT, AMPLITUDE_MODEL_CSV),
    reason="mechanism_amplitude_law outputs not found",
)
class TestAmplitudeLaw:
    """Two-layer amplitude structure: mechanism × predictable amplitude A(L,ρ)."""

    def setup_method(self):
        self.df = pd.read_csv(AMPLITUDE_MODEL_CSV)
        self.text = AMPLITUDE_VERDICT.read_text()

    def _loo_r2(self, mechanism, amp_model):
        mech_col = "mechanism" if "mechanism" in self.df.columns else "feature"
        mod_col = "amplitude_model" if "amplitude_model" in self.df.columns else "model"
        row = self.df[(self.df[mech_col] == mechanism) & (self.df[mod_col] == amp_model)]
        assert len(row) == 1, f"Row ({mechanism}, {amp_model}) not found in amplitude_model_summary"
        return float(row.iloc[0]["R2_LOO"])

    def test_verdict_pass(self):
        assert "VERDICT: PASS" in self.text

    def test_iso_count_full_loo_r2(self):
        """iso_count full LOO R² should be ≈ 0.977 (> 0.95)."""
        r2 = self._loo_r2("iso_count", "full")
        assert r2 > 0.95, f"iso_count full LOO R² = {r2:.3f} should be > 0.95"

    def test_iso_count_size_rho_loo_r2(self):
        """iso_count size+rho model LOO R² > 0.96."""
        r2 = self._loo_r2("iso_count", "size_rho")
        assert r2 > 0.96, f"iso_count size_rho LOO R² = {r2:.3f} should be > 0.96"

    def test_rule_only_loo_r2_negative(self):
        """Rule alone has negative LOO R²: rule is NOT the amplitude driver."""
        r2 = self._loo_r2("iso_count", "rule_only")
        assert r2 < 0, f"iso_count rule_only LOO R² = {r2:.3f} should be negative"

    def test_size_only_substantial(self):
        """Size alone predicts amplitude with LOO R² > 0.75."""
        r2 = self._loo_r2("iso_count", "size_only")
        assert r2 > 0.75, f"iso_count size_only LOO R² = {r2:.3f} should be > 0.75"

    def test_fate_all_full_loo_r2(self):
        """fate_all full LOO R² also > 0.95 — amplitude law is mechanism-agnostic."""
        r2 = self._loo_r2("fate_all", "full")
        assert r2 > 0.95, f"fate_all full LOO R² = {r2:.3f} should be > 0.95"


@pytest.mark.skipif(
    not _outputs_exist(BRIDGE_VERDICT, BRIDGE_SUMMARY_CSV),
    reason="ca_lgds_bridge outputs not found",
)
class TestLGDSBridge:
    """Fine-net horizon tasks are rank-1 coherent; heterogeneous targets are not."""

    def setup_method(self):
        self.df = pd.read_csv(BRIDGE_SUMMARY_CSV)
        self.text = BRIDGE_VERDICT.read_text()

    def _row(self, family):
        row = self.df[self.df["family"] == family]
        assert len(row) == 1, f"Family '{family}' not found in bridge_summary.csv"
        return row.iloc[0]

    def test_verdict_pass(self):
        assert "VERDICT: PASS" in self.text

    def test_horizon_fine_net_mean_cos_near_1(self):
        """Horizon fine-net tasks are nearly collinear (mean |cos| > 0.99)."""
        r = self._row("horizon_fine_net")
        cos = float(r["mean_pairwise_abs_cosine"])
        assert cos > 0.99, (
            f"horizon_fine_net mean |cos| = {cos:.4f} should be > 0.99"
        )

    def test_horizon_fine_net_rank1_energy(self):
        """Rank-1 energy for fine-net horizon > 0.99."""
        r = self._row("horizon_fine_net")
        e = float(r["rank1_cumulative_energy"])
        assert e > 0.99, (
            f"horizon_fine_net rank-1 energy = {e:.4f} should be > 0.99"
        )

    def test_selection_multi_target_mean_cos_low(self):
        """Heterogeneous targets have mean |cos| < 0.60."""
        r = self._row("selection_multi_target")
        cos = float(r["mean_pairwise_abs_cosine"])
        assert cos < 0.60, (
            f"selection_multi_target mean |cos| = {cos:.4f} should be < 0.60"
        )

    def test_horizon_beats_selection_coherence(self):
        """Fine-net tasks are more coherent than the heterogeneous set."""
        r_h = self._row("horizon_fine_net")
        r_s = self._row("selection_multi_target")
        cos_h = float(r_h["mean_pairwise_abs_cosine"])
        cos_s = float(r_s["mean_pairwise_abs_cosine"])
        assert cos_h > cos_s + 0.30, (
            f"horizon mean |cos| ({cos_h:.3f}) should exceed selection mean |cos| ({cos_s:.3f}) by > 0.30"
        )


@pytest.mark.skipif(
    not _outputs_exist(PRESTATE_VERDICT, PRESTATE_SUMMARY_CSV),
    reason="prestate_class_horizon outputs not found",
)
class TestPrestateClass:
    """Non-leaky prestate: t=0 iso_count predicts horizon residual response."""

    def setup_method(self):
        self.df = pd.read_csv(PRESTATE_SUMMARY_CSV)
        self.text = PRESTATE_VERDICT.read_text()

    def test_iso_count_r2_positive_at_all_horizons(self):
        """iso_count R² > 0 at every horizon (frac_R2_positive = 1.0)."""
        sub = self.df[self.df["model"] == "iso_count"]
        assert len(sub) > 0, "No iso_count rows in prestate_class_summary.csv"
        assert (sub["frac_R2_positive"] == 1.0).all(), (
            "iso_count frac_R2_positive should be 1.0 at every horizon"
        )

    def test_iso_count_shuffle_r2_near_zero(self):
        """Shuffle null R² < 0.005 for iso_count."""
        sub = self.df[self.df["model"] == "iso_count"]
        assert (sub["mean_shuffle_R2"] < 0.005).all(), (
            f"iso_count shuffle R² should be < 0.005; got {sub['mean_shuffle_R2'].tolist()}"
        )

    def test_iso_count_r2_above_floor(self):
        """iso_count mean R² > 0.15 at every horizon."""
        sub = self.df[self.df["model"] == "iso_count"]
        for _, row in sub.iterrows():
            r2 = row["mean_R2"]
            k = row.get("horizon", "?")
            assert r2 > 0.15, f"iso_count mean R² = {r2:.3f} below 0.15 at horizon {k}"

    def test_class_counts_comparable_to_iso_count(self):
        """class_counts R² is at most modestly above iso_count (≤ 0.05 gap)."""
        sub_iso = self.df[self.df["model"] == "iso_count"]
        sub_cls = self.df[self.df["model"] == "class_counts"]
        if len(sub_iso) > 0 and len(sub_cls) > 0:
            mean_iso = sub_iso["mean_R2"].mean()
            mean_cls = sub_cls["mean_R2"].mean()
            gap = mean_cls - mean_iso
            assert gap < 0.05, (
                f"class_counts R² ({mean_cls:.3f}) exceeds iso_count ({mean_iso:.3f}) by {gap:.3f} — "
                "iso_count should be a near-sufficient prestate summary"
            )


# ===========================================================================
# 7. Artifact generator: make_response_law_artifacts.py
# ===========================================================================

MACROS_TEX = REPO_ROOT / "paper" / "macros.tex"
SCRIPTS_DIR = REPO_ROOT / "scripts"


@pytest.mark.skipif(
    not _outputs_exist(OUTPUTS_ROOT / "selection_principle_horizon" / "horizon_global_summary.csv"),
    reason="horizon outputs not found — run ca_horizon_response_test.py first",
)
class TestArtifactGenerator:
    """make_response_law_artifacts.py writes correct macros.tex and tables."""

    def setup_method(self):
        """Re-run the artifact generator before each test to get fresh output."""
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "make_response_law_artifacts.py")],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, (
            f"make_response_law_artifacts.py failed:\n{result.stderr}"
        )
        self.macros_text = MACROS_TEX.read_text() if MACROS_TEX.exists() else ""

    def test_macros_file_exists(self):
        assert MACROS_TEX.exists(), "paper/macros.tex was not created"

    def test_horizon_slope_macros_present(self):
        """All 7 horizon slope macros present (letter-only names) and negative."""
        import re
        k_name = {1: "kone", 5: "kfive", 10: "kten", 25: "ktwentyfive",
                  50: "kfifty", 100: "khundred", 200: "ktwohundred"}
        for k, name in k_name.items():
            macro = f"\\horizonSlope{k}"  # old-style with digit
            assert macro not in self.macros_text, (
                f"{macro} uses a digit — LaTeX name must be letters only"
            )
            macro_new = f"\\horizon{name}"
            assert macro_new in self.macros_text, (
                f"Missing macro {macro_new} in macros.tex"
            )
            m = re.search(rf"\\newcommand{{\\horizon{name}}}{{(-?\d+\.\d+)}}", self.macros_text)
            assert m, f"Could not parse {macro_new} value"
            val = float(m.group(1))
            assert val < 0, f"{macro_new} = {val} should be negative"

    def test_amp_loo_full_macro(self):
        """ampLOOfull should be present and > 0.95."""
        import re
        m = re.search(r"\\newcommand{\\ampLOOfull}{(-?\d+\.\d+)}", self.macros_text)
        assert m, "ampLOOfull macro not found in macros.tex"
        val = float(m.group(1))
        assert val > 0.95, f"\\ampLOOfull = {val} should be > 0.95"

    def test_iso_count_cv_r2_macro(self):
        """isocountR macro present and in [0.30, 0.45]."""
        import re
        m = re.search(r"\\newcommand{\\isocountR}{(-?\d+\.\d+)}", self.macros_text)
        assert m, "isocountR macro not found in macros.tex"
        val = float(m.group(1))
        assert 0.30 <= val <= 0.45, f"\\isocountR = {val} outside [0.30, 0.45]"

    def test_horizon_table_exists(self):
        tab = REPO_ROOT / "paper" / "tables" / "tab_horizon.tex"
        assert tab.exists(), "paper/tables/tab_horizon.tex was not created"

    def test_mechanism_table_exists(self):
        tab = REPO_ROOT / "paper" / "tables" / "tab_mechanism.tex"
        assert tab.exists(), "paper/tables/tab_mechanism.tex was not created"

    def test_no_nan_in_macros(self):
        """No macro should have 'nan' as its value."""
        assert "nan" not in self.macros_text, (
            "macros.tex contains 'nan' — some lookup failed"
        )

    def test_no_digit_in_macro_names(self):
        """LaTeX command names must not contain digits."""
        import re
        bad = re.findall(r"\\newcommand{\\(\w*\d\w*)}", self.macros_text)
        assert len(bad) == 0, (
            f"Macro names with digits (invalid LaTeX): {bad}"
        )

    def test_all_paper_macros_defined(self):
        """Every macro used in paper.tex is present in macros.tex."""
        required = [
            "allfatesR", "ampCV", "ampLOOfull", "ampLOOsizerho",
            "globalresiR", "golresiR", "golslopeCV", "golslopemean",
            "hlresiR", "hlslopeCV", "hlslopemean",
            "horizonCos", "horizonRegret", "horizonRkhundred", "horizonRkone",
            "horizonkhundred", "horizonkone", "horizonnegfrac",
            "localwinR", "nullresiR",
            "prestateIsoMean", "prestateIsoMin", "prestateIsoShuffle",
            "selCos", "selRegret",
        ]
        for name in required:
            assert f"\\{name}" in self.macros_text, (
                f"Required macro \\{name} missing from macros.tex"
            )

    def test_key_figures_copied(self):
        """Key figure files should be present in paper/figures/."""
        fig_dir = REPO_ROOT / "paper" / "figures"
        for fname in [
            "fig_beta_iso_vs_horizon.png",
            "fig_fate_r2.png",
            "fig_transfer_standardized_r2.png",
            "fig_amplitude_by_condition.png",
        ]:
            assert (fig_dir / fname).exists(), f"Missing figure: {fname}"

    def test_background_figures_present(self):
        """Appendix E background figures (E1-E4) must be generated."""
        fig_dir = REPO_ROOT / "paper" / "figures"
        for fname in [
            "figE1_disagreement_scatter.pdf",
            "figE1_disagreement_scatter.png",
            "figE2_trajectories.pdf",
            "figE2_trajectories.png",
            "figE3_scale_R2.pdf",
            "figE3_scale_R2.png",
            "figE4_old_slopes.pdf",
            "figE4_old_slopes.png",
        ]:
            assert (fig_dir / fname).exists(), f"Missing background figure: {fname}"

    def test_background_source_data_present(self):
        """Study A/B/D source CSVs for Appendix E must exist in outputs/data/."""
        data_dir = REPO_ROOT / "outputs" / "data"
        for fname in [
            "fig1_studyA_scatter_source.csv",
            "fig2_studyA_traces_source.csv",
            "fig4_studyB_r2_vs_B_source.csv",
            "fig6_studyD_slope_summary_source.csv",
        ]:
            assert (data_dir / fname).exists(), f"Missing source CSV: {fname}"

    def test_background_source_data_nonempty(self):
        """Source CSVs for Appendix E must have >5 rows each."""
        import pandas as pd
        data_dir = REPO_ROOT / "outputs" / "data"
        for fname, min_rows in [
            ("fig1_studyA_scatter_source.csv", 100),
            ("fig2_studyA_traces_source.csv",  10),
            ("fig4_studyB_r2_vs_B_source.csv", 4),
            ("fig6_studyD_slope_summary_source.csv", 4),
        ]:
            df = pd.read_csv(data_dir / fname)
            assert len(df) >= min_rows, (
                f"{fname}: expected >= {min_rows} rows, got {len(df)}"
            )
