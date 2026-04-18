"""
test_regression.py
==================
Regression tests for ca.py core computations.

Run with:
    pytest test_regression.py -v

Tests cover:
  1. Periodic (torus) BFS component counting
  2. Embedded / alone cell classification
  3. Narrative gap (G) computation
  4. Negative controls: stored study_B data R² < 0.04
  5. Study A partial correlation in expected range
  6. Study D empirical slope CV below threshold
"""

import json
import os
import sys

import numpy as np
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

DATA_DIR = os.path.join(os.path.dirname(__file__), "outputs", "data")
STATS_EXIST = os.path.isdir(DATA_DIR)


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
