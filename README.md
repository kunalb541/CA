# Embedded Isolates Define a Target-Specific Temporal Response Law in Cellular Automata

**Kunal Bhatia** · Independent Researcher, Heidelberg, Germany  
ORCID: [0009-0007-4447-6325](https://orcid.org/0009-0007-4447-6325)

---

## Core result

The count of *embedded isolated cells* — alive cells with no 4-connected live neighbours but at least one diagonal live neighbour — is the **sufficient non-leaky prestate object summary** for future fine-component change in GoL and HighLife.

Specifically:
1. **Target-specific selection (GoL: PASS, HighLife: WEAK PASS, Global: PASS).** iso_count adds ΔR² ≈ 0.001 for the fine-net target; iso-shuffle and target-shuffle nulls are near zero (ΔR² < 0.001).
2. **Temporal response law.** β_iso(k) ≈ −0.70 to −0.80 across all tested horizons k ∈ {1,5,10,25,50,100,200}, two rules (GoL + HighLife), two grid sizes (L=64,128), four density bands. All 112 condition-horizon slopes are negative.
3. **Mechanism: local component-context loss.** Not simple cell death. Local-window loss CV R² = 0.538 vs iso_count alone = 0.355. The mechanism transfers across all 16 conditions after standardization (fate_all R²_z ≈ 0.545, frac_R2_positive = 1.0).
4. **Two-layer amplitude structure.** Standardized mechanism is transferable; raw amplitude is predictable from (L, ρ) with LOO R² = 0.977.
5. **LGDS bridge.** Fine-net horizon tasks are rank-1 coherent (mean |cos| = 0.999); heterogeneous targets are not (mean |cos| = 0.427).

---

## Repository structure

```
ca.py                    # Core simulation engine (GoL/HighLife BFS, isolate classifier)
test_regression.py       # 93 regression tests covering all results + artifact generator
scripts/                 # Analysis scripts (run from repo root)
  ca_selection_principle_test.py
  ca_horizon_response_test.py
  ca_isolate_fate_mechanism.py
  ca_isolate_transition_classes.py
  ca_lgds_bridge_test.py
  ca_mechanism_transfer_test.py
  ca_mechanism_transfer_standardized.py
  ca_mechanism_amplitude_law.py
  ca_prestate_class_horizon_test.py
  make_response_law_artifacts.py   # generates all figures, macros, tables
outputs/                 # All simulation outputs (pre-computed)
  selection_principle/
  selection_principle_horizon/
  isolate_fate/
  isolate_transition_classes/
  ca_lgds_bridge/
  mechanism_transfer/
  mechanism_transfer_standardized/
  mechanism_amplitude_law/
  prestate_class_horizon/
  data/                  # Source CSVs for background appendix figures (Study A/B/D)
    fig1_studyA_scatter_source.csv      # observer disagreement scatter (n=1000)
    fig2_studyA_traces_source.csv       # fine/coarse cumulative trajectories
    fig4_studyB_r2_vs_B_source.csv      # R² vs block size per target
    fig6_studyD_slope_summary_source.csv# old GoL-only slope summary (6 conditions)
paper/                   # Manuscript (21 pages)
  paper.tex
  paper.pdf
  refs.bib
  macros.tex             # auto-generated: 49 LaTeX macros, letters-only names
  build.sh
  figures/               # 8 flagship + 4 background appendix figures
  tables/
paper.pdf                # Root copy of compiled PDF
build.sh                 # Root build script
```

---

## Paper structure

The paper is 21 pages (18 main + 3 appendix). Main sections:

| Section | Content |
|---------|---------|
| I | Introduction |
| II | Definitions and Protocol |
| III | Result 1 — Target-Specific Selection |
| IV | Result 2 — Non-Leaky Prestate Summary |
| V | Result 3 — Temporal Response Law |
| VI | Result 4 — Mechanism Carrier |
| VII | Result 5 — Transfer, Amplitude Law, Two-Layer Structure |
| VIII | Task-Direction Coherence (LGDS bridge) |
| IX | Discussion |
| X | Conclusion |
| App. A | Residualisation Protocol |
| App. B | Null Definitions |
| App. C | Transition-Class Coding |
| App. D | Test and Regression Battery |
| **App. E** | **Background Observer-Scale Diagnostics** (lineage figures, not load-bearing) |

### Appendix E figures

These figures reproduce earlier observer-scale diagnostics from the same research program.
They are **background/lineage only** — they show why target-indexed description was a
natural next question. They are **not load-bearing evidence** for the response-law claims.

| Figure | File | Data source | Content |
|--------|------|-------------|---------|
| E1 | `figE1_disagreement_scatter.pdf` | `fig1_studyA_scatter_source.csv` | Observer gap G vs early fine change, r = −0.765 |
| E2 | `figE2_trajectories.pdf` | `fig2_studyA_traces_source.csv` | Fine/coarse cumulative trajectories for max/min G worlds |
| E3 | `figE3_scale_R2.pdf` | `fig4_studyB_r2_vs_B_source.csv` | R² vs block size B per target (different targets peak at different scales) |
| E4 | `figE4_old_slopes.pdf` | `fig6_studyD_slope_summary_source.csv` | Old GoL-only β ≈ −1.52 slopes (different protocol; not comparable to new β ≈ −0.70) |

---

## Reproducing paper artifacts from existing outputs

```bash
./build.sh
```

This runs `scripts/make_response_law_artifacts.py` (reads `outputs/`, writes
`paper/figures/` — 8 flagship + 4 background figures — and `paper/tables/`),
then compiles `paper/paper.tex` to `paper/paper.pdf` and copies it to the root.

No simulations are re-run. All pre-computed outputs are committed to `outputs/`.

---

## Rerunning individual analyses

All scripts are in `scripts/` and should be run from the **repo root**:

```bash
source /path/to/venv/bin/activate
python scripts/ca_selection_principle_test.py
python scripts/ca_horizon_response_test.py
# etc.
```

Outputs are written to `outputs/<module>/`.

---

## Running tests

```bash
pytest test_regression.py -v
```

93 tests cover:
- BFS component counter and isolate classifier correctness
- Selection principle verdicts (GoL PASS, HighLife WEAK PASS, Global PASS)
- Horizon response (all 112 slopes negative, CIs negative, R² above floor, null near zero)
- Mechanism ordering (local-window > iso_count > coarse)
- Standardized transfer (frac R²-positive = 1.0)
- Amplitude-law LOO R² thresholds
- LGDS / task-direction coherence
- Prestate non-leakiness across all horizons
- Artifact generator: no NaN macros, no digit macro names, all figure PDFs present
- **Background appendix figures E1–E4 present and source data non-empty**

---

## Citation

```
Bhatia, K. (2026). Embedded Isolates Define a Target-Specific Temporal Response Law
in Cellular Automata. Preprint. https://github.com/kunalb541/CA
```

---

## License

MIT — see [LICENSE](LICENSE).
