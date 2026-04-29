# Embedded Isolates Define a Target-Specific Temporal Response Law in Cellular Automata

**Kunal Bhatia** · Independent Researcher, Heidelberg, Germany  
ORCID: [0009-0007-4447-6325](https://orcid.org/0009-0007-4447-6325)

> **Note:** This repo previously contained an earlier CA observer-disagreement paper. That paper has been superseded by the response-law paper documented here. The original simulation engine (`ca.py`) and regression tests are preserved.

---

## Core result

The count of *embedded isolated cells* — alive cells with no 4-connected live neighbours but at least one diagonal live neighbour — is the **sufficient non-leaky prestate object summary** for future fine-component change in GoL and HighLife.

Specifically:
1. **Target-specific selection (GoL: PASS, Global: PASS).** iso_count adds ΔR² ≈ 0.001 for the fine-net target; null ΔR² ≈ −0.002. The response is target-specific, not a density artefact.
2. **Temporal response law.** β_iso(k) ≈ −0.70 to −0.80 across all tested horizons k ∈ {1,5,10,25,50,100,200}, two rules, two sizes, three density bands. All 112 condition-horizon slopes are negative.
3. **Mechanism: local component-context loss.** Not simple cell death. Local-window loss CV R² = 0.538 vs iso_count alone = 0.355. The mechanism transfers across density, size, rule, and condition after condition-standardization (fate_all R²_z ≈ 0.545, frac_R2_positive = 1.0).
4. **Two-layer amplitude structure.** Standardized mechanism is transferable; raw amplitude is predictable from (L, ρ) with LOO R² = 0.977.
5. **LGDS bridge.** Fine-net horizon tasks are rank-1 coherent (mean |cos| = 0.999); heterogeneous targets are not (mean |cos| = 0.427).

---

## Repository structure

```
ca.py                    # Core simulation engine (GoL/HighLife BFS, isolate classifier)
test_regression.py       # Regression tests for core computations
scripts/                 # Analysis scripts (moved from root)
  ca_selection_principle_test.py
  ca_horizon_response_test.py
  ca_isolate_fate_mechanism.py
  ca_isolate_transition_classes.py
  ca_lgds_bridge_test.py
  ca_mechanism_transfer_test.py
  ca_mechanism_transfer_standardized.py
  ca_mechanism_amplitude_law.py
  ca_prestate_class_horizon_test.py
  make_response_law_artifacts.py
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
paper/                   # Manuscript
  paper.tex
  paper.pdf
  refs.bib
  macros.tex
  build.sh
  figures/
  tables/
paper.pdf                # Root copy of compiled PDF
build.sh                 # Root build script
```

---

## Reproducing paper artifacts from existing outputs

```bash
./build.sh
```

This runs `scripts/make_response_law_artifacts.py` (reads outputs/, writes paper/figures/ and paper/tables/) then compiles `paper/paper.tex` to `paper/paper.pdf` and copies it to root.

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

---

## Citation

```
Bhatia, K. (2026). Embedded Isolates Define a Target-Specific Temporal Response Law
in Cellular Automata. Preprint. https://github.com/kunalb541/CA
```

---

## License

MIT — see [LICENSE](LICENSE).
