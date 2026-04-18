# Observer Disagreement, Predictive Scale, and a Stable Component-Change Law in Conway's Game of Life

**Kunal Bhatia** · Independent Researcher, Heidelberg, Germany  
ORCID: [0009-0007-4447-6325](https://orcid.org/0009-0007-4447-6325)

Preprint submitted to *Chaos* (AIP). Not peer reviewed.

---

## What this paper establishes

Different coarse-grained descriptions of the same deterministic system can yield different accounts of net change. Using Conway's Game of Life (GoL) as a controlled test bed, we establish three empirical regularities:

1. **Predictable observer disagreement.** A *fine observer* (4-connected component count on the torus) and a *coarse observer* (occupied 8×8 block count) can accumulate opposite net-change narratives over the same deterministic run. The disagreement — measured as a normalised narrative gap *G* — is strongly predicted by early fine-scale component dynamics: partial *r* = −0.848 (95% bootstrap CI [−0.862, −0.830], *N* = 1000) after controlling for initial density.

2. **Target-relative predictive scale.** The observation scale that best forecasts a future quantity depends on which quantity. Time-averaged live-cell count peaks at *B* = 2; occupied-block count peaks at *B* = 8. The bootstrap confidence intervals for these two peak locations are non-overlapping ([2, 4] vs [8, 8]), providing direct evidence against a single universally optimal scale.

3. **A stable component-change law.** Embedded isolated cells — alive, with no 4-connected live neighbours but at least one diagonal neighbour — predict future component-level decline with a stable OLS slope across six size-by-density conditions (mean slope ≈ −1.52, CV = 0.087, well below the pre-specified threshold of 0.20).

Two stronger mechanism claims are tested and explicitly rejected:
- Global mediation through component net change: mediation fraction −0.072, bootstrap 95% CI entirely below zero.
- Residual coupling beyond death and local-neighbourhood dynamics: χ < 0 in all six conditions under the dynamic local null.

All results are established computationally within GoL. Generalisation beyond this system is an open empirical question.

---

## Repository structure

```
ca.py                  # Simulation engine, all studies, figure/table generation
paper.tex              # Manuscript (RevTeX 4-2, AIP/Chaos format)
paper.pdf              # Compiled preprint
build.sh               # Full build script (runs ca.py then compiles LaTeX)
test_regression.py     # Regression tests for core computations

outputs/
  data/                # Simulation data CSVs and stats JSON files
  figures/             # All paper figures (PDF + PNG)
  tables/              # Summary tables (CSV)
  logs/                # LaTeX auxiliary files, adjudication note
```

---

## Reproducing all results

**Requirements:** Python 3.9+ with numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, tqdm; LaTeX with revtex4-2 and latexmk.

```bash
# Activate environment (adjust path as needed)
source /path/to/your/venv/bin/activate

# Full rebuild: simulations + figures + compiled PDF
bash build.sh
```

Running `build.sh` will:
1. Execute `ca.py`, which runs all four studies (~30–60 minutes depending on hardware).
2. Write all outputs to `outputs/`.
3. Compile `paper.tex` with latexmk and copy `paper.pdf` to the project root.

To run only the simulation and inspect outputs without compiling LaTeX:
```bash
python ca.py
```

---

## Running the tests

```bash
pytest test_regression.py -v
```

The test suite covers:
- Periodic (torus) BFS component counting vs non-periodic counting.
- Embedded and alone cell classification edge cases.
- Narrative gap computation for known inputs.
- That all negative controls produce R² < 0.04 (regression against stored data). Note: 4 of the 20 conditions use B=16 static features, which are near-constant at the study densities (all blocks occupied); those conditions pass trivially. Removing them, 0/16 non-trivial conditions exceed the threshold.

---

## Key design decisions

### Fine observer: periodic BFS, not `scipy.ndimage.label`

`scipy.ndimage.label` operates on a flat array and cannot detect components that span opposite edges of a toroidal grid. All component counting uses a hand-written BFS that wraps indices modulo the grid dimensions.

### Embedded cell definition

A cell is **embedded** if: (i) it is alive, (ii) all four 4-connected neighbours are dead, (iii) at least one diagonal neighbour is alive. This is a purely structural, contemporaneous criterion — the outcome variable (future component net change) plays no role in the definition.

### Bootstrap resampling unit

In Study A, the resampling unit is the world row (one row = one 250-step simulation). The bootstrap therefore captures sampling variability over the world distribution, not over individual time steps.

### Dynamic local null (Study D)

Each embedded cell's local structural contribution is estimated by a counterfactual: remove the focal cell from the full torus, step both versions one step, and compare 4-connected component counts inside the 5×5 neighbourhood patch centred on the focal site (non-periodic within the patch). The +1.0 offset in `dynamic_local_delta_for_focal` ensures the decomposition β_emp = β_death + β_local + β_residual simplifies to β_residual = β_emp − mean(δ_sync).

### No hyperparameter search

All Ridge regression uses α = 1.0 fixed. No tuning was performed across conditions. The 5-fold CV splits use a fixed seed; all thresholds (CV ≤ 0.20, R² > 0.04) were specified before analysis.

---

## Reproducibility notes

- All seeds are stored in the data CSVs; any individual world can be regenerated exactly.
- Study D uses six separate seeds (BASE_SEED + k for k = 0..5); the per-condition slopes and χ values in Table 6 are exactly reproducible.
- The bootstrap CIs in Studies A and B are reproducible from the stored seeds (SEED = 20260325 for Study A, 20260326 for Study B).

---

## Citation

```
Bhatia, K. (2026). Observer Disagreement, Predictive Scale, and a Stable
Component-Change Law in Conway's Game of Life. Preprint.
https://github.com/kunalb541/CA
```

---

## License

MIT — see [LICENSE](LICENSE).
