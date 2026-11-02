# Native CalculiX implementation and trial results

The pipeline now generates quadratic S8R shell decks, runs native CalculiX
2.23, parses modal frequencies and motor-tip shapes, tracks modes through a
parameter sweep, trains a frequency surrogate, and evaluates a Campbell
separation screen. These are simulation results for a provisional concept
model. No ANSYS solve or physical modal test was performed.

## Model and numerical check

The model has four shell arms and a coplanar equivalent hub shell. The hub
represents two separated plates with matching idealized areal mass, membrane
stiffness, and bending stiffness. Arm roots are connected through hub-corner
interpolation equations. Four motor/mount masses and one central payload mass
are lumped. Material inputs are effective orthotropic constants rather than
a resolved laminate layup.

The baseline native solve gave six near-zero rigid frequencies, a total mass
of **1.08193 kg**, and flexible frequencies of **21.653, 32.888, 137.592,
137.632, 150.645, and 167.989 Hz**. Refining the Gmsh target size from
25 mm to 12.5 mm changed the first five frequencies by **0.568%, 1.402%,
0.555%, 0.554%, and 1.937%**, respectively. This passes the project's
2% baseline mesh check. It is a numerical convergence check, not a validation
against a manufactured frame.

## Sweep and held-out predictions

| Native run | Successful designs | Rejected candidates | Held-out designs | Median absolute percentage error, modes 1–3 | 95th percentile error, modes 1–3 |
| --- | ---: | ---: | ---: | --- | --- |
| 10-case trial | 10 | 1 | — | Training intentionally skipped | — |
| 50-case pilot | 50 | 15 | 10 | 1.492%, 0.964%, 1.115% | 5.267%, 6.072%, 4.603% |
| 500-case run | 500 | 128 | 100 | 0.334%, 0.750%, 0.310% | 0.804%, 2.090%, 0.871% |

All 128 rejected candidates in the full run failed the mode-tracking
ambiguity threshold; their raw solver outputs are generated locally under
`build/calculix_full500/data/solver_cases` and are omitted from this repository. The dataset contains only
native CalculiX rows, identified by `calculix_ccx_native`. The reported
held-out errors measure interpolation within the sampled input ranges and
within this one FE model. They do not measure real-world accuracy.

The optional named-model comparison used the same 400/100 design split and
training-only three-fold selection. It selected an ANN with hidden layers
(64, 32), whose held-out median errors were **1.486%, 1.991%, and 1.739%**.
The operational NumPy quadratic ridge model had lower held-out errors on
this split and remains the model used for the Campbell screen. The six
ANN/GPR/XGBoost configurations and their metrics are recorded in
`results/calculix_full500/ml_compare/comparison.json`.

The reference-design Campbell screen **fails**. The predicted third tracked
frequency is **138.02 Hz**, inside both the 1P excitation range of
50–150 Hz and the two-blade pass range of 100–300 Hz for the specified
3,000–9,000 RPM operating range. The first two tracked frequencies pass the
separation screen. This indicates frequency overlap; forcing, damping, and
response amplitudes were not computed.

## Audit files and next engineering decisions

- `results/calculix_full500/run_manifest.json`: solver provenance and summary.
- `results/calculix_full500/baseline_calculix/`: baseline mesh, input deck, and
  native solver output.
- `results/calculix_full500/mesh_convergence.json`: baseline refinement check.
- `results/calculix_full500/data/dataset.json` and `failures.json`: accepted
  designs and rejected candidates.
- `results/calculix_full500/surrogate_model.json` and `design_check.json`:
  trained model and reference-design screen.
- `results/calculix_full500/ml_compare/`: optional named-model comparison
  report and selected ANN model.
- `results/calculix_full500/campbell.svg`: Campbell plot.

The next design iteration should move the third mode outside the operating
excitation bands while checking mass and the lower modes. Before a hardware
decision, replace assumed masses, dimensions, clamp stiffness, and material
constants with measured values, then compare predicted mode shapes and
frequencies against a suspended-frame impact test.
