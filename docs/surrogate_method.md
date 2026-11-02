# Sweep, surrogate and RPM screen

The sweep has two explicitly selected sources. The default native path calls `calculix_runner.solve_calculix`; `--backend beam` retains the local linear beam demonstration. Dataset rows are tagged by solver and cannot be mixed. Neither source is measured vibration data. The reference values are concept assumptions; replace them with sourced materials, measured masses and dimensions, and impact-test comparisons before interpreting the screen for real hardware.

## Reproduce

From the project directory, with Python and NumPy installed:

```sh
python3 sweep.py --baseline config/reference_design.json --out build/pilot --count 50 --seed 42
python3 surrogate.py train --dataset build/pilot/dataset.json --out build/pilot/model.json --seed 42
python3 campbell.py --model build/pilot/model.json --design config/reference_design.json --out build/pilot/check
```

For the expanded sweep, change `--out` to `build/sweep500` and `--count` to `500`, then train and check using that directory. The seed makes sampling and the train/test split repeatable. `dataset.json` includes source, input ranges, each design, raw and tracked frequencies, MAC values and mesh metadata. Tip mode vectors are used during matching but are not retained in this compact dataset. `failures.json` records unsuccessful candidate solves or ambiguous mode matches; these do not count toward the requested successful design total. The output report includes `check.json` and a standalone `campbell.svg`.

The default Latin hypercube varies arm length, width and thickness; hub size and plate thickness; motor and payload mass; longitudinal and transverse elastic moduli; in-plane shear modulus; and density. The remaining provisional orthotropic constants stay fixed. RPM and blade count affect excitation rather than the unpreloaded structural modes. Geometry is deep-copied for each candidate and hub plate gap follows arm thickness to preserve clamp clearance. Inputs and solver outputs use SI units. The transverse modulus affects the CalculiX shell model but is **not used by the legacy Euler–Bernoulli beam model**. Neither model includes joint slip, fastener preload, damping, blade aerodynamics, or propeller gyroscopic effects.

## Mode tracking

The first two targets are assigned to baseline modes by modal assurance criterion (MAC) on twelve tip displacement components. If the symmetric X-frame's third and fourth modes form a near-degenerate pair within 1%, the third target is their mean frequency and its similarity check compares the two-dimensional subspace. A sample with any similarity below 0.5 is written to `failures.json` as ambiguous rather than silently relabeled. Tip sampling does not fully characterize internal plate or joint deformation; an experimental modal test is needed for credible physical mode-family labels.

## Model and evaluation

`surrogate.py` compares a quadratic ridge regression and an RBF kernel ridge regression, both implemented in NumPy. They are not an ANN, Gaussian process or XGBoost. An untouched 20% of successful designs is reserved at the design level for final evaluation. Five-fold cross-validation on the other 80% selects the family and regularization. Frequencies are fitted in log space so predictions stay positive. The model JSON records the selected fit, every CV score, test design IDs, predicted versus actual values, per-mode median and 95th-percentile absolute percentage errors, and whether the original 5%/10% goals were met. These are **held-out errors against the same source solver**, not errors against hardware.

Inference rejects any structural input outside the saved training ranges. The RPM check plots 1P (`RPM / 60`) and blade pass (`blades × RPM / 60`) over the specified operating interval. For each mode and excitation line it computes the minimum fractional separation over the *continuous* RPM range and requires it to exceed 15% plus the held-out 95th-percentile model error. A crossing is flagged for review; passing this frequency screen does not establish vibration amplitude, damping, fatigue life or flight safety.

The 95th-percentile test error is an empirical allowance, not a calibrated statistical confidence interval. The 50-design pilot has only ten held-out designs, so the expanded 500-design evaluation is the more useful estimate of error against this beam model.
