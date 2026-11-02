# ANN, GPR and XGBoost comparison

`ml_compare.py` adds the presentation's three named model families as an **optional** comparison. It does not replace the NumPy surrogate used by the RPM check. Training targets come from one explicitly tagged solver dataset: native CalculiX or legacy local beam FEM, never a mixture or hardware measurements.

## Run

Install `requirements-ml.txt` in a Python environment, then run from the project directory:

```sh
python ml_compare.py train --dataset build/full/data/dataset.json --out build/ml_compare --seed 42
python ml_compare.py predict --model build/ml_compare/best_model.joblib --design config/reference_design.json
```

`comparison.json` contains the full machine-readable evaluation; `best_model.joblib` contains the training-CV-selected fitted estimator, feature keys/ranges and log-target scaling. Only load a joblib model from a trusted source. The `predict` command rejects structural inputs beyond the saved sample ranges. `run_pipeline.py --compare-ml` runs the same comparison under its chosen output directory.

## Evaluation protocol

The design-level 80/20 split reproduces `surrogate.train` for the same dataset and seed (500 designs produce 400 train and 100 held-out). Within cross-validation, target scaling is fitted separately on each fold's fit portion; final scaling uses all 400 training designs. Each family has two compact configurations: MLP hidden widths, fixed GPR RBF length scale, or XGBoost tree depth. Three-fold cross-validation on the training designs chooses the best configuration by median absolute percentage error across the tracked modes. Each candidate is then fitted on all 400 training designs and scored once on the untouched 100 held-out designs. The saved model is selected by **CV**, not by the held-out scores.

The report provides held-out median and 95th-percentile absolute percentage error and R² separately for each tracked mode. The third target may represent a near-degenerate mode pair's mean frequency; it does not distinguish the two eigenvectors. All metrics measure agreement with the tagged source dataset. They do not quantify material uncertainty, joint behavior, damping, or physical prediction accuracy.
