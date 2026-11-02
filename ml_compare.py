"""Optional ANN/GPR/XGBoost comparison on one labeled FEM dataset.

Requires requirements-ml.txt. The held-out split matches surrogate.train for
the same dataset and seed; selection uses training-only cross-validation.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def _candidates(seed: int) -> list[tuple[str, str, Any]]:
    models = []
    for hidden in ((32,), (64, 32)):
        models.append(("ANN_MLPRegressor", f"hidden={hidden}",
                       make_pipeline(StandardScaler(), MLPRegressor(
                           hidden_layer_sizes=hidden, activation="relu", solver="adam",
                           alpha=0.001, learning_rate_init=0.005,
                           max_iter=300, early_stopping=True, validation_fraction=0.15,
                           n_iter_no_change=20, random_state=seed))))
    for length_scale in (1.5, 3.0):
        kernel = (ConstantKernel(1.0, constant_value_bounds="fixed") *
                  RBF(length_scale=length_scale, length_scale_bounds="fixed") +
                  WhiteKernel(noise_level=1e-5, noise_level_bounds="fixed"))
        models.append(("GPR_GaussianProcessRegressor", f"length_scale={length_scale}",
                       make_pipeline(StandardScaler(), GaussianProcessRegressor(
                           kernel=kernel, optimizer=None, alpha=1e-8,
                           random_state=seed))))
    for depth in (3, 5):
        models.append(("XGBoost_XGBRegressor", f"max_depth={depth}",
                       make_pipeline(StandardScaler(), XGBRegressor(
                           n_estimators=220, max_depth=depth, learning_rate=0.05,
                           subsample=0.9, colsample_bytree=0.9,
                           objective="reg:squarederror", tree_method="hist",
                           n_jobs=1, random_state=seed))))
    return models


def _ape(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return 100 * np.abs(predicted - actual) / actual


def _predict_hz(estimator: Any, x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    standardized = np.asarray(estimator.predict(x))
    if standardized.ndim == 1:
        standardized = standardized[:, None]
    return np.exp(standardized * scale + mean)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    error = _ape(actual, predicted)
    return {
        "median_ape_percent_by_mode": np.median(error, axis=0).tolist(),
        "p95_ape_percent_by_mode": np.percentile(error, 95, axis=0).tolist(),
        "r2_by_mode": [float(r2_score(actual[:, i], predicted[:, i]))
                       for i in range(actual.shape[1])],
        "overall_median_ape_percent": float(np.median(error)),
    }


def compare(dataset: dict[str, Any], seed: int = 42) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return machine-readable report and fitted CV-selected model payload."""
    records = dataset["records"]
    metadata = dataset["metadata"]
    source = metadata.get("source")
    if source not in {"local_linear_beam_FEM", "calculix_ccx_native"}:
        raise ValueError("unsupported FEM dataset provenance")
    if any(row.get("solver") != source for row in records):
        raise ValueError("mixed or missing solver provenance in dataset")
    if len(records) < 50:
        raise ValueError("at least 50 successful designs needed")
    keys = sorted(metadata["ranges"])
    ranges = {key: metadata["ranges"][key] for key in keys}
    x = np.array([[float(row["inputs"][key]) for key in keys] for row in records])
    y = np.array([row["frequencies_hz"] for row in records], dtype=float)
    limits = np.array([ranges[key] for key in keys], dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.min(y) <= 0:
        raise ValueError("nonfinite input or nonpositive frequency")
    if np.any(x < limits[:, 0]) or np.any(x > limits[:, 1]):
        raise ValueError("dataset input outside declared range")
    if any(min(row["mode_mac"]) < 0.5 for row in records):
        raise ValueError("ambiguous tracked mode in dataset")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(records))
    n_test = max(1, int(round(0.2 * len(records))))
    test_idx, train_idx = indices[:n_test], indices[n_test:]
    log_y = np.log(y)
    log_mean = np.mean(log_y[train_idx], axis=0)
    log_scale = np.std(log_y[train_idx], axis=0)
    log_scale[log_scale < 1e-8] = 1.0
    standardized_y = (log_y - log_mean) / log_scale
    folds = list(KFold(n_splits=3, shuffle=True, random_state=seed).split(train_idx))
    comparisons = []
    final_estimators = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for family, config, estimator in _candidates(seed):
            cv_errors = []
            for fit_relative, val_relative in folds:
                fit_idx, val_idx = train_idx[fit_relative], train_idx[val_relative]
                fold_mean = np.mean(log_y[fit_idx], axis=0)
                fold_scale = np.std(log_y[fit_idx], axis=0)
                fold_scale[fold_scale < 1e-8] = 1.0
                fold_y = (log_y[fit_idx] - fold_mean) / fold_scale
                estimator.fit(x[fit_idx], fold_y)
                estimate = _predict_hz(estimator, x[val_idx], fold_mean, fold_scale)
                cv_errors.extend(_ape(y[val_idx], estimate).flatten().tolist())
            cv_median = float(np.median(cv_errors))
            estimator.fit(x[train_idx], standardized_y[train_idx])
            predicted = _predict_hz(estimator, x[test_idx], log_mean, log_scale)
            comparisons.append({
                "family": family, "configuration": config,
                "cv_median_ape_percent": cv_median,
                "held_out": _metrics(y[test_idx], predicted),
            })
            final_estimators[(family, config)] = estimator
    selected = min(comparisons, key=lambda row: row["cv_median_ape_percent"])
    selected_key = (selected["family"], selected["configuration"])
    report = {
        "schema_version": 1,
        "source": source,
        "not_ansys_or_test_data": True,
        "seed": seed, "input_keys": keys, "input_ranges": ranges,
        "train_design_count": len(train_idx), "held_out_design_count": len(test_idx),
        "held_out_design_ids": [records[i]["design_id"] for i in test_idx],
        "selection_rule": "lowest training-only 3-fold CV median APE across tracked modes",
        "selected_family": selected["family"],
        "selected_configuration": selected["configuration"],
        "comparisons": comparisons,
        "target_description": metadata.get("target_description"),
    }
    model = {
        "schema_version": 1, "source": report["source"],
        "input_keys": keys, "input_ranges": ranges,
        "selected_family": selected["family"],
        "selected_configuration": selected["configuration"],
        "log_target_mean": log_mean, "log_target_scale": log_scale,
        "estimator": final_estimators[selected_key],
    }
    return report, model


def predict(model: dict[str, Any], design: dict[str, Any]) -> list[float]:
    keys = model["input_keys"]
    values = np.array([float(design[key]) for key in keys])
    limits = np.array([model["input_ranges"][key] for key in keys], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("design inputs must be finite")
    outside = [key for key, value, (lo, hi) in zip(keys, values, limits)
               if value < lo or value > hi]
    if outside:
        raise ValueError("out of trained range: " + ", ".join(outside))
    prediction = _predict_hz(model["estimator"], values[None, :],
                             model["log_target_mean"], model["log_target_scale"])
    return [float(value) for value in prediction[0]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fitting = sub.add_parser("train")
    fitting.add_argument("--dataset", type=Path, required=True)
    fitting.add_argument("--out", type=Path, required=True,
                         help="directory for comparison.json and best_model.joblib")
    fitting.add_argument("--seed", type=int, default=42)
    inference = sub.add_parser("predict")
    inference.add_argument("--model", type=Path, required=True)
    inference.add_argument("--design", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        report, model = compare(json.loads(args.dataset.read_text()), args.seed)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        joblib.dump(model, args.out / "best_model.joblib")
        print(json.dumps({"selected_family": report["selected_family"],
                          "selected_configuration": report["selected_configuration"],
                          "report": str(args.out / "comparison.json"),
                          "model": str(args.out / "best_model.joblib")}, indent=2))
    else:
        model = joblib.load(args.model)
        result = predict(model, json.loads(args.design.read_text()))
        print(json.dumps({"predicted_frequencies_hz": result,
                          "source": model["source"],
                          "family": model["selected_family"]}, indent=2))


if __name__ == "__main__":
    main()
