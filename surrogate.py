"""Train a small, dependency-light surrogate on one labeled FEM dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _features(z: np.ndarray) -> np.ndarray:
    n, d = z.shape
    cols = [np.ones(n)]
    cols.extend(z[:, i] for i in range(d))
    cols.extend(z[:, i] * z[:, j] for i in range(d) for j in range(i, d))
    return np.column_stack(cols)


def _rbf(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
    distances = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2) / a.shape[1]
    return np.exp(-gamma * distances)


def _fit(z: np.ndarray, y: np.ndarray, family: str, regularization: float,
         gamma: float | None = None) -> dict[str, Any]:
    if family == "quadratic_ridge":
        x = _features(z)
        penalty = np.eye(x.shape[1])
        penalty[0, 0] = 0
        coef = np.linalg.solve(x.T @ x + regularization * penalty, x.T @ y)
        return {"family": family, "regularization": regularization, "coef": coef.tolist()}
    if family == "rbf_kernel_ridge":
        assert gamma is not None
        gram = _rbf(z, z, gamma)
        alpha = np.linalg.solve(gram + regularization * np.eye(len(z)), y)
        return {"family": family, "regularization": regularization,
                "gamma": gamma, "training_z": z.tolist(), "alpha": alpha.tolist()}
    raise ValueError(f"unknown family: {family}")


def _predict_z(fit: dict[str, Any], z: np.ndarray) -> np.ndarray:
    if fit["family"] == "quadratic_ridge":
        return _features(z) @ np.asarray(fit["coef"])
    return _rbf(z, np.asarray(fit["training_z"]), float(fit["gamma"])) @ np.asarray(fit["alpha"])


def _percent_errors(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return 100 * np.abs(predicted - actual) / actual


def train(dataset: dict[str, Any], seed: int = 42) -> dict[str, Any]:
    """80/20 design-level split; CV uses training designs only."""
    records = dataset["records"]
    if len(records) < 20:
        raise ValueError("at least 20 successful designs needed for evaluation")
    metadata = dataset["metadata"]
    source = metadata.get("source")
    if source not in {"local_linear_beam_FEM", "calculix_ccx_native"}:
        raise ValueError("unsupported dataset provenance")
    if any(row.get("solver") != source for row in records):
        raise ValueError("mixed or missing solver provenance in dataset")
    keys = sorted(metadata["ranges"])
    limits = np.array([metadata["ranges"][key] for key in keys], dtype=float)
    x = np.array([[float(row["inputs"][key]) for key in keys] for row in records])
    y = np.array([row["frequencies_hz"] for row in records], dtype=float)
    if y.ndim != 2 or not np.isfinite(y).all() or np.min(y) <= 0:
        raise ValueError("invalid frequency targets")
    if not all(min(row["mode_mac"]) >= 0.5 for row in records):
        raise ValueError("dataset includes ambiguous mode tracking")
    if not np.isfinite(x).all() or np.any(x < limits[:, 0]) or np.any(x > limits[:, 1]):
        raise ValueError("dataset inputs outside stated parameter ranges")
    z = 2 * (x - limits[:, 0]) / (limits[:, 1] - limits[:, 0]) - 1
    log_y = np.log(y)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(records))
    n_test = max(1, int(round(0.2 * len(records))))
    test_idx, train_idx = indices[:n_test], indices[n_test:]
    # Standardize log targets using only the training split.
    log_mean = np.mean(log_y[train_idx], axis=0)
    log_scale = np.std(log_y[train_idx], axis=0)
    log_scale[log_scale < 1e-8] = 1.0
    standard_y = (log_y - log_mean) / log_scale
    train_order = rng.permutation(train_idx)
    folds = np.array_split(train_order, min(5, len(train_order)))
    candidates = [("quadratic_ridge", lam, None) for lam in (0.001, 0.01, 0.1, 1.0)]
    candidates += [("rbf_kernel_ridge", lam, gamma)
                   for gamma in (0.1, 0.5, 2.0) for lam in (0.001, 0.01, 0.1)]
    cv_results = []
    for family, lam, gamma in candidates:
        errors = []
        for validation in folds:
            fit_idx = np.setdiff1d(train_idx, validation)
            fitted = _fit(z[fit_idx], standard_y[fit_idx], family, lam, gamma)
            estimated = np.exp(_predict_z(fitted, z[validation]) * log_scale + log_mean)
            errors.extend(_percent_errors(y[validation], estimated).flatten().tolist())
        cv_results.append({"family": family, "regularization": lam, "gamma": gamma,
                           "cv_median_ape_percent": float(np.median(errors))})
    choice = min(cv_results, key=lambda item: item["cv_median_ape_percent"])
    fitted = _fit(z[train_idx], standard_y[train_idx], choice["family"],
                  choice["regularization"], choice["gamma"])
    predicted = np.exp(_predict_z(fitted, z[test_idx]) * log_scale + log_mean)
    errors = _percent_errors(y[test_idx], predicted)
    metrics = {
        "test_design_ids": [records[i]["design_id"] for i in test_idx],
        "train_design_count": len(train_idx), "test_design_count": len(test_idx),
        "median_ape_percent_by_mode": np.median(errors, axis=0).tolist(),
        "p95_ape_percent_by_mode": np.percentile(errors, 95, axis=0).tolist(),
        "max_ape_percent_by_mode": np.max(errors, axis=0).tolist(),
        "test_actual_hz": y[test_idx].tolist(),
        "test_predicted_hz": predicted.tolist(),
        "meets_original_accuracy_goal": bool(np.all(np.median(errors, axis=0) <= 5)
                                             and np.all(np.percentile(errors, 95, axis=0) <= 10)),
    }
    return {
        "schema_version": 1, "source": source,
        "model_description": "NumPy quadratic ridge or RBF kernel ridge; not GPR, ANN, or XGBoost",
        "input_keys": keys, "input_ranges": {key: metadata["ranges"][key] for key in keys},
        "target_count": y.shape[1], "seed": seed,
        "fit": fitted, "log_target_mean": log_mean.tolist(),
        "log_target_scale": log_scale.tolist(), "chosen_cv": choice,
        "cv_results": cv_results, "held_out_metrics": metrics,
    }


def predict(model: dict[str, Any], design: dict[str, Any]) -> list[float]:
    keys = model["input_keys"]
    limits = np.array([model["input_ranges"][key] for key in keys], dtype=float)
    values = np.array([float(design[key]) for key in keys])
    if not np.isfinite(values).all():
        raise ValueError("design inputs must be finite")
    outside = [key for key, value, (lo, hi) in zip(keys, values, limits)
               if value < lo or value > hi]
    if outside:
        raise ValueError("out of trained range: " + ", ".join(outside))
    z = (2 * (values - limits[:, 0]) / (limits[:, 1] - limits[:, 0]) - 1)[None, :]
    standardized = _predict_z(model["fit"], z)[0]
    result = np.exp(standardized * np.asarray(model["log_target_scale"])
                    + np.asarray(model["log_target_mean"]))
    return [float(value) for value in result]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fitting = sub.add_parser("train")
    fitting.add_argument("--dataset", type=Path, required=True)
    fitting.add_argument("--out", type=Path, required=True)
    fitting.add_argument("--seed", type=int, default=42)
    inference = sub.add_parser("predict")
    inference.add_argument("--model", type=Path, required=True)
    inference.add_argument("--design", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        model = train(json.loads(args.dataset.read_text()), args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(model, indent=2) + "\n")
        print(json.dumps({"chosen_cv": model["chosen_cv"],
                          "held_out_metrics": model["held_out_metrics"]}, indent=2))
    else:
        model = json.loads(args.model.read_text())
        print(json.dumps({"predicted_frequencies_hz": predict(model, json.loads(args.design.read_text())),
                          "source": model["source"]}, indent=2))


if __name__ == "__main__":
    main()
