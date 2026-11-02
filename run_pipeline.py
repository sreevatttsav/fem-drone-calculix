"""Run the reproducible concept geometry -> FEM -> surrogate workflow.

Choose native CalculiX or the legacy local beam solver explicitly. Neither
backend executes ANSYS or substitutes for a physical impact test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
from pathlib import Path

import numpy as np

from campbell import check_design, render_svg
from fem import mesh_convergence, solve_modal
from geometry import DEFAULT_CONFIG, export, load_design
from mechanics import report as mechanics_report
from surrogate import train
from sweep import SOURCES, generate_dataset, solve_design


def _save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def run(config_path: Path, out_dir: Path, count: int, seed: int,
        elements_per_arm: int, compare_ml: bool = False,
        backend: str = "calculix", mesh_size_m: float = 0.01,
        ccx_path: str = "ccx") -> dict:
    if backend not in SOURCES:
        raise ValueError(f"unsupported backend: {backend}")
    if compare_ml and count < 50:
        raise ValueError("named-model comparison requires at least 50 successful designs")
    if backend == "calculix" and shutil.which(ccx_path) is None:
        raise FileNotFoundError(
            f"CalculiX executable '{ccx_path}' was not found; install ccx or pass --ccx-path. "
            "Use --backend beam only if you intentionally want the legacy screening model.")
    prior_manifest = out_dir / "run_manifest.json"
    if prior_manifest.exists():
        prior_source = json.loads(prior_manifest.read_text()).get("simulation_source")
        if prior_source != SOURCES[backend]:
            raise ValueError(
                f"output directory contains {prior_source} results; use a separate directory for {SOURCES[backend]}")
    design = load_design(config_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    export(design, out_dir / "geometry")
    _save_json(out_dir / "mechanics.json", mechanics_report(design))
    if backend == "beam":
        modal = solve_modal(design, elements_per_arm=elements_per_arm)
        convergence = mesh_convergence(design, coarse=elements_per_arm,
                                       fine=2 * elements_per_arm)
    else:
        modal = solve_design(design, backend, out_dir / "baseline_calculix",
                             elements_per_arm, 6, mesh_size_m, ccx_path)
        fine = solve_design(design, backend, out_dir / "baseline_calculix_fine",
                            elements_per_arm, 6, mesh_size_m / 2, ccx_path)
        coarse_hz = np.asarray(modal["flexible_frequencies_hz"])
        fine_hz = np.asarray(fine["flexible_frequencies_hz"])
        differences = 100 * np.abs(coarse_hz - fine_hz) / fine_hz
        convergence = {"coarse": modal, "fine": fine,
                       "relative_difference_percent": differences.tolist(),
                       "meets_two_percent_first_five": bool(np.all(differences[:5] <= 2)),
                       "coarse_mesh_size_m": mesh_size_m,
                       "fine_mesh_size_m": mesh_size_m / 2}
    _save_json(out_dir / "baseline_fem.json", modal)
    _save_json(out_dir / "mesh_convergence.json", convergence)
    if backend == "calculix" and count > 10 and not convergence["meets_two_percent_first_five"]:
        raise RuntimeError("CalculiX mesh convergence exceeds 2% for a target mode; inspect mesh_convergence.json before a larger sweep")
    dataset = generate_dataset(design, count=count, seed=seed,
                               elements_per_arm=elements_per_arm,
                               output_dir=out_dir / "data", backend=backend,
                               mesh_size_m=mesh_size_m, ccx_path=ccx_path)
    model = check = None
    held_out_summary = None
    if count >= 20:
        model = train(dataset, seed=seed)
        _save_json(out_dir / "surrogate_model.json", model)
        check = check_design(model, design)
        _save_json(out_dir / "design_check.json", check)
        (out_dir / "campbell.svg").write_text(render_svg(check))
        held_out = model["held_out_metrics"]
        held_out_summary = {
            "train_design_count": held_out["train_design_count"],
            "test_design_count": held_out["test_design_count"],
            "median_ape_percent_by_mode": held_out["median_ape_percent_by_mode"],
            "p95_ape_percent_by_mode": held_out["p95_ape_percent_by_mode"],
            "meets_original_accuracy_goal": held_out["meets_original_accuracy_goal"],
        }
    manifest = {
        "purpose": "unvalidated_concept_modal_workflow",
        "simulation_source": SOURCES[backend],
        "backend": backend,
        "solver_version": modal.get("solver_version"),
        "solver_executable": modal.get("solver_executable"),
        "mesh_generator": modal.get("mesh", {}).get("generator"),
        "baseline_source_files": modal.get("source_files", {}),
        "baseline_mass_kg": modal.get("total_mass_kg"),
        "baseline_flexible_frequencies_hz": modal.get("flexible_frequencies_hz"),
        "ansys_solve_performed": False,
        "calculix_solve_performed": backend == "calculix",
        "physical_test_performed": False,
        "design_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "seed": seed,
        "successful_designs": dataset["metadata"]["successful_designs"],
        "failed_designs": dataset["metadata"]["failed_designs"],
        "mesh_convergence_first_five_within_two_percent": convergence.get(
            "meets_two_percent_first_five", convergence.get("passes_2_percent")),
        "held_out_metrics": held_out_summary,
        "screen_passes": check["passes"] if check is not None else None,
        "pipeline_stage": "pilot_sweep_only" if model is None else "surrogate_and_screen_complete",
    }
    if compare_ml:
        import joblib
        from ml_compare import compare

        comparison, selected_model = compare(dataset, seed=seed)
        ml_out = out_dir / "ml_compare"
        _save_json(ml_out / "comparison.json", comparison)
        joblib.dump(selected_model, ml_out / "best_model.joblib")
        manifest["named_model_comparison"] = {
            "selected_family": comparison["selected_family"],
            "selected_configuration": comparison["selected_configuration"],
            "comparison_path": "ml_compare/comparison.json",
            "operational_predictor": "NumPy surrogate_model.json",
        }
    _save_json(out_dir / "run_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "build" / "run")
    parser.add_argument("--count", type=int, default=50,
                        help="number of successful designs, including baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--elements-per-arm", type=int, default=4)
    parser.add_argument("--backend", choices=tuple(SOURCES), default="calculix")
    parser.add_argument("--mesh-size-m", type=float, default=0.01)
    parser.add_argument("--ccx-path", default="ccx")
    parser.add_argument("--compare-ml", action="store_true",
                        help="also compare ANN, GPR, XGBoost; requires requirements-ml.txt")
    args = parser.parse_args()
    try:
        result = run(args.config, args.out, args.count, args.seed,
                     args.elements_per_arm, args.compare_ml,
                     args.backend, args.mesh_size_m, args.ccx_path)
    except FileNotFoundError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
