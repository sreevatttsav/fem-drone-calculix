"""Reproducible parameter sweep with separate beam and native CalculiX data."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from fem import solve_modal


SOURCES = {"beam": "local_linear_beam_FEM", "calculix": "calculix_ccx_native"}
DEGENERATE_RELATIVE_TOLERANCE = 0.01


DESIGN_KEYS = (
    "arm_length_m", "arm_width_m", "arm_thickness_m", "hub_width_m",
    "hub_plate_thickness_m", "motor_mass_kg", "payload_mass_kg",
    "E_long_Pa", "E_trans_Pa", "G_Pa", "rho_kg_m3",
)
DEFAULT_FRACTIONS = {
    "arm_length_m": (0.75, 1.25),
    "arm_width_m": (0.75, 1.25),
    "arm_thickness_m": (0.75, 1.25),
    "hub_width_m": (0.85, 1.15),
    "hub_plate_thickness_m": (0.8, 1.2),
    "motor_mass_kg": (0.75, 1.25),
    "payload_mass_kg": (0.5, 1.5),
    "E_long_Pa": (0.9, 1.1),
    "E_trans_Pa": (0.9, 1.1),
    "G_Pa": (0.9, 1.1),
    "rho_kg_m3": (0.95, 1.05),
}


def default_ranges(baseline: dict[str, Any]) -> dict[str, list[float]]:
    """Return a documented pilot range; measured/supplier values can override it."""
    result = {}
    for key, (lo, hi) in DEFAULT_FRACTIONS.items():
        center = float(baseline[key])
        if center <= 0:
            raise ValueError(f"{key} must be positive for a default range")
        result[key] = [center * lo, center * hi]
    return result


def solve_design(design: dict[str, Any], backend: str, case_dir: str | Path | None = None,
                 elements_per_arm: int = 4, flexible_count: int = 6,
                 mesh_size_m: float = 0.01, ccx_path: str = "ccx") -> dict[str, Any]:
    """Run one explicitly selected solver; never substitute another backend."""
    if backend == "beam":
        return solve_modal(design, elements_per_arm=elements_per_arm,
                           flexible_count=flexible_count)
    if backend == "calculix":
        if case_dir is None:
            raise ValueError("CalculiX requires a case output directory")
        from calculix_runner import solve_calculix

        result = solve_calculix(design, output_dir=Path(case_dir),
                                mesh_size_m=mesh_size_m,
                                flexible_count=flexible_count, ccx_path=ccx_path)
        if result.get("provenance") != SOURCES["calculix"]:
            raise ValueError("CalculiX solver result has incorrect provenance")
        return result
    raise ValueError(f"unsupported backend: {backend}")


def latin_hypercube(ranges: dict[str, list[float]], count: int, seed: int) -> list[dict[str, float]]:
    """Seeded Latin hypercube: exactly one sample per interval in each dimension."""
    if count < 1:
        raise ValueError("count must be positive")
    if not ranges:
        raise ValueError("at least one range is required")
    rng = np.random.default_rng(seed)
    columns = {}
    for key in sorted(ranges):
        if key not in DESIGN_KEYS:
            raise ValueError(f"unsupported sweep parameter: {key}")
        lo, hi = map(float, ranges[key])
        if not np.isfinite([lo, hi]).all() or lo <= 0 or hi <= lo:
            raise ValueError(f"invalid positive range for {key}: {ranges[key]}")
        unit_intervals = (rng.permutation(count) + rng.random(count)) / count
        columns[key] = lo + (hi - lo) * unit_intervals
    return [{key: float(columns[key][row]) for key in columns} for row in range(count)]


def _shape_vectors(result: dict[str, Any]) -> np.ndarray | None:
    for key in ("mode_shapes_tip_xyz", "tip_mode_shapes", "mode_shape_vectors", "mode_shapes_tip", "mode_vectors"):
        if key in result:
            arr = np.asarray(result[key], dtype=float)
            if arr.ndim >= 2:
                return arr.reshape(arr.shape[0], -1)
    return None


def _mac(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.dot(a, a) * np.dot(b, b))
    return float(np.dot(a, b) ** 2 / denominator) if denominator > 0 else 0.0


def _subspace_mac(a: np.ndarray, b: np.ndarray) -> float:
    """Basis-independent overlap of two two-dimensional mode subspaces."""
    qa = np.linalg.qr(a.T)[0][:, :2]
    qb = np.linalg.qr(b.T)[0][:, :2]
    return float(np.sum((qa.T @ qb) ** 2) / 2)


def match_modes(reference: np.ndarray | None, candidate: np.ndarray | None,
                frequencies: list[float], count: int = 3,
                reference_frequencies: list[float] | None = None) -> tuple[list[float], list[float], str]:
    """Track target modes by tip-shape MAC; fail closed if matching is ambiguous."""
    if reference is None or candidate is None:
        raise ValueError("mode-shape vectors required for physical mode tracking")
    if reference.shape[1] != candidate.shape[1] or min(len(reference), len(candidate)) < count:
        raise ValueError("incompatible or insufficient mode-shape vectors")
    if len(frequencies) < len(candidate):
        raise ValueError("frequency and mode-shape counts differ")
    if (reference_frequencies is not None and count == 3 and len(reference) >= 4
            and len(candidate) >= 4 and
            abs(reference_frequencies[2] - reference_frequencies[3])
            <= DEGENERATE_RELATIVE_TOLERANCE * reference_frequencies[2]):
        # The symmetric X-frame has a degenerate mode doublet. Its individual
        # eigenvectors may rotate arbitrarily; only the two-dimensional space
        # and shared frequency are meaningful.
        pairs = [(j, j + 1) for j in range(len(frequencies) - 1)
                 if abs(frequencies[j] - frequencies[j + 1])
                 <= DEGENERATE_RELATIVE_TOLERANCE * frequencies[j]]
        if not pairs:
            raise ValueError("reference degenerate doublet has no candidate counterpart")
        pair = max(pairs, key=lambda p: _subspace_mac(reference[2:4], candidate[list(p)]))
        available = [j for j in range(len(candidate)) if j not in pair]
        assignments = itertools.permutations(available, 2)
        first = max(assignments, key=lambda p: sum(_mac(reference[i], candidate[j]) for i, j in enumerate(p)))
        quality = [_mac(reference[i], candidate[j]) for i, j in enumerate(first)]
        quality.append(_subspace_mac(reference[2:4], candidate[list(pair)]))
        return ([float(frequencies[j]) for j in first] +
                [float(np.mean([frequencies[j] for j in pair]))], quality,
                "tip_MAC_first_two_plus_degenerate_subspace_MAC")
    scores = np.array([[_mac(a, b) for b in candidate] for a in reference[:count]])
    assignments = itertools.permutations(range(len(candidate)), count)
    best = max(assignments, key=lambda p: sum(scores[i, j] for i, j in enumerate(p)))
    quality = [float(scores[i, j]) for i, j in enumerate(best)]
    return [float(frequencies[j]) for j in best], quality, "tip_displacement_MAC"


def generate_dataset(baseline: dict[str, Any], count: int = 50, seed: int = 42,
                     ranges: dict[str, list[float]] | None = None,
                     elements_per_arm: int = 4, flexible_count: int = 6,
                     target_count: int = 3, output_dir: str | Path | None = None,
                     backend: str = "beam", mesh_size_m: float = 0.01,
                     ccx_path: str = "ccx") -> dict[str, Any]:
    """Solve the baseline and sampled designs, retaining failed runs separately.

    `count` is total successful unique designs including the baseline. Failed runs
    are replaced by additional deterministic LHS batches (up to 4 * count tries).
    """
    if count < 2:
        raise ValueError("count must include baseline and at least one sweep design")
    if target_count < 1 or flexible_count < target_count:
        raise ValueError("target_count must fit flexible_count")
    if backend not in SOURCES:
        raise ValueError(f"unsupported backend: {backend}")
    if backend == "calculix" and output_dir is None:
        raise ValueError("CalculiX sweep requires output_dir for auditable solver files")
    source = SOURCES[backend]
    if output_dir is not None:
        existing = Path(output_dir) / "dataset.json"
        if existing.exists():
            prior_source = json.loads(existing.read_text()).get("metadata", {}).get("source")
            if prior_source != source:
                raise ValueError(f"output directory contains {prior_source} data; use a separate directory for {source}")
    ranges = default_ranges(baseline) if ranges is None else ranges
    cases_root = Path(output_dir) / "solver_cases" if output_dir is not None else None
    baseline_result = solve_design(baseline, backend,
                                   cases_root / "D0000" if cases_root is not None else None,
                                   elements_per_arm, flexible_count, mesh_size_m, ccx_path)
    baseline_shape = _shape_vectors(baseline_result)
    base_freqs = baseline_result["flexible_frequencies_hz"]
    if baseline_shape is None:
        raise ValueError(f"{backend} solver did not provide tip mode shapes for tracking")
    if len(base_freqs) < target_count:
        raise ValueError("baseline has insufficient flexible modes")
    baseline_record = {
        "design_id": "D0000", "inputs": dict(baseline),
        "frequencies_hz": [float(x) for x in base_freqs[:target_count]],
        "raw_flexible_frequencies_hz": [float(x) for x in base_freqs],
        "mode_mac": [1.0] * target_count,
        "mode_tracking": "baseline_tip_displacement_MAC",
        "mode_shapes_tip_xyz": baseline_result["mode_shapes_tip_xyz"],
        "mode_descriptors": baseline_result.get("mode_descriptors", []),
        "rigid_frequencies_hz": baseline_result.get("rigid_frequencies_hz", []),
        "total_mass_kg": baseline_result.get("total_mass_kg"),
        "solver": source, "mesh": baseline_result.get("mesh", {}),
        "solver_provenance": baseline_result.get("provenance"),
        "solver_version": baseline_result.get("solver_version"),
        "source_files": baseline_result.get("source_files", {}),
    }
    records, failures = [baseline_record], []
    tried = 0
    candidates = []
    for batch in range(4):
        candidates.extend(latin_hypercube(ranges, count - 1, seed + batch))
    for offsets in candidates:
        if len(records) >= count:
            break
        tried += 1
        design = copy.deepcopy(baseline)
        design.update(offsets)
        # Preserve a snug clamped-arm concept as arm stock thickness varies.
        if "geometry" in design:
            design["geometry"]["hub_plate_gap_m"] = design["arm_thickness_m"]
        design_id = f"D{tried:04d}"
        try:
            result = solve_design(design, backend,
                                  cases_root / design_id if cases_root is not None else None,
                                  elements_per_arm, flexible_count, mesh_size_m, ccx_path)
            freqs = [float(x) for x in result["flexible_frequencies_hz"]]
            tracked, mac, method = match_modes(baseline_shape, _shape_vectors(result),
                                               freqs, target_count, base_freqs)
            if min(mac) < 0.5:
                raise ValueError(f"ambiguous mode tracking: minimum MAC {min(mac):.3f}")
            if not np.isfinite(tracked).all() or min(tracked) <= 0:
                raise ValueError("nonpositive or nonfinite modal frequency")
            records.append({
                "design_id": design_id, "inputs": design,
                "frequencies_hz": tracked,
                "raw_flexible_frequencies_hz": freqs,
                "mode_mac": mac, "mode_tracking": method,
                "mode_shapes_tip_xyz": result["mode_shapes_tip_xyz"],
                "mode_descriptors": result.get("mode_descriptors", []),
                "rigid_frequencies_hz": result.get("rigid_frequencies_hz", []),
                "total_mass_kg": result.get("total_mass_kg"),
                "solver": source, "mesh": result.get("mesh", {}),
                "solver_provenance": result.get("provenance"),
                "solver_version": result.get("solver_version"),
                "source_files": result.get("source_files", {}),
            })
        except (ValueError, RuntimeError, KeyError, np.linalg.LinAlgError) as exc:
            failures.append({"design_id": design_id, "inputs": design, "error": str(exc)})
    payload = {
        "metadata": {
            "schema_version": 1, "source": source, "backend": backend,
            "not_ansys_or_test_data": True, "seed": seed, "requested_successes": count,
            "successful_designs": len(records), "failed_designs": len(failures),
            "attempted_samples": tried, "ranges": ranges,
            "elements_per_arm": elements_per_arm, "flexible_count": flexible_count,
            "mesh_size_m": mesh_size_m if backend == "calculix" else None,
            "ccx_path": ccx_path if backend == "calculix" else None,
            "target_count": target_count,
            "degenerate_mode_tolerance_fraction": DEGENERATE_RELATIVE_TOLERANCE,
            "target_description": "first two modes by tip MAC; degenerate third/fourth doublet by subspace MAC when present",
        },
        "records": records,
    }
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "dataset.json").write_text(json.dumps(payload, indent=2) + "\n")
        (out / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    if len(records) < count:
        raise RuntimeError(f"only {len(records)}/{count} successful designs; inspect failures")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="SI-unit design JSON")
    parser.add_argument("--ranges", type=Path, help="JSON object of [minimum, maximum] ranges")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--elements-per-arm", type=int, default=4)
    parser.add_argument("--backend", choices=tuple(SOURCES), default="calculix")
    parser.add_argument("--mesh-size-m", type=float, default=0.01)
    parser.add_argument("--ccx-path", default="ccx")
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    ranges = json.loads(args.ranges.read_text()) if args.ranges else None
    try:
        result = generate_dataset(baseline, args.count, args.seed, ranges,
                                  args.elements_per_arm, output_dir=args.out,
                                  backend=args.backend, mesh_size_m=args.mesh_size_m,
                                  ccx_path=args.ccx_path)
    except FileNotFoundError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result["metadata"], indent=2))


if __name__ == "__main__":
    main()
