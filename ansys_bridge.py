"""ANSYS Mechanical handoff and frequency-export reader.

This module does not execute ANSYS. It writes a review checklist and reads a
simple CSV export produced by an actual ANSYS modal solve.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from fem import _validate


def write_handoff(design: dict[str, Any], destination: str | Path) -> Path:
    """Write a human-reviewable Mechanical setup specification as JSON."""
    d = _validate(design, 4, 6)
    detail = {key: float(value) for key, value in design.get("geometry", {}).items()}
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    specification = {
        "provenance": "ansys_mechanical_handoff_only_not_solved",
        "unit_system": "SI: m, kg, Pa, Hz",
        "design": design,
        "geometry": {
            "layout": "Four identical radial X arms at 45, 135, 225, 315 degrees",
            "arm_span_from_hub_corner_m": d["arm_length_m"],
            "hub": "two square plates above and below the arm plane",
            "hub_plate_gap_m": detail.get("hub_plate_gap_m"),
            "arm_clamp_overlap_m": detail.get("arm_clamp_overlap_m"),
            "motor_mount_diameter_m": detail.get("motor_mount_diameter_m"),
            "motor_mount_thickness_m": detail.get("motor_mount_thickness_m"),
            "motor_centers": "at each arm tip",
        },
        "material": {
            "density_kg_m3": d["rho_kg_m3"],
            "E_long_Pa": d["E_long_Pa"],
            "E_trans_Pa": d["E_trans_Pa"],
            "G_Pa": d["G_Pa"],
            "note": "Verify laminate directions, Poisson ratios, ply stack, and joint properties with supplier/test data before a solid or shell solve.",
        },
        "analysis": [
            "Use free-free modal analysis (no supports) to represent flight.",
            "Bond or model the hub-arm clamps using the actual joint construction; inspect their stiffness sensitivity.",
            "Attach four point/remote motor masses at arm tips and one payload mass at its measured center of gravity.",
            "Request at least 12 modes so that six rigid modes and six flexible modes are available.",
            "Check six near-zero rigid modes, total mass, and mesh convergence of the first five flexible frequencies.",
            "Export mode number and frequency in Hz as CSV columns mode,frequency_hz; export mode shapes separately for physical identification.",
        ],
        "important_difference": "The NumPy beam solver assumes a perfectly rigid hub and arm clamps. The Mechanical model should test realistic joint and plate compliance.",
    }
    path.write_text(json.dumps(specification, indent=2) + "\n", encoding="utf-8")
    return path


def read_modal_export(csv_path: str | Path, rigid_count: int = 6,
                      flexible_count: int = 6) -> dict[str, Any]:
    """Read a Mechanical CSV with columns ``mode,frequency_hz``.

    The function preserves solver provenance and never silently substitutes
    beam-model frequencies for ANSYS results.
    """
    path = Path(csv_path)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"mode", "frequency_hz"}.issubset(rows[0]):
        raise ValueError("Expected CSV columns mode,frequency_hz and at least one row")
    parsed: list[tuple[int, float]] = []
    for row in rows:
        try:
            mode = int(row["mode"])
            frequency = float(row["frequency_hz"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid mode or frequency in ANSYS export") from exc
        if mode < 1 or frequency < 0 or not math.isfinite(frequency):
            raise ValueError("Mode numbers must be positive and frequencies finite and nonnegative")
        parsed.append((mode, frequency))
    parsed.sort()
    if [mode for mode, _ in parsed] != list(range(1, len(parsed) + 1)):
        raise ValueError("Mode numbers must be unique and consecutive from 1")
    if len(parsed) < rigid_count + flexible_count:
        raise ValueError("Export contains too few modes")
    frequencies = [frequency for _, frequency in parsed]
    return {
        "provenance": "ansys_mechanical_csv_import_unverified",
        "source_file": str(path.resolve()),
        "rigid_frequencies_hz": frequencies[:rigid_count],
        "flexible_frequencies_hz": frequencies[rigid_count:rigid_count + flexible_count],
        "mode_count": len(parsed),
        "note": "Verify six near-zero modes, geometry, material, masses, constraints, and mode shapes in ANSYS before treating results as validated.",
    }
