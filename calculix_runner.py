"""Run a native CalculiX free-free modal deck and parse its modal outputs.

The deck writer lives in calculix_mesh.py. This module never falls back to the
NumPy screening model when CalculiX is missing or fails.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


_NUMBER = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][+-]?\d+)?"
_VALUES = re.compile(_NUMBER)
_DAT_ROW = re.compile(r"^\s*(\d+)\s+(" + _NUMBER + r")\s+(" + _NUMBER + r")\s+(" + _NUMBER + r")(?:\s+(" + _NUMBER + r"))?\s*$")
_FRD_NODE = re.compile(r"^\s*-1\s+(\d+)\s+(.+)$")


def _frd_node_and_values(line: str) -> tuple[int, str] | None:
    """Decode an FRD -1 row with I10/I5 node field or spaced values."""
    prefix = re.match(r"^\s*-1", line)
    if not prefix:
        return None
    start = prefix.end()
    for width in (10, 5):
        field = line[start:start + width]
        if field.strip().isdigit():
            return int(field), line[start + width:]
    match = _FRD_NODE.match(line)
    return (int(match.group(1)), match.group(2)) if match else None


def _float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def parse_dat_frequencies(path: str | Path) -> list[float]:
    """Read cycles/time from the modal eigenvalue table in a .dat file."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    table_seen = False
    frequencies: dict[int, float] = {}
    in_table = False
    for line in lines:
        upper = line.upper()
        if "E I G E N V A L U E" in upper and "O U T P U T" in upper:
            table_seen = True
            in_table = True
            continue
        if not in_table:
            continue
        row = _DAT_ROW.match(line)
        if row:
            mode = int(row.group(1))
            frequency = _float(row.group(4))
            if not math.isfinite(frequency) or frequency < 0:
                raise ValueError(f"Invalid CalculiX frequency for mode {mode}")
            if mode in frequencies:
                raise ValueError(f"Duplicate CalculiX mode {mode}")
            frequencies[mode] = frequency
        elif frequencies and line.strip() and line.strip()[0].isalpha():
            in_table = False
    if not table_seen or not frequencies:
        raise ValueError("No CalculiX eigenvalue table found in .dat output")
    if sorted(frequencies) != list(range(1, len(frequencies) + 1)):
        raise ValueError("CalculiX .dat mode numbers are not consecutive from 1")
    return [frequencies[index] for index in range(1, len(frequencies) + 1)]


def parse_frd_displacements(path: str | Path, tip_node_ids: list[int] | tuple[int, ...]) -> list[list[float]]:
    """Read each ASCII FRD DISP block as normalized tip XYZ displacements.

    CalculiX writes consecutive fields without spaces for some negative
    values, so the parser tokenizes exponential numbers instead of splitting
    solely on whitespace. Four tip nodes must be supplied in azimuth order.
    """
    if len(tip_node_ids) != 4 or len(set(tip_node_ids)) != 4:
        raise ValueError("Expected four distinct tip node IDs")
    wanted = set(int(node) for node in tip_node_ids)
    modes: list[list[float]] = []
    current: dict[int, list[float]] | None = None
    with Path(path).open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            stripped = line.lstrip()
            if stripped.startswith("-4 "):
                if current is not None:
                    raise ValueError("FRD DISP block missing -3 terminator")
                if re.match(r"-4\s+DISP\b", stripped):
                    current = {}
                continue
            if current is None:
                continue
            if stripped.startswith("-3"):
                missing = wanted - current.keys()
                if missing:
                    raise ValueError(f"FRD DISP block missing tip nodes: {sorted(missing)}")
                sample = np.array([value for node in tip_node_ids for value in current[int(node)]], dtype=float)
                norm = float(np.linalg.norm(sample))
                if not math.isfinite(norm) or norm <= 0:
                    raise ValueError("FRD tip mode shape has zero or nonfinite norm")
                modes.append((sample / norm).tolist())
                current = None
                continue
            record = _frd_node_and_values(line)
            if record is None:
                continue
            node, value_text = record
            if node not in wanted:
                continue
            values = [_float(value) for value in _VALUES.findall(value_text)]
            if len(values) < 3 or any(not math.isfinite(value) for value in values[:3]):
                raise ValueError(f"Invalid FRD displacement for tip node {node}")
            if node in current:
                raise ValueError(f"Duplicate FRD displacement for tip node {node}")
            current[node] = values[:3]
    if current is not None:
        raise ValueError("FRD DISP block is unterminated")
    if not modes:
        raise ValueError("No ASCII DISP mode shapes found in .frd output")
    return modes


def _solver_version(binary: str, solve_output: str) -> str:
    match = re.search(r"(?:CalculiX|ccx)\s*(?:version|v)?\s*([0-9]+(?:\.[0-9]+)+)", solve_output, re.I)
    if match:
        return match.group(1)
    try:
        probe = subprocess.run([binary, "-v"], text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    match = re.search(r"(?:CalculiX|ccx)\s*(?:version|v)?\s*([0-9]+(?:\.[0-9]+)+)", probe.stdout or "", re.I)
    return match.group(1) if match else "unavailable"


def solve_calculix(design: dict[str, Any], output_dir: str | Path,
                   mesh_size_m: float = 0.012, flexible_count: int = 6,
                   ccx_path: str = "ccx", mesh_generator: str = "auto",
                   width_division_scale: float = 1.0,
                   hub_stiffness_multiplier: float = 1.0) -> dict[str, Any]:
    """Generate deck, execute ``ccx -i <stem>``, and parse all requested modes.

    The first six modes are expected to be free-free rigid modes. A failed
    run, absent executable, missing modes, or missing tip displacements raises
    an exception instead of returning an approximate result.
    """
    if not isinstance(flexible_count, int) or flexible_count < 1:
        raise ValueError("flexible_count must be a positive integer")
    if not math.isfinite(mesh_size_m) or mesh_size_m <= 0:
        raise ValueError("mesh_size_m must be positive and finite")
    binary = shutil.which(ccx_path)
    if binary is None:
        raise FileNotFoundError(f"CalculiX executable not found: {ccx_path}. Install ccx or pass ccx_path explicitly.")
    binary = str(Path(binary).resolve())
    from calculix_mesh import write_calculix_deck

    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    deck = write_calculix_deck(design, out, mesh_size_m=mesh_size_m,
                               mode_count=6 + flexible_count,
                               mesh_generator=mesh_generator,
                               width_division_scale=width_division_scale,
                               hub_stiffness_multiplier=hub_stiffness_multiplier)
    inp = Path(deck["inp_path"]).resolve()
    if not inp.exists() or inp.parent != out:
        raise RuntimeError("CalculiX deck writer returned an invalid input path")
    command = [binary, "-i", inp.stem]
    try:
        completed = subprocess.run(command, cwd=out, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, timeout=1800, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("CalculiX modal solve timed out after 1800 seconds") from exc
    (out / "ccx_stdout.log").write_text(completed.stdout or "", encoding="utf-8")
    if completed.returncode != 0:
        tail = (completed.stdout or "")[-1500:]
        raise RuntimeError(f"CalculiX failed with exit code {completed.returncode}. See ccx_stdout.log. Tail: {tail}")
    dat = inp.with_suffix(".dat")
    frd = inp.with_suffix(".frd")
    if not dat.exists() or not frd.exists():
        raise RuntimeError("CalculiX completed without both .dat and .frd modal outputs")
    frequencies = parse_dat_frequencies(dat)
    shapes = parse_frd_displacements(frd, deck["tip_node_ids"])
    needed = 6 + flexible_count
    if len(frequencies) < needed or len(shapes) < needed:
        raise RuntimeError(f"CalculiX returned {len(frequencies)} frequencies and {len(shapes)} DISP blocks; need {needed}")
    # A free-free assembly must have six frequencies very small relative to
    # the first flexible mode; otherwise a constraint or disconnected part is likely.
    first_flexible = frequencies[6]
    if first_flexible <= 0 or max(frequencies[:6]) > max(0.01, 0.01 * first_flexible):
        raise RuntimeError("CalculiX result does not show six near-zero free-free rigid modes")
    selected = frequencies[6:needed]
    selected_shapes = shapes[6:needed]
    labels = []
    for shape in selected_shapes:
        vertical = sum(shape[2::3][index]**2 for index in range(4))
        labels.append("out_of_plane" if vertical >= 0.8 else "in_plane" if vertical <= 0.2 else "mixed")
    return {
        "rigid_frequencies_hz": frequencies[:6],
        "flexible_frequencies_hz": selected,
        "frequencies_hz": selected,
        "mode_shapes_tip_xyz": selected_shapes,
        "mode_descriptors": labels,
        "total_mass_kg": float(deck["mass_estimate_kg"]),
        "mesh": deck["mesh"],
        "provenance": "calculix_ccx_native",
        "solver_version": _solver_version(binary, completed.stdout or ""),
        "solver_executable": binary,
        "source_files": {"inp": str(inp), "dat": str(dat), "frd": str(frd)},
    }
