"""Free-free 3D Euler-Bernoulli beam model of a symmetric X quadrotor.

All inputs and outputs use SI units. This is a screening model: the central
plate is rigid and arm roots are rigidly attached to its corners. Joint
compliance, laminate coupling, motor housings, and propeller dynamics require
the higher-fidelity model and physical validation described in docs/fem_method.md.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


REQUIRED = (
    "arm_length_m", "arm_width_m", "arm_thickness_m", "hub_width_m",
    "hub_plate_thickness_m", "motor_mass_kg", "payload_mass_kg",
    "E_long_Pa", "E_trans_Pa", "G_Pa", "rho_kg_m3", "rpm_min",
    "rpm_max", "blades",
)


def _validate(design: dict[str, Any], elements_per_arm: int, flexible_count: int) -> dict[str, Any]:
    missing = [key for key in REQUIRED if key not in design]
    if missing:
        raise ValueError(f"Missing design fields: {', '.join(missing)}")
    values = {key: float(design[key]) for key in REQUIRED}
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("Design fields must be finite")
    positive = REQUIRED[:5] + REQUIRED[7:11]
    if any(values[key] <= 0 for key in positive):
        raise ValueError("Geometry, elastic moduli, and density must be positive")
    if values["motor_mass_kg"] < 0 or values["payload_mass_kg"] < 0:
        raise ValueError("Masses cannot be negative")
    if values["rpm_min"] < 0 or values["rpm_max"] <= values["rpm_min"]:
        raise ValueError("Expected 0 <= rpm_min < rpm_max")
    if values["blades"] < 1 or int(values["blades"]) != values["blades"]:
        raise ValueError("blades must be a positive integer")
    if not isinstance(elements_per_arm, int) or elements_per_arm < 1:
        raise ValueError("elements_per_arm must be a positive integer")
    if not isinstance(flexible_count, int) or flexible_count < 1:
        raise ValueError("flexible_count must be a positive integer")
    if "geometry" in design:
        if not isinstance(design["geometry"], dict):
            raise ValueError("geometry must be an object")
        values["geometry"] = design["geometry"]
    return values


def _add_block(matrix: np.ndarray, ids: list[int], block: np.ndarray) -> None:
    matrix[np.ix_(ids, ids)] += block


def _local_matrices(length: float, width: float, thickness: float,
                    young: float, shear: float, density: float) -> tuple[np.ndarray, np.ndarray]:
    """Return local beam K and consistent M for [u,v,w,rx,ry,rz] at each end."""
    area = width * thickness
    iy = width * thickness**3 / 12  # vertical bending (w, ry)
    iz = thickness * width**3 / 12  # in-plane bending (v, rz)
    # Saint-Venant torsion constant for rectangular section, width >= thickness.
    a, b = sorted((width, thickness), reverse=True)
    torsion = a * b**3 * (1 / 3 - 0.21 * b / a * (1 - b**4 / (12 * a**4)))
    polar_area = iy + iz
    k = np.zeros((12, 12), dtype=float)
    m = np.zeros((12, 12), dtype=float)
    _add_block(k, [0, 6], young * area / length * np.array([[1, -1], [-1, 1]]))
    _add_block(m, [0, 6], density * area * length / 6 * np.array([[2, 1], [1, 2]]))
    _add_block(k, [3, 9], shear * torsion / length * np.array([[1, -1], [-1, 1]]))
    _add_block(m, [3, 9], density * polar_area * length / 6 * np.array([[2, 1], [1, 2]]))
    kb = np.array([
        [12, 6 * length, -12, 6 * length],
        [6 * length, 4 * length**2, -6 * length, 2 * length**2],
        [-12, -6 * length, 12, -6 * length],
        [6 * length, 2 * length**2, -6 * length, 4 * length**2],
    ]) / length**3
    mb = np.array([
        [156, 22 * length, 54, -13 * length],
        [22 * length, 4 * length**2, 13 * length, -3 * length**2],
        [54, 13 * length, 156, -22 * length],
        [-13 * length, -3 * length**2, -22 * length, 4 * length**2],
    ]) / 420
    _add_block(k, [1, 5, 7, 11], young * iz * kb)
    _add_block(m, [1, 5, 7, 11], density * area * length * mb)
    flip = np.diag([1, -1, 1, -1])
    _add_block(k, [2, 4, 8, 10], young * iy * flip @ kb @ flip)
    _add_block(m, [2, 4, 8, 10], density * area * length * flip @ mb @ flip)
    return k, m


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def _assemble(design: dict[str, Any], n: int) -> tuple[np.ndarray, np.ndarray, list[int], float]:
    width = design["arm_width_m"]
    thickness = design["arm_thickness_m"]
    geometry = design.get("geometry", {})
    overlap = float(geometry.get("arm_clamp_overlap_m", 0.0))
    gap = float(geometry.get("hub_plate_gap_m", design["arm_thickness_m"]))
    mount_diameter = float(geometry.get("motor_mount_diameter_m", 0.0))
    mount_thickness = float(geometry.get("motor_mount_thickness_m", 0.0))
    if overlap < 0 or overlap >= design["hub_width_m"] / math.sqrt(2):
        raise ValueError("Invalid arm_clamp_overlap_m")
    if gap < design["arm_thickness_m"] or mount_diameter < 0 or mount_thickness < 0:
        raise ValueError("Invalid hub gap or mount dimensions")
    arm_length = design["arm_length_m"]
    root_radius = design["hub_width_m"] / math.sqrt(2)
    dof = 6 + 4 * n * 6
    k = np.zeros((dof, dof))
    m = np.zeros((dof, dof))
    plate_mass = design["rho_kg_m3"] * design["hub_width_m"]**2 * design["hub_plate_thickness_m"]
    hub_mass = 2 * plate_mass
    overlap_mass = design["rho_kg_m3"] * width * thickness * overlap
    center_mass = hub_mass + design["payload_mass_kg"] + 4 * overlap_mass
    m[:3, :3] += np.eye(3) * center_mass
    hub_w = design["hub_width_m"]
    hub_t = design["hub_plate_thickness_m"]
    # Centered payload is represented as a point; its rotational inertia is zero.
    plate_z = (gap + hub_t) / 2
    m[3, 3] += hub_mass * ((hub_w**2 + hub_t**2) / 12 + plate_z**2)
    m[4, 4] += hub_mass * ((hub_w**2 + hub_t**2) / 12 + plate_z**2)
    m[5, 5] += hub_mass * hub_w**2 / 6
    mount_mass = design["rho_kg_m3"] * math.pi * (mount_diameter / 2)**2 * mount_thickness
    local_k, local_m = _local_matrices(arm_length / n, width, thickness,
                                      design["E_long_Pa"], design["G_Pa"],
                                      design["rho_kg_m3"])
    tips: list[int] = []
    for arm in range(4):
        angle = math.pi / 4 + arm * math.pi / 2
        ex = np.array([math.cos(angle), math.sin(angle), 0.0])
        ey = np.array([-math.sin(angle), math.cos(angle), 0.0])
        ez = np.array([0.0, 0.0, 1.0])
        if overlap:
            # The clamped overlap is part of the rigid hub. Its section mass
            # remains in the assembly, including its offset rotary inertia.
            centroid_r = root_radius - overlap / 2
            second_moment = (
                (centroid_r**2 + overlap**2 / 12) * np.outer(ex, ex)
                + width**2 / 12 * np.outer(ey, ey)
                + thickness**2 / 12 * np.outer(ez, ez)
            )
            m[3:6, 3:6] += overlap_mass * (
                np.trace(second_moment) * np.eye(3) - second_moment)
        rotation = np.stack([ex, ey, ez])
        frame = np.zeros((12, 12))
        for offset in (0, 6):
            frame[offset:offset + 3, offset:offset + 3] = rotation
            frame[offset + 3:offset + 6, offset + 3:offset + 6] = rotation
        kg = frame.T @ local_k @ frame
        mg = frame.T @ local_m @ frame
        root = root_radius * ex
        rigid_root = np.zeros((6, 6))
        rigid_root[:3, :3] = np.eye(3)
        rigid_root[:3, 3:6] = -_skew(root)
        rigid_root[3:6, 3:6] = np.eye(3)
        for segment in range(n):
            mapping = np.zeros((12, dof))
            if segment == 0:
                mapping[:6, :6] = rigid_root
            else:
                prev = 6 + (arm * n + segment - 1) * 6
                mapping[:6, prev:prev + 6] = np.eye(6)
            nxt = 6 + (arm * n + segment) * 6
            mapping[6:12, nxt:nxt + 6] = np.eye(6)
            k += mapping.T @ kg @ mapping
            m += mapping.T @ mg @ mapping
        tip = 6 + (arm * n + n - 1) * 6
        tips.append(tip)
        m[tip:tip + 3, tip:tip + 3] += np.eye(3) * (design["motor_mass_kg"] + mount_mass)
    total_mass = center_mass + 4 * (design["rho_kg_m3"] * width * thickness * arm_length + design["motor_mass_kg"] + mount_mass)
    return k, m, tips, total_mass


def solve_modal(design: dict[str, Any], elements_per_arm: int = 4,
                flexible_count: int = 6) -> dict[str, Any]:
    """Solve the unconstrained model; return modes ordered by frequency.

    Six zero eigenvalues represent rigid-body motion. The reported flexible
    modes start at eigenvalue index six. Mode-shape tip samples have arm order
    45°, 135°, 225°, 315° and XYZ coordinates for each arm.
    """
    d = _validate(design, elements_per_arm, flexible_count)
    k, m, tips, total_mass = _assemble(d, elements_per_arm)
    if flexible_count > k.shape[0] - 6:
        raise ValueError("flexible_count exceeds available degrees of freedom")
    chol = np.linalg.cholesky(m)
    inv_chol = np.linalg.solve(chol, np.eye(chol.shape[0]))
    standard = inv_chol @ k @ inv_chol.T
    standard = (standard + standard.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(standard)
    scale = max(float(eigenvalues[-1]), 1.0)
    eigenvalues[np.abs(eigenvalues) < 1e-9 * scale] = 0
    if np.any(eigenvalues[6:] <= 0):
        raise RuntimeError("Unexpected non-positive flexible eigenvalue; check model connectivity")
    vectors = inv_chol.T @ eigenvectors
    frequencies = np.sqrt(np.maximum(eigenvalues, 0)) / (2 * math.pi)
    selected = frequencies[6:6 + flexible_count]
    samples: list[list[float]] = []
    descriptors: list[str] = []
    for mode in range(6, 6 + flexible_count):
        sample = np.concatenate([vectors[tip:tip + 3, mode] for tip in tips])
        norm = np.linalg.norm(sample)
        sample = sample / norm if norm > 0 else sample
        vertical = sum(sample[2::3] ** 2)
        if vertical >= 0.8:
            label = "out_of_plane"
        elif vertical <= 0.2:
            label = "in_plane"
        else:
            label = "mixed"
        samples.append(sample.tolist())
        descriptors.append(label)
    return {
        "rigid_frequencies_hz": frequencies[:6].tolist(),
        "flexible_frequencies_hz": selected.tolist(),
        "frequencies_hz": selected.tolist(),
        "mode_descriptors": descriptors,
        "mode_shapes_tip_xyz": samples,
        "total_mass_kg": float(total_mass),
        "mesh": {"elements_per_arm": elements_per_arm, "total_elements": 4 * elements_per_arm,
                 "degrees_of_freedom": int(k.shape[0]), "formulation": "3D_Euler_Bernoulli_free_free"},
        "provenance": "numpy_beam_screening_model",
    }


def mesh_convergence(design: dict[str, Any], coarse: int = 4, fine: int = 8,
                     flexible_count: int = 5) -> dict[str, Any]:
    """Compare frequency-ordered flexible modes on two nested arm meshes."""
    if fine <= coarse:
        raise ValueError("fine must exceed coarse")
    low = solve_modal(design, coarse, flexible_count)
    high = solve_modal(design, fine, flexible_count)
    a = np.array(low["flexible_frequencies_hz"])
    b = np.array(high["flexible_frequencies_hz"])
    differences = 100 * np.abs(a - b) / b
    return {"coarse": low, "fine": high, "relative_difference_percent": differences.tolist(),
            "max_relative_difference_percent": float(np.max(differences)),
            "passes_2_percent": bool(np.all(differences <= 2))}
