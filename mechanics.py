"""Hand estimates for a single X-quad arm with lumped motor and mount masses.

Use this model to check FEM trends. It intentionally omits hub/joint compliance,
motor-mount stiffness, coupling between arms, payload motion and damping.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from geometry import DEFAULT_CONFIG, load_design


def beam_estimate(design: dict) -> dict:
    """Rayleigh estimate for the first bending mode of a cantilever arm.

    All calculations use SI units. The effective vibrating arm mass is
    0.236*m_arm, a standard first-mode approximation for a uniform beam.
    The axial laminate modulus along the arm controls bending in either plane.
    """
    length = float(design["arm_length_m"])
    width = float(design["arm_width_m"])
    thickness = float(design["arm_thickness_m"])
    E = float(design["E_long_Pa"])
    arm_mass = float(design["rho_kg_m3"]) * length * width * thickness
    geometry = design.get("geometry", {})
    mount_radius = float(geometry.get("motor_mount_diameter_m", 0.0)) / 2
    mount_thickness = float(geometry.get("motor_mount_thickness_m", 0.0))
    mount_mass = float(design["rho_kg_m3"]) * math.pi * mount_radius**2 * mount_thickness
    effective_mass = float(design["motor_mass_kg"]) + mount_mass + 0.236*arm_mass
    if effective_mass <= 0:
        raise ValueError("Effective mass must be positive")
    inertias = {
        "out_of_plane": width*thickness**3/12,
        "in_plane": thickness*width**3/12,
    }
    modes = {}
    for name, inertia in inertias.items():
        stiffness = 3*E*inertia/length**3
        modes[name] = {"second_moment_m4": inertia, "tip_stiffness_N_m": stiffness,
                       "frequency_Hz": math.sqrt(stiffness/effective_mass)/(2*math.pi)}
    return {"beam_span_m": length, "arm_mass_kg": arm_mass,
            "mount_mass_kg": mount_mass,
            "effective_mass_kg": effective_mass, "modes": modes}


def sensitivity(design: dict, relative_step: float = 0.001) -> dict:
    """Central finite difference elasticities d ln(f) / d ln(parameter)."""
    if not 0 < relative_step < 0.1:
        raise ValueError("relative_step must be between 0 and 0.1")
    parameters = ("arm_length_m", "arm_width_m", "arm_thickness_m",
                  "E_long_Pa", "rho_kg_m3", "motor_mass_kg")
    result = {}
    for mode in ("out_of_plane", "in_plane"):
        result[mode] = {}
        for key in parameters:
            lo = dict(design)
            hi = dict(design)
            lo[key] = float(design[key])*(1-relative_step)
            hi[key] = float(design[key])*(1+relative_step)
            fl = beam_estimate(lo)["modes"][mode]["frequency_Hz"]
            fh = beam_estimate(hi)["modes"][mode]["frequency_Hz"]
            result[mode][key] = (math.log(fh)-math.log(fl))/(math.log(hi[key])-math.log(lo[key]))
    return result


def report(design: dict) -> dict:
    estimate = beam_estimate(design)
    estimate["log_sensitivity"] = sensitivity(design)
    estimate["assumptions"] = [
        "One uniform, fixed-root rectangular arm; hub and clamp are perfectly rigid.",
        "One motor and carbon motor-mount disk are point masses at the arm tip; their rotary inertia is omitted.",
        "0.236 of arm mass contributes to the first-mode effective mass.",
        "E_long_Pa is the effective beam-axis modulus for both bending directions.",
        "E_trans_Pa and G_Pa are retained for FEM but not used in this beam estimate.",
        "Payload mass is carried by the hub and omitted from the isolated-arm estimate.",
    ]
    return estimate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = report(load_design(args.config))
    output = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    else:
        print(output, end="")
