"""Run native CalculiX mesh studies and generate numerical evidence and figures."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np

from calculix_runner import solve_calculix
from geometry import load_design
from sweep import default_ranges


ROOT = Path(__file__).resolve().parents[1]


def save(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def solve(design: dict, work: Path, name: str, h: float, ccx: str,
          **options) -> dict:
    started = time.perf_counter()
    result = solve_calculix(design, work / name, mesh_size_m=h,
                            ccx_path=ccx, mesh_generator=options.pop("mesh_generator", "gmsh"),
                            **options)
    return {
        "mesh_size_m": h,
        "frequencies_hz": result["flexible_frequencies_hz"],
        "rigid_frequencies_hz": result["rigid_frequencies_hz"],
        "mass_kg": result["total_mass_kg"],
        "shell_elements": result["mesh"]["shell_element_count"],
        "node_count": result["mesh"]["node_count"],
        "wall_seconds": time.perf_counter() - started,
        "mesh_generator": result["mesh"]["generator"],
        "options": options,
    }


def difference_percent(a: list[float], b: list[float]) -> list[float]:
    return (100 * np.abs(np.asarray(a) - np.asarray(b)) / np.asarray(b)).tolist()


def three_mesh(rows: list[dict]) -> dict:
    if len(rows) != 3:
        raise ValueError("three mesh results required")
    h0, h1, h2 = (row["mesh_size_m"] for row in rows)
    if not math.isclose(h0 / h1, 2) or not math.isclose(h1 / h2, 2):
        raise ValueError("Richardson estimate requires successive half sizes")
    f0, f1, f2 = (np.asarray(row["frequencies_hz"]) for row in rows)
    order, limit, note = [], [], []
    for a, b, c in zip(f0, f1, f2):
        ratio = (a-b)/(b-c) if b != c else math.nan
        if not math.isfinite(ratio) or ratio <= 1:
            order.append(None); limit.append(None)
            note.append("nonmonotonic or insufficient asymptotic convergence")
            continue
        p = math.log(ratio, 2)
        f_inf = c + (c-b)/(2**p-1)
        if not math.isfinite(f_inf):
            order.append(None); limit.append(None)
            note.append("nonfinite extrapolation")
        else:
            order.append(p); limit.append(f_inf); note.append("estimated")
    return {
        "source": "calculix_ccx_native",
        "solver_cases": rows,
        "relative_change_25_to_12_5_percent": difference_percent(f0.tolist(),f1.tolist()),
        "relative_change_12_5_to_6_25_percent": difference_percent(f1.tolist(),f2.tolist()),
        "observed_order": order,
        "richardson_limit_hz": limit,
        "estimate_status": note,
        "physical_validation": False,
    }


def corner_designs(baseline: dict, ranges: dict) -> dict[str, dict]:
    cases = {
        "arm_thickness_min": {"arm_thickness_m": "low"},
        "arm_thickness_max": {"arm_thickness_m": "high"},
        "arm_length_min": {"arm_length_m": "low"},
        "arm_length_max": {"arm_length_m": "high"},
        "payload_mass_min": {"payload_mass_kg": "low"},
        "payload_mass_max": {"payload_mass_kg": "high"},
        "combined_soft_heavy": {
            "arm_thickness_m": "low", "arm_length_m": "high",
            "payload_mass_kg": "high", "E_long_Pa": "low",
            "E_trans_Pa": "low", "G_Pa": "low"},
        "combined_stiff_light": {
            "arm_thickness_m": "high", "arm_length_m": "low",
            "payload_mass_kg": "low", "E_long_Pa": "high",
            "E_trans_Pa": "high", "G_Pa": "high"},
    }
    designs = {}
    for name, selections in cases.items():
        design = copy.deepcopy(baseline)
        for key, side in selections.items():
            design[key] = ranges[key][0 if side == "low" else 1]
        design["geometry"]["hub_plate_gap_m"] = design["arm_thickness_m"]
        designs[name] = design
    return designs


def run_corner_check(config: dict, work: Path, ccx: str,
                     coarse_size_m: float = 0.025) -> dict:
    ranges=default_ranges(config)
    corner_rows=[]
    for name,design in corner_designs(config,ranges).items():
        coarse=solve(design,work,f"corner_{name}_coarse",coarse_size_m,ccx)
        fine=solve(design,work,f"corner_{name}_fine",coarse_size_m/2,ccx)
        change=difference_percent(coarse["frequencies_hz"],fine["frequencies_hz"])
        corner_rows.append({"name":name,"inputs":{k:design[k] for k in ranges},
                            "coarse":coarse,"fine":fine,
                            "change_percent_by_mode":change,
                            "max_target_change_percent":max(change[:6])})
    return {"source":"calculix_ccx_native","cases":corner_rows,
            "recommend_finer_mesh_for":[r["name"] for r in corner_rows
                                        if r["max_target_change_percent"]>2]}


def line_plot(rows: list[dict]) -> str:
    freqs=np.asarray([row["frequencies_hz"] for row in rows])
    hs=np.asarray([row["mesh_size_m"] for row in rows])
    low=float(np.min(freqs))*0.9; high=float(np.max(freqs))*1.05
    left,right,top,bottom=80,780,50,480
    x=lambda h: left+(h-hs.min())/(hs.max()-hs.min())*(right-left)
    y=lambda f: bottom-(f-low)/(high-low)*(bottom-top)
    colors=["#2563eb","#dc2626","#16a34a","#9333ea","#ea580c","#0891b2"]
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="550" viewBox="0 0 960 550">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="80" y="30" font-family="sans-serif" font-size="20">Baseline mesh convergence</text>',
           f'<path d="M{left},{top}V{bottom}H{right}" fill="none" stroke="#333"/>']
    for i in range(freqs.shape[1]):
        points=" ".join(f"{x(h):.1f},{y(f):.1f}" for h,f in zip(hs,freqs[:,i]))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colors[i]}" stroke-width="2"/>')
        for h,f in zip(hs,freqs[:,i]):
            parts.append(f'<circle cx="{x(h):.1f}" cy="{y(f):.1f}" r="4" fill="{colors[i]}"/>')
        parts.append(f'<text x="810" y="{90+i*28}" font-family="sans-serif" font-size="13" fill="{colors[i]}">Mode {i+1}</text>')
    for h in hs:
        parts.append(f'<text x="{x(h):.1f}" y="502" text-anchor="middle" font-family="sans-serif" font-size="12">{h*1000:g} mm</text>')
    parts += ['<text x="430" y="532" text-anchor="middle" font-family="sans-serif" font-size="13">Target element size</text>',
              '<text transform="translate(20 260) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="13">Frequency (Hz)</text>',
              '</svg>']
    return "\n".join(parts)+"\n"


def report_text(baseline: dict, corners: dict, hypotheses: dict) -> str:
    mode_lines=[]
    for i,(order,limit,status) in enumerate(zip(baseline["observed_order"],
                                                  baseline["richardson_limit_hz"],
                                                  baseline["estimate_status"]),1):
        mode_lines.append(f"| {i} | {order:.3f} | {limit:.3f} | {status} |"
                          if order is not None else f"| {i} | — | — | {status} |")
    maximum=max(row["max_target_change_percent"] for row in corners["cases"])
    failing=[row["name"] for row in corners["cases"] if row["max_target_change_percent"]>2]
    baseline_rows=["| Mode | 25 mm (Hz) | 12.5 mm (Hz) | 6.25 mm (Hz) | 25→12.5 change | 12.5→6.25 change |",
                   "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    a,b,c=(row["frequencies_hz"] for row in baseline["solver_cases"])
    for i,values in enumerate(zip(a,b,c,
                                  baseline["relative_change_25_to_12_5_percent"],
                                  baseline["relative_change_12_5_to_6_25_percent"]),1):
        x,y,z,d1,d2=values
        baseline_rows.append(f"| {i} | {x:.4f} | {y:.4f} | {z:.4f} | {d1:.3f}% | {d2:.3f}% |")
    corner_table=["| Corner | Largest six-mode change | Coarse/fine elements | Coarse/fine runtime |",
                  "| --- | ---: | ---: | ---: |"]
    for row in corners["cases"]:
        coarse,fine=row["coarse"],row["fine"]
        corner_table.append(
            f"| {row['name']} | {row['max_target_change_percent']:.3f}% | "
            f"{coarse['shell_elements']}/{fine['shell_elements']} | "
            f"{coarse['wall_seconds']:.2f}/{fine['wall_seconds']:.2f} s |")
    variant_table=["| One-at-a-time variant | Largest frequency change from baseline |",
                   "| --- | ---: |"]
    for name,row in hypotheses["variants"].items():
        variant_table.append(
            f"| {name} | {max(row['change_from_gmsh_baseline_percent']):.3f}% |")
    return "\n".join([
        "# Mesh and hub investigation", "",
        "Generated by `python3 -m tools.convergence_study`. All frequencies",
        "come from native CalculiX; JSON and SVG evidence is in",
        "`results/convergence/`. Material, mass, and clamp inputs remain provisional.", "",
        "## Baseline", "", *baseline_rows, "",
        "| Mode | Observed order | Richardson limit (Hz) | Status |",
        "| ---: | ---: | ---: | --- |", *mode_lines, "",
        "Nonmonotonic frequencies do not have a defensible Richardson limit.",
        "The plotted values are solver outputs at each mesh target size.", "",
        "## Corner cases", "", *corner_table, "",
        f"Maximum target-mode change between the two corner meshes: **{maximum:.3f}%**.",
        f"Cases exceeding 2%: **{', '.join(failing) if failing else 'none'}**.",
        "A finer or adaptive mesh is recommended in any exceeding region.",
        "The per-case element counts and wall times quantify the cost in",
        "`results/convergence/corner_cases.json`.", "",
        "## One-at-a-time hypotheses", "", *variant_table, "",
        "The Gmsh versus structured-mesh comparison and structured aspect-ratio",
        "perturbation are recorded in `mesh_hypotheses.json`. A separate",
        "hub-stiffness perturbation holds its density fixed. These isolate",
        "discretization and hub-compliance effects but do not establish physical",
        "joint stiffness. A zero difference between Gmsh and the structured",
        "default on this geometry indicates matching transfinite meshes; it",
        "does not show general mesher independence.", "",
        "An experimental explicit two-plate and standoff deck failed the",
        "six-rigid-mode validity check; it is not included as a quantitative",
        "comparison or production model. The equivalent-shell idealization",
        "therefore remains an unresolved model-form uncertainty.", "",
        "Surrogate held-out error and FE mesh change are distinct. Neither is",
        "an estimate of real-world prediction error.", "",
    ])+"\n"


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ccx-path",default="bin/ccx_2.23")
    parser.add_argument("--work",type=Path,default=ROOT/"build/convergence_study")
    parser.add_argument("--out",type=Path,default=ROOT/"results/convergence")
    parser.add_argument("--report-only",action="store_true",
                        help="regenerate Markdown from existing numerical JSON")
    args=parser.parse_args()
    if args.report_only:
        baseline=json.loads((args.out/"baseline_three_mesh.json").read_text())
        corners=json.loads((args.out/"corner_cases.json").read_text())
        hypotheses=json.loads((args.out/"mesh_hypotheses.json").read_text())
        (ROOT/"docs/convergence_investigation.md").write_text(
            report_text(baseline,corners,hypotheses))
        return
    config=load_design(ROOT/"config/reference_design.json")
    args.out.mkdir(parents=True,exist_ok=True)
    rows=[solve(config,args.work,f"baseline_{h:g}",h,args.ccx_path)
          for h in (.025,.0125,.00625)]
    baseline=three_mesh(rows)
    save(args.out/"baseline_three_mesh.json",baseline)
    (args.out/"baseline_three_mesh.svg").write_text(line_plot(rows))
    corners=run_corner_check(config,args.work,args.ccx_path)
    save(args.out/"corner_cases.json",corners)
    reference=rows[0]["frequencies_hz"]
    variants={
        "structured_default":solve(config,args.work,"structured_default",.025,args.ccx_path,
                                   mesh_generator="structured"),
        "structured_wide":solve(config,args.work,"structured_wide",.025,args.ccx_path,
                                mesh_generator="structured",width_division_scale=2.0),
        "structured_narrow":solve(config,args.work,"structured_narrow",.025,args.ccx_path,
                                  mesh_generator="structured",width_division_scale=0.5),
        "hub_stiff_x10":solve(config,args.work,"hub_stiff_x10",.025,args.ccx_path,
                              hub_stiffness_multiplier=10.0),
    }
    for row in variants.values():
        row["change_from_gmsh_baseline_percent"]=difference_percent(
            row["frequencies_hz"],reference)
    hypotheses={"source":"calculix_ccx_native","gmsh_baseline":rows[0],
                "variants":variants,"explicit_plate_status":
                "experimental separated-plate deck failed six-rigid-mode validation; excluded"}
    save(args.out/"mesh_hypotheses.json",hypotheses)
    (ROOT/"docs/convergence_investigation.md").write_text(
        report_text(baseline,corners,hypotheses))
    print(json.dumps({"baseline_orders":baseline["observed_order"],
                      "corner_cases":len(corners["cases"]),
                      "max_corner_change_percent":max(r["max_target_change_percent"]
                                                       for r in corners["cases"])},indent=2))


if __name__=="__main__":
    main()
