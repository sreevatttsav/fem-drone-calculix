"""Parametric X-quad concept geometry (SI inputs, millimetre CAD exports).

The STL files are independent solids for each arm, plate and motor mount; they
are not fused at the overlaps. The emitted FreeCAD macro exports the same
solids as STEP when run in FreeCAD.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "reference_design.json"


def load_design(path: str | Path = DEFAULT_CONFIG) -> dict:
    design = json.loads(Path(path).read_text())
    positive = ("arm_length_m", "arm_width_m", "arm_thickness_m", "hub_width_m",
                "hub_plate_thickness_m", "motor_mass_kg", "payload_mass_kg",
                "E_long_Pa", "E_trans_Pa", "E_through_Pa", "G_Pa",
                "G13_Pa", "G23_Pa", "rho_kg_m3")
    for name in positive:
        if float(design[name]) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("nu12", "nu13", "nu23"):
        if not 0 <= float(design[name]) < 0.5:
            raise ValueError(f"{name} must be in [0, 0.5)")
    if design.get("units") != "SI":
        raise ValueError("Input units must be SI")
    if design["rpm_min"] < 0 or design["rpm_max"] <= design["rpm_min"]:
        raise ValueError("Invalid RPM range")
    if int(design["blades"]) < 1:
        raise ValueError("blades must be at least one")
    g = design["geometry"]
    if not 0 < g["arm_clamp_overlap_m"] < design["hub_width_m"] / math.sqrt(2):
        raise ValueError("Arm clamp overlap must lie inside the hub")
    if g["hub_plate_gap_m"] < design["arm_thickness_m"]:
        raise ValueError("Hub plate gap must accommodate the arm")
    for name in ("motor_mount_diameter_m", "motor_mount_thickness_m"):
        if g[name] <= 0:
            raise ValueError(f"{name} must be positive")
    return design


def layout(design: dict) -> dict:
    """Return physical positions in metres. Arm length starts at hub edge."""
    half = design["hub_width_m"] / 2
    hub_edge_radius = design["hub_width_m"] / math.sqrt(2)
    motor_radius = hub_edge_radius + design["arm_length_m"]
    root_radius = hub_edge_radius - design["geometry"]["arm_clamp_overlap_m"]
    arms = []
    for index, degrees in enumerate((45, 135, 225, 315), start=1):
        a = math.radians(degrees)
        u = (math.cos(a), math.sin(a))
        v = (-u[1], u[0])
        root = tuple(root_radius * c for c in u)
        motor = tuple(motor_radius * c for c in u)
        arms.append({"name": f"arm_{index}", "angle_deg": degrees,
                     "direction": u, "perpendicular": v, "root_xy_m": root,
                     "motor_xy_m": motor})
    return {"hub_half_width_m": half, "hub_edge_radius_m": hub_edge_radius,
            "motor_radius_m": motor_radius, "arms": arms}


def arm_polygon(design: dict, arm: dict) -> list[tuple[float, float]]:
    x0, y0 = arm["root_xy_m"]
    x1, y1 = arm["motor_xy_m"]
    vx, vy = arm["perpendicular"]
    w = design["arm_width_m"] / 2
    return [(x0 - w*vx, y0 - w*vy), (x1 - w*vx, y1 - w*vy),
            (x1 + w*vx, y1 + w*vy), (x0 + w*vx, y0 + w*vy)]


def _prism_triangles(polygon: list[tuple[float, float]], z0: float, z1: float):
    n = len(polygon)
    low = [(x, y, z0) for x, y in polygon]
    high = [(x, y, z1) for x, y in polygon]
    triangles = []
    for i in range(1, n - 1):
        triangles.extend(((high[0], high[i], high[i+1]),
                          (low[0], low[i+1], low[i])))
    for i in range(n):
        j = (i + 1) % n
        triangles.extend(((low[i], low[j], high[j]),
                          (low[i], high[j], high[i])))
    return triangles


def _circle(cx: float, cy: float, radius: float, n: int = 32):
    return [(cx + radius*math.cos(2*math.pi*i/n),
             cy + radius*math.sin(2*math.pi*i/n)) for i in range(n)]


def part_meshes(design: dict) -> dict:
    """Independent watertight STL part meshes, coordinates in metres."""
    loc = layout(design)
    gap = design["geometry"]["hub_plate_gap_m"]
    tp = design["hub_plate_thickness_m"]
    ta = design["arm_thickness_m"]
    h = loc["hub_half_width_m"]
    square = [(-h, -h), (h, -h), (h, h), (-h, h)]
    meshes = {
        "hub_lower": _prism_triangles(square, -gap/2-tp, -gap/2),
        "hub_upper": _prism_triangles(square, gap/2, gap/2+tp),
    }
    for arm in loc["arms"]:
        meshes[arm["name"]] = _prism_triangles(arm_polygon(design, arm), -ta/2, ta/2)
        x, y = arm["motor_xy_m"]
        mount = _circle(x, y, design["geometry"]["motor_mount_diameter_m"]/2)
        meshes["mount_" + arm["name"]] = _prism_triangles(
            mount, ta/2, ta/2+design["geometry"]["motor_mount_thickness_m"])
    return meshes


def write_stl(path: Path, name: str, triangles) -> None:
    # STL convention is mm. Meshes are independent components, not a fused part.
    with path.open("w") as stream:
        stream.write(f"solid {name}\n")
        for tri in triangles:
            a, b, c = tri
            u = [b[i]-a[i] for i in range(3)]
            v = [c[i]-a[i] for i in range(3)]
            cross = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2],
                     u[0]*v[1]-u[1]*v[0])
            mag = math.sqrt(sum(x*x for x in cross))
            normal = tuple(x/mag for x in cross)
            stream.write("  facet normal %.8g %.8g %.8g\n    outer loop\n" % normal)
            for vertex in tri:
                stream.write("      vertex %.8g %.8g %.8g\n" % tuple(1000*x for x in vertex))
            stream.write("    endloop\n  endfacet\n")
        stream.write(f"endsolid {name}\n")


def write_svg(path: Path, design: dict) -> None:
    loc = layout(design)
    extent = (loc["motor_radius_m"] + design["geometry"]["motor_mount_diameter_m"]) * 1000
    size = 2*extent
    h = design["hub_width_m"]*500
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{-extent:.3f} {-extent:.3f} {size:.3f} {size:.3f}">',
             f'<rect x="{-extent:.3f}" y="{-extent:.3f}" width="{size:.3f}" height="{size:.3f}" fill="#f8fafc"/>',
             '<g transform="scale(1,-1)" stroke-linejoin="round">']
    for arm in loc["arms"]:
        points = " ".join(f"{1000*x:.3f},{1000*y:.3f}" for x, y in arm_polygon(design, arm))
        lines.append(f'<polygon points="{points}" fill="#334155" stroke="#0f172a" stroke-width="1"/>')
        x, y = arm["motor_xy_m"]
        r = 500*design["geometry"]["motor_mount_diameter_m"]
        lines.append(f'<circle cx="{1000*x:.3f}" cy="{1000*y:.3f}" r="{r:.3f}" fill="#2563eb" fill-opacity=".85" stroke="#1e3a8a"/>')
    lines.append(f'<rect x="{-h:.3f}" y="{-h:.3f}" width="{2*h:.3f}" height="{2*h:.3f}" fill="#94a3b8" fill-opacity=".85" stroke="#0f172a" stroke-width="1.5"/>')
    lines.extend(('</g>', '</svg>'))
    path.write_text("\n".join(lines) + "\n")


def write_freecad_macro(path: Path, step_path: Path, design: dict) -> None:
    """Emit a FreeCAD Python macro; opening it in FreeCAD creates a STEP assembly."""
    loc = layout(design)
    gap = design["geometry"]["hub_plate_gap_m"]*1000
    tp = design["hub_plate_thickness_m"]*1000
    ta = design["arm_thickness_m"]*1000
    half = design["hub_width_m"]*500
    objects = []
    script = ["import FreeCAD as App", "import Part", "import Import", "doc = App.newDocument('XQuadConcept')", "objects = []",
              "def add_prism(name, points, z0, z1):",
              "    wire = Part.makePolygon([App.Vector(x,y,z0) for x,y in points] + [App.Vector(points[0][0],points[0][1],z0)])",
              "    solid = Part.Face(wire).extrude(App.Vector(0,0,z1-z0))",
              "    obj = doc.addObject('PartDesign::Feature', name)",
              "    obj.Shape = solid", "    objects.append(obj)"]
    square = [[-half,-half],[half,-half],[half,half],[-half,half]]
    script.extend((f"add_prism('HubLower', {square!r}, {-gap/2-tp!r}, {-gap/2!r})",
                   f"add_prism('HubUpper', {square!r}, {gap/2!r}, {gap/2+tp!r})"))
    for arm in loc["arms"]:
        poly = [[1000*x,1000*y] for x,y in arm_polygon(design,arm)]
        script.append(f"add_prism('{arm['name']}', {poly!r}, {-ta/2!r}, {ta/2!r})")
        x,y = arm["motor_xy_m"]
        r = design["geometry"]["motor_mount_diameter_m"]*500
        z = ta/2
        height = design["geometry"]["motor_mount_thickness_m"]*1000
        script.extend((f"obj = doc.addObject('PartDesign::Feature', 'Mount_{arm['name']}')",
                       f"obj.Shape = Part.makeCylinder({r!r}, {height!r}, App.Vector({1000*x!r},{1000*y!r},{z!r}))",
                       "objects.append(obj)"))
    script.extend(("doc.recompute()", f"Import.export(objects, {str(step_path)!r})",
                   "App.Console.PrintMessage('Wrote STEP assembly\\n')"))
    path.write_text("\n".join(script) + "\n")


def export(design: dict, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, triangles in part_meshes(design).items():
        write_stl(out / f"{name}.stl", name, triangles)
    write_svg(out / "plan_view.svg", design)
    write_freecad_macro(out / "export_step.FCMacro", out / "x_quad_assembly.step", design)
    manifest = {"unit_inputs": "SI", "cad_export_units": "mm", "part_count": len(part_meshes(design)),
                "parts": sorted(part_meshes(design)), "layout": layout(design),
                "step_macro": "export_step.FCMacro", "step_file_when_macro_run": "x_quad_assembly.step"}
    (out / "geometry_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "geometry")
    args = parser.parse_args()
    result = export(load_design(args.config), args.out)
    print(json.dumps({"out": str(args.out), "part_count": result["part_count"]}, indent=2))
