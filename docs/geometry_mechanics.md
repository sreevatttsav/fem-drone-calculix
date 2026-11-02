# Reference geometry and beam check

`config/reference_design.json` is a **concept**, with SI units throughout. It
describes a symmetric X quadrotor: four straight flat arms at 45°, 135°, 225°
and 315°, a pair of square central clamp plates, and four circular motor-mount
plates. Arm length is measured from the outer hub corner along an arm to its
motor centre. The clamp overlap extends inward from that corner. Geometry
exports convert metres to millimetres, as recorded in `geometry_manifest.json`.

The concept values (190 mm arm span, 25 × 4 mm arm section, 120 mm hub,
80 g motor, 500 g hub payload) are illustrative. The elastic constants and
density are **uncertified illustrative equivalent laminate properties**, not
supplier data. Replace them with the actual laminate layup's measured or
supplier-provided effective properties and weigh the built components. The
motor and payload masses are analytical inputs; the drawn mounts and plates are
physical geometry, but the scripts do not infer their actual mass from the
illustrative density. The FEM modeller must ensure each mass is counted once.

## Physical build bill for the baseline test

| Item | Qty | Concept dimension or placement | Mass to record on build |
|---|---:|---|---|
| Flat carbon arms | 4 | 190 mm from hub corner to motor centre; 25 mm wide × 4 mm thick; 25 mm overlap under hub plates | Weigh each arm after cutting |
| Square carbon hub plates | 2 | 120 × 120 × 2 mm; plate inner faces at ±2 mm, contacting the 4 mm arms | Weigh each plate |
| Circular motor mounts | 4 | 40 mm diameter × 4 mm thick, centred 274.85 mm radially from hub centre | Weigh each mount |
| Motors | 4 | Centre on the four 45° diagonals; 80 g each is the **concept target** | Weigh each motor and its mount hardware separately |
| Hub payload | 1 | Centred at hub; 500 g is the **concept target** | Weigh battery, electronics and their retention hardware |
| Arm clamp hardware | 4 sets | Two fastener locations per arm along its centreline, 8 mm and 18 mm inward from hub corner; holes and fasteners are **not yet included** in CAD | Weigh actual bolts, nuts, spacers and washers |

Before manufacturing, detail the hole diameter and edge distance to suit the
selected fastener and laminate, then add those holes to CAD and FEM. The
fastener locations above are layout guidance, not a strength-checked joint.
Record assembled total mass and each component centre of mass. Inspect plate
contact and joint tightness before the modal test; a nominal plate gap equal
to arm thickness does not by itself define the real joint stiffness.

Run from the project root:

```sh
python3 geometry.py --config config/reference_design.json --out build/geometry
python3 mechanics.py --config config/reference_design.json --out build/mechanics.json
python3 -m unittest discover -s tests -p 'test_geometry_mechanics.py'
```

The geometry command creates ten separate watertight ASCII STL surface meshes,
a plan-view SVG, a geometry manifest and `export_step.FCMacro`. The independent
STL solids overlap at intended assembly joints; they are mesh interchange
artifacts, not a single fused manufacturable solid. Open the macro in FreeCAD
to create editable solid features and export `x_quad_assembly.step` in mm.
FreeCAD is not required to generate the STL and SVG outputs. The automated
CalculiX path builds shell midsurfaces directly from the same design inputs;
the STEP assembly remains useful for later hardware and joint refinement.

The hand estimate uses a fixed-root uniform cantilever with point masses for
the motor and carbon mount disk:

`f = (1 / 2π) sqrt[3 E I / (L³ (m_motor + m_mount + 0.236 m_arm))]`.

The out-of-plane second moment is `w t³ / 12`; the in-plane one is `t w³ / 12`.
Both use the effective axial modulus along the arm (`E_long_Pa`) because the
dominant beam bending stress acts along the arm. `E_trans_Pa` and `G_Pa` are
provided for the later FEM orthotropic material and do not enter this simple
calculation. The payload at the hub, root clamp compliance, joint slip, motor
and mount rotary inertia, full-frame mode coupling and damping are omitted.
The root is idealized at the hub edge; the 25 mm of arm under the hub plate is
assumed clamped with no free bending there. Hence this estimate
checks direction and rough scale of FEM trends rather than setting an exact
acceptance target. The reported sensitivities are dimensionless central
finite-difference derivatives of log frequency versus log input.
