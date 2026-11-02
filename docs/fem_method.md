# Modal FEM methods

## Native CalculiX concept model

`calculix_mesh.write_calculix_deck` builds five separate midsurface groups:
one equivalent hub shell and four oriented arms. The equivalent hub preserves
the idealized two-plate areal mass, membrane stiffness, and bending stiffness.
Gmsh generates incomplete quadratic
quadrilaterals when available; a deterministic structured generator can make
the same `S8R` element type. The deck uses SI units, equivalent orthotropic
engineering constants, shell thicknesses, mass elements for motors/mounts and
payload, and translational interpolation equations in the clamp regions. There are
no grounded boundary conditions. The clamp stiffness is a concept assumption,
not a bolted-joint prediction.

`calculix_runner.solve_calculix` executes the deck with native `ccx`, parses
frequencies from `.dat` and modal tip displacements from ASCII `.frd`, and
checks that six rigid modes precede the flexible modes. It raises an error if
the solver is missing, fails, or produces incomplete modal output. The first
five flexible frequencies are compared between the selected and a half-sized
mesh. Larger sweeps require at most 2% change. This is a numerical convergence
check, not physical validation.

The reference material has E1=50 GPa, E2=E3=12 GPa, G12=G13=4 GPa,
G23≈4.615 GPa, ν12=ν13=0.28, ν23=0.30, and density 1600 kg/m³. These
are provisional effective laminate values, not supplier or measured data.
Arm material axis 1 follows the arm length; hub plate axis 1 follows global X.
Measured dimensions and laminate properties should replace them before
physical interpretation.

## Legacy NumPy beam screen

`fem.solve_modal` remains available through `--backend beam`. It uses four
Euler–Bernoulli arms linked to a rigid hub with tip masses. Its free-free modes
and tip samples have the same output shape as the CalculiX backend but a
separate provenance tag. The beam model uses `E_long_Pa` and `G_Pa`; it does
not resolve the laminate transverse or through-thickness behavior. Its
previous 500-case dataset must not be described as CalculiX results.

Both models sample XYZ displacements at four motor centres in azimuth order
45°, 135°, 225°, 315°. Near-degenerate mode pairs are compared as a subspace;
ambiguous matches are recorded as failed cases rather than silently relabeled.
