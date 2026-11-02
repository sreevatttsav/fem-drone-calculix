# Code map

`geometry.py` validates the SI reference design, lays out the four radial
arms, and exports independent STL parts, a plan-view SVG, a geometry manifest,
and a FreeCAD macro. Its exported solids are presentation and fabrication
concept artifacts; the modal solver uses the shell geometry generated
separately.

`calculix_mesh.py` builds a Gmsh geometry for one equivalent hub patch and
four arm patches, or uses its structured mesh generator when Gmsh is absent.
It writes the S8R input deck with effective orthotropic materials, lumped
masses, and arm-root interpolation equations.

`calculix_runner.py` invokes native CalculiX, parses frequencies from the
DAT eigenvalue table and motor-tip displacements from the FRD file, validates
the free-free rigid modes, and records solver and mesh provenance. It does
not substitute another solver on failure.

`fem.py` is the historical NumPy Euler-Bernoulli beam solver. It assembles
free-free mass and stiffness matrices and returns modal frequencies and tip
shapes with a provenance tag distinct from CalculiX.

`mechanics.py` computes simple beam estimates, a local finite-difference
sensitivity report, and mass-related engineering checks. It is an analytical
comparison, not a solver for the shell model.

`sweep.py` defines parameter ranges and seeded Latin hypercube sampling,
selects one solver backend, performs tip-shape MAC matching against the
reference, and saves accepted cases and failures separately. It currently
treats one baseline degenerate pair as a subspace.

`surrogate.py` validates a single-provenance dataset, reserves a design-level
hold-out set, selects a NumPy quadratic ridge or RBF kernel ridge fit by
training-only cross-validation, and predicts positive frequencies in log
space. Its inference API rejects inputs outside the trained ranges.

`ml_compare.py` optionally compares ANN, GPR, and XGBoost configurations
on the same labeled dataset. It reports validation and held-out metrics;
the operational pipeline still uses the NumPy surrogate.

`campbell.py` predicts frequencies with the surrogate, checks separation
from 1P and blade-pass excitation over the configured RPM interval, and
renders a standalone SVG. It currently receives only the surrogate's
tracked targets.

`validation.py` provides spectral peak extraction for accelerometer traces.
It is a first-pass analysis utility and does not claim that a peak is a
validated structural mode.

`run_pipeline.py` coordinates geometry export, mechanics report, baseline
and refined modal solves, parameter sweep, surrogate training, optional
named-model comparison, and Campbell report. It writes a run manifest
identifying simulation source and validation status.

## Data flow

`config/reference_design.json` -> design validation and geometry ->
CalculiX mesh/deck -> native DAT and FRD output -> mode tracking and
`dataset.json` -> surrogate training -> predicted design frequencies ->
Campbell separation checks and SVG. Every solve is labeled
`calculix_ccx_native`; the legacy beam path has a separate label and
output directory.
