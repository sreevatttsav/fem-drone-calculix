# Multirotor frame modal-response project

This project generates an X-quadrotor concept frame, solves free-free modes,
trains a frequency surrogate, and screens 1P and blade-pass excitation. All
reference dimensions, masses, clamp behavior, and effective carbon-fiber
properties are provisional. No physical frame validation has been performed.

## Solvers and outputs

The default pipeline uses native CalculiX `ccx` for every training case.
`calculix_mesh.py` generates five oriented `S8R` shell surfaces, an idealized
connected clamp, and lumped motor/payload masses. The separated hub plates are
represented by one coplanar equivalent shell matched for idealized areal mass,
membrane stiffness, and bending stiffness. It meshes with native Gmsh
when available, or its deterministic structured mesher otherwise.
`calculix_runner.py` reads modal frequencies and motor-tip displacements from
the solver outputs. The legacy NumPy beam solver remains available with
`--backend beam` and keeps a distinct dataset source. Results from the two
solvers cannot be mixed in one output directory or surrogate dataset.

The completed native 500-design run is summarized in
[CALCULIX_RESULTS.md](CALCULIX_RESULTS.md), with compact artifacts in
`results/calculix_full500`. Rerunning the pipeline generates full raw solver
files in `build/`. The previous 500-design demonstration
in [RESULTS.md](RESULTS.md) used the **beam solver**. The ANSYS CSV bridge remains
as historical code but is not part of the CalculiX pipeline.

## Setup and runs

Install Python 3.10+, NumPy, and Gmsh. The included `bin/ccx_2.23` was built
for this macOS ARM machine and depends on Homebrew ARPACK and GCC runtime
libraries. Its source archives and build script are in `bin/native_ccx_sources`.
On another system, build or install CalculiX and pass its executable through
`--ccx-path`. The pipeline records solver version and mesh provenance. From
this directory:

```bash
python3 -m unittest discover -s tests -v
python3 run_pipeline.py --backend calculix --ccx-path bin/ccx_2.23 --mesh-size-m 0.025 --count 10 --out build/calculix_trial10
python3 run_pipeline.py --backend calculix --ccx-path bin/ccx_2.23 --mesh-size-m 0.025 --count 50 --out build/calculix_pilot50
python3 run_pipeline.py --backend calculix --ccx-path bin/ccx_2.23 --mesh-size-m 0.025 --count 500 --out build/calculix_full500
```

Use separate output directories for each run. The 10-case trial saves FEM
outputs, convergence, dataset, failures, and manifest; training starts at 20
successful cases. The 50/500-case runs also save a surrogate, held-out error
metrics, design check, and Campbell diagram. A larger CalculiX sweep stops
when its baseline mesh-refinement check exceeds 2% for a target frequency.
Solver failures and ambiguous mode matches are logged; they never fall back to
beam predictions.

For the legacy screening run:

```bash
python3 run_pipeline.py --backend beam --count 50 --out build/beam_pilot50
```

The optional ANN/GPR/XGBoost comparison requires `requirements-ml.txt` and
`--compare-ml` with at least 50 successful cases. The operational Campbell
screen uses the NumPy surrogate selected by `surrogate.py`.

## Engineering limits

The concept laminate input now specifies nine effective orthotropic elastic
constants plus density. The first model uses idealized rigid arm clamps and
does not resolve plies, bolt preload, slip, damping, aerodynamic forcing, or
propeller gyroscopic effects. Its masses and geometry must be replaced with
measurements and its modes compared with a suspended-frame impact test before
using predictions for a real frame. A Campbell separation screen predicts
frequency proximity, not vibration amplitude or flight safety.

See [the FEM method](docs/fem_method.md), [surrogate method](docs/surrogate_method.md),
the [mesh investigation](docs/convergence_investigation.md), and
[experimental validation protocol](docs/experimental_validation.md).
