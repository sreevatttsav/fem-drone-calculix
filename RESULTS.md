# Illustrative baseline and 500-design run

These results come from the included **local 3D beam FEM**, using concept
dimensions and estimated equivalent carbon-laminate properties. They are not
CalculiX results, measured vibration data, or a flight-safety assessment.

| Check | Result |
|---|---:|
| Assembled screening-model mass | 1.08193 kg |
| Free-free rigid modes | Six at 0 Hz |
| First three flexible targets | 27.85, 35.49, 142.11 Hz |
| First-five mesh change, 4 to 8 elements per arm | 0.00438% maximum |
| Successful unique designs | 500 |
| Excluded proposals with ambiguous mode matches | 66 |
| Train / held-out test designs | 400 / 100 |
| Selected model | NumPy quadratic ridge |
| Held-out median percentage error, modes 1–3 | 0.123%, 0.113%, 0.112% |
| Held-out 95th-percentile percentage error, modes 1–3 | 0.443%, 0.594%, 0.390% |
| Best of ANN, GPR, XGBoost by training-only cross-validation | ANN, one 32-unit hidden layer |
| That ANN's held-out median percentage error, modes 1–3 | 2.023%, 1.953%, 1.323% |

The third target is the shared frequency of a symmetric third/fourth-mode
doublet, tracked by subspace overlap. Its individual eigenvectors are not
treated as a stable mode identity.

For the illustrative 3,000–9,000 RPM range and two-blade rotor, the baseline
**fails** the 15% plus model-error frequency-separation screen. The 142.11 Hz
target intersects blade-pass excitation near 4,263 RPM and 1P near 8,527 RPM.
This flags frequencies for review; it does not predict the resulting vibration
amplitude. The low held-out errors measure how well the surrogate copies its
own beam FEM across the sampled designs, not accuracy against a physical frame.
The NumPy quadratic model outperformed the three named-model families on this
beam-generated dataset. The detailed comparison and R² values are in
`build/full/ml_compare/comparison.json`; all models use the same 100-design
held-out set.

Next evidence gates: replace concept inputs with measured/supplier values,
perform and review the CalculiX shell/joint model, and complete the
suspended-frame impact test. Rebuild the dataset from validated detailed-FEM
cases before using the surrogate for design decisions.
