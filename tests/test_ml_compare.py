"""Optional named-model comparison tests; run with work/ml_venv/bin/python."""

import sys
import tempfile
import unittest
import importlib.util
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sweep import latin_hypercube

OPTIONAL_ML = all(importlib.util.find_spec(name) is not None
                  for name in ("joblib", "sklearn", "xgboost"))
if OPTIONAL_ML:
    import joblib
    from ml_compare import compare, predict


@unittest.skipUnless(OPTIONAL_ML, "requires requirements-ml.txt")
class MLComparisonTests(unittest.TestCase):
    def test_split_model_selection_and_range_check(self):
        ranges = {"arm_length_m": [0.1, 0.3], "motor_mass_kg": [0.04, 0.12]}
        designs = latin_hypercube(ranges, 50, 17)
        rows = []
        for i, design in enumerate(designs):
            length, mass = design["arm_length_m"], design["motor_mass_kg"]
            rows.append({"design_id": f"D{i:04d}", "inputs": design,
                         "solver": "local_linear_beam_FEM",
                         "frequencies_hz": [100 + 50 * length - 30 * mass,
                                            200 + 80 * length - 40 * mass,
                                            300 + 90 * length - 20 * mass],
                         "mode_mac": [1, 1, 1]})
        dataset = {"metadata": {"source": "local_linear_beam_FEM", "ranges": ranges,
                                "target_description": "synthetic test branches"}, "records": rows}
        report, model = compare(dataset, seed=42)
        expected = np.random.default_rng(42).permutation(50)[:10]
        self.assertEqual(report["held_out_design_ids"], [rows[i]["design_id"] for i in expected])
        self.assertEqual(report["selected_family"],
                         min(report["comparisons"], key=lambda row: row["cv_median_ape_percent"])["family"])
        self.assertEqual(len(report["comparisons"]), 6)
        for result in report["comparisons"]:
            self.assertEqual(len(result["held_out"]["median_ape_percent_by_mode"]), 3)
            self.assertEqual(len(result["held_out"]["p95_ape_percent_by_mode"]), 3)
            self.assertEqual(len(result["held_out"]["r2_by_mode"]), 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            joblib.dump(model, path)
            loaded = joblib.load(path)
            result = predict(loaded, designs[0])
            self.assertEqual(len(result), 3)
            self.assertTrue(all(value > 0 for value in result))
            with self.assertRaisesRegex(ValueError, "out of trained range"):
                predict(loaded, {"arm_length_m": 0.4, "motor_mass_kg": 0.08})


if __name__ == "__main__":
    unittest.main()
