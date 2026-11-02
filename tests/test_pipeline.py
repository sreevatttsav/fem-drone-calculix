"""Focused tests for sampling, mode matching, inference bounds, and RPM screen."""

import sys
import json
import tempfile
import types
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sweep import generate_dataset, latin_hypercube, match_modes, solve_design
from surrogate import train, predict
from campbell import check_design, render_svg


class PipelineTests(unittest.TestCase):
    def test_latin_hypercube_uses_each_stratum_once(self):
        ranges = {"arm_length_m": [0.1, 0.2], "motor_mass_kg": [0.05, 0.1]}
        rows = latin_hypercube(ranges, 20, 7)
        self.assertEqual(rows, latin_hypercube(ranges, 20, 7))
        for key, (lo, hi) in ranges.items():
            bins = [min(19, int((row[key] - lo) / (hi - lo) * 20)) for row in rows]
            self.assertEqual(sorted(bins), list(range(20)))

    def test_mode_matching_uses_shape_not_frequency_rank(self):
        reference = np.eye(3)
        candidate = np.array([reference[1], -reference[0], reference[2]])
        selected, quality, method = match_modes(reference, candidate, [10, 20, 30])
        self.assertEqual(selected, [20, 10, 30])
        self.assertEqual(quality, [1, 1, 1])
        self.assertEqual(method, "tip_displacement_MAC")

    def test_degenerate_pair_uses_subspace_overlap(self):
        reference = np.eye(4)
        c = 2 ** -0.5
        candidate = np.array([reference[0], reference[1],
                              c * (reference[2] + reference[3]),
                              c * (reference[2] - reference[3])])
        selected, quality, method = match_modes(reference, candidate,
                                                [10, 20, 30, 30.15], 3,
                                                [10, 20, 30, 30])
        self.assertEqual(selected[:2], [10, 20])
        self.assertAlmostEqual(selected[2], 30.075)
        self.assertAlmostEqual(quality[2], 1)
        self.assertIn("subspace", method)

    def test_train_predict_and_check(self):
        ranges = {"arm_length_m": [0.1, 0.3], "motor_mass_kg": [0.04, 0.12]}
        rows = latin_hypercube(ranges, 40, 17)
        dataset = {"metadata": {"source": "local_linear_beam_FEM", "ranges": ranges}, "records": []}
        for i, design in enumerate(rows):
            length, mass = design["arm_length_m"], design["motor_mass_kg"]
            target = [100 + 50 * length - 30 * mass, 200 + 100 * length - 20 * mass]
            dataset["records"].append({"design_id": f"D{i:04d}", "inputs": design,
                                       "solver": "local_linear_beam_FEM",
                                       "frequencies_hz": target, "mode_mac": [1, 1]})
        model = train(dataset)
        estimate = predict(model, {"arm_length_m": 0.2, "motor_mass_kg": 0.08})
        self.assertEqual(len(estimate), 2)
        self.assertTrue(all(value > 0 for value in estimate))
        with self.assertRaisesRegex(ValueError, "out of trained range"):
            predict(model, {"arm_length_m": 0.4, "motor_mass_kg": 0.08})
        report = check_design(model, {"arm_length_m": 0.2, "motor_mass_kg": 0.08,
                                      "rpm_min": 3000, "rpm_max": 9000, "blades": 2})
        self.assertEqual(len(report["checks"]), 4)
        self.assertIn("<svg", render_svg(report))
        dataset["metadata"]["source"] = "calculix_ccx_native"
        for row in dataset["records"]:
            row["solver"] = "calculix_ccx_native"
        calc_model = train(dataset)
        self.assertEqual(calc_model["source"], "calculix_ccx_native")
        self.assertIn("CalculiX concept", render_svg(check_design(calc_model, {
            "arm_length_m": 0.2, "motor_mass_kg": 0.08,
            "rpm_min": 3000, "rpm_max": 9000, "blades": 2})))
        dataset["records"][0]["solver"] = "local_linear_beam_FEM"
        with self.assertRaisesRegex(ValueError, "mixed"):
            train(dataset)

    def test_campbell_legend_labels_have_distinct_positions(self):
        report = {"rpm_range": [3000, 9000], "blades": 2,
                  "predicted_frequencies_hz": [27.8, 35.5, 142.1],
                  "held_out_p95_error_percent_by_mode": [0.4, 0.6, 0.4],
                  "passes": False}
        root = ET.fromstring(render_svg(report))
        labels = [element for element in root.iter("{http://www.w3.org/2000/svg}text")
                  if element.text and (element.text.startswith("Mode ")
                                       or element.text in ("1P", "blade pass"))]
        self.assertEqual(len(labels), 5)
        positions = [(float(label.attrib["x"]), float(label.attrib["y"])) for label in labels]
        self.assertTrue(all(x >= 680 for x, _ in positions))
        self.assertEqual(len({y for _, y in positions}), len(positions))
        self.assertTrue(all(b - a >= 24 for a, b in zip(sorted(y for _, y in positions),
                                                        sorted(y for _, y in positions)[1:])))

    def test_calculix_sweep_provenance_failures_and_tracking(self):
        baseline = json.loads((Path(__file__).resolve().parents[1] /
                               "config/reference_design.json").read_text())
        calls = []
        shapes = np.eye(6, 12).tolist()

        def fake_solve(design, output_dir, mesh_size_m, flexible_count, ccx_path):
            calls.append(Path(output_dir).name)
            if len(calls) == 2:
                raise ValueError("simulated mesh failure")
            return {"provenance": "calculix_ccx_native",
                    "flexible_frequencies_hz": [10, 20, 30, 30, 40, 50],
                    "mode_shapes_tip_xyz": shapes,
                    "rigid_frequencies_hz": [0] * 6,
                    "mode_descriptors": ["test"] * 6,
                    "mesh": {"mesh_size_m": mesh_size_m},
                    "solver_version": "test", "source_files": {"dat": "case.dat"}}

        fake_module = types.SimpleNamespace(solve_calculix=fake_solve)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"calculix_runner": fake_module}):
            payload = generate_dataset(baseline, count=10, seed=42,
                                       output_dir=directory, backend="calculix")
            self.assertEqual(payload["metadata"]["source"], "calculix_ccx_native")
            self.assertEqual(payload["metadata"]["successful_designs"], 10)
            self.assertEqual(payload["metadata"]["failed_designs"], 1)
            self.assertEqual(len(json.loads((Path(directory) / "failures.json").read_text())), 1)
            self.assertTrue(all(row["solver"] == "calculix_ccx_native"
                                for row in payload["records"]))
            self.assertTrue(all(row["mode_mac"] == [1, 1, 1]
                                for row in payload["records"]))
            self.assertEqual(payload["records"][1]["mode_tracking"],
                             "tip_MAC_first_two_plus_degenerate_subspace_MAC")
            self.assertEqual(calls[0], "D0000")
            with self.assertRaisesRegex(ValueError, "use a separate directory"):
                generate_dataset(baseline, count=10, seed=42,
                                 output_dir=directory, backend="beam")

    def test_calculix_never_accepts_wrong_solver_provenance(self):
        fake_module = types.SimpleNamespace(solve_calculix=lambda *args, **kwargs:
                                            {"provenance": "local_linear_beam_FEM"})
        with patch.dict(sys.modules, {"calculix_runner": fake_module}):
            with self.assertRaisesRegex(ValueError, "incorrect provenance"):
                solve_design({}, "calculix", Path("/tmp/fake_calculix_case"))

    def test_missing_ccx_fails_before_pipeline_writes_outputs(self):
        from run_pipeline import run

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with self.assertRaisesRegex(FileNotFoundError, "CalculiX executable"):
                run(Path("unused.json"), output, 10, 42, 4,
                    backend="calculix", ccx_path="ccx_definitely_missing_for_test")
            self.assertFalse(output.exists())

    def test_ten_case_beam_pilot_skips_training_cleanly(self):
        from run_pipeline import run

        config = Path(__file__).resolve().parents[1] / "config/reference_design.json"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            manifest = run(config, output, 10, 42, 4, backend="beam")
            self.assertEqual(manifest["simulation_source"], "local_linear_beam_FEM")
            self.assertEqual(manifest["pipeline_stage"], "pilot_sweep_only")
            self.assertIsNone(manifest["held_out_metrics"])
            self.assertFalse((output / "surrogate_model.json").exists())
            self.assertTrue((output / "data/dataset.json").exists())


if __name__ == "__main__":
    unittest.main()
