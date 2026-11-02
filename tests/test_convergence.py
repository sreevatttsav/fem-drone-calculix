"""Numerical-study checks independent of the installed solver."""

import unittest

from geometry import load_design
from sweep import default_ranges
from tools.convergence_study import corner_designs, three_mesh


class ConvergenceStudyTests(unittest.TestCase):
    def test_richardson_quadratic_sequence(self):
        rows = [
            {"mesh_size_m": 0.04, "frequencies_hz": [116.0, 210.0]},
            {"mesh_size_m": 0.02, "frequencies_hz": [104.0, 202.5]},
            {"mesh_size_m": 0.01, "frequencies_hz": [101.0, 200.625]},
        ]
        result = three_mesh(rows)
        self.assertAlmostEqual(result["observed_order"][0], 2.0)
        self.assertAlmostEqual(result["observed_order"][1], 2.0)
        self.assertAlmostEqual(result["richardson_limit_hz"][0], 100.0)
        self.assertAlmostEqual(result["richardson_limit_hz"][1], 200.0)

    def test_nonmonotonic_sequence_has_no_extrapolation(self):
        rows = [
            {"mesh_size_m": 0.04, "frequencies_hz": [100.0]},
            {"mesh_size_m": 0.02, "frequencies_hz": [101.0]},
            {"mesh_size_m": 0.01, "frequencies_hz": [100.5]},
        ]
        result = three_mesh(rows)
        self.assertIsNone(result["observed_order"][0])
        self.assertIsNone(result["richardson_limit_hz"][0])

    def test_corner_designs_stay_in_sampled_ranges(self):
        baseline = load_design()
        ranges = default_ranges(baseline)
        designs = corner_designs(baseline, ranges)
        self.assertEqual(len(designs), 8)
        for design in designs.values():
            for key, (low, high) in ranges.items():
                self.assertGreaterEqual(design[key], low)
                self.assertLessEqual(design[key], high)
            self.assertEqual(design["geometry"]["hub_plate_gap_m"],
                             design["arm_thickness_m"])


if __name__ == "__main__":
    unittest.main()
