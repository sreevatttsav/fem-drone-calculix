import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ansys_bridge import read_modal_export, write_handoff
from fem import _assemble, _validate, mesh_convergence, solve_modal


class ModalBeamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design = json.loads((ROOT / "config" / "reference_design.json").read_text())

    def test_rigid_modes_and_shape_contract(self):
        result = solve_modal(self.design, elements_per_arm=2, flexible_count=6)
        self.assertEqual(len(result["rigid_frequencies_hz"]), 6)
        self.assertTrue(all(abs(f) < 0.001 for f in result["rigid_frequencies_hz"]))
        self.assertEqual(len(result["flexible_frequencies_hz"]), 6)
        self.assertTrue(all(f > 0 for f in result["flexible_frequencies_hz"]))
        self.assertEqual(result["flexible_frequencies_hz"], sorted(result["flexible_frequencies_hz"]))
        self.assertEqual(len(result["mode_shapes_tip_xyz"]), 6)
        self.assertTrue(all(len(shape) == 12 for shape in result["mode_shapes_tip_xyz"]))
        self.assertTrue(all(abs(sum(v*v for v in shape) - 1) < 1e-9
                            for shape in result["mode_shapes_tip_xyz"]))

    def test_total_mass_matches_ten_geom_solids_plus_lumped_masses(self):
        d = self.design
        g = d["geometry"]
        rho = d["rho_kg_m3"]
        plates = 2 * rho * d["hub_width_m"]**2 * d["hub_plate_thickness_m"]
        arms = 4 * rho * d["arm_width_m"] * d["arm_thickness_m"] * (
            d["arm_length_m"] + g["arm_clamp_overlap_m"])
        mounts = 4 * rho * math.pi * (g["motor_mount_diameter_m"]/2)**2 * g["motor_mount_thickness_m"]
        expected = plates + arms + mounts + 4*d["motor_mass_kg"] + d["payload_mass_kg"]
        self.assertAlmostEqual(solve_modal(d, 1, 1)["total_mass_kg"], expected, places=12)

    def test_clamped_overlap_is_rigid_hub_mass_not_flexible_span(self):
        d = self.design
        no_overlap = dict(d, geometry=dict(d["geometry"], arm_clamp_overlap_m=0))
        k_with, m_with, _, _ = _assemble(_validate(d, 2, 1), 2)
        k_without, m_without, _, _ = _assemble(_validate(no_overlap, 2, 1), 2)
        self.assertTrue(__import__("numpy").allclose(k_with, k_without))
        self.assertGreater(m_with[0, 0], m_without[0, 0])
        self.assertGreater(m_with[3, 3], m_without[3, 3])

    def test_longer_arms_reduce_first_flexible_frequency(self):
        baseline = solve_modal(self.design, 2, 1)["flexible_frequencies_hz"][0]
        longer = dict(self.design, arm_length_m=self.design["arm_length_m"] * 1.2)
        changed = solve_modal(longer, 2, 1)["flexible_frequencies_hz"][0]
        self.assertLess(changed, baseline)

    def test_mesh_convergence_and_input_validation(self):
        check = mesh_convergence(self.design, 2, 4, 5)
        self.assertEqual(len(check["relative_difference_percent"]), 5)
        self.assertTrue(all(math.isfinite(v) for v in check["relative_difference_percent"]))
        with self.assertRaises(ValueError):
            mesh_convergence(self.design, 4, 2)
        with self.assertRaises(ValueError):
            solve_modal(dict(self.design, arm_width_m=0), 2, 1)

    def test_ansys_handoff_and_import_are_explicitly_unverified(self):
        with tempfile.TemporaryDirectory() as directory:
            spec_path = write_handoff(self.design, Path(directory) / "handoff.json")
            spec = json.loads(spec_path.read_text())
            self.assertIn("not_solved", spec["provenance"])
            self.assertEqual(spec["geometry"]["hub_plate_gap_m"],
                             self.design["geometry"]["hub_plate_gap_m"])
            csv_path = Path(directory) / "modal.csv"
            with csv_path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["mode", "frequency_hz"])
                for index in range(1, 13):
                    writer.writerow([index, 0 if index <= 6 else index * 10])
            imported = read_modal_export(csv_path)
            self.assertEqual(imported["flexible_frequencies_hz"][0], 70)
            self.assertIn("unverified", imported["provenance"])


if __name__ == "__main__":
    unittest.main()
