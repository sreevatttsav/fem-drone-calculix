import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calculix_runner import parse_dat_frequencies, parse_frd_displacements, solve_calculix


class CalculixParserTests(unittest.TestCase):
    def test_dat_frequency_column_and_mode_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "modal.dat"
            path.write_text("""
 E I G E N V A L U E   O U T P U T
 MODE NO    EIGENVALUE                       FREQUENCY
                             REAL PART            IMAGINARY PART
               (RAD/TIME)      (CYCLES/TIME     (RAD/TIME)
 1   -0.1000000E-08  0.0000000E+00  0.0000000E+00  0.1000000E-03
 2    0.3947842E+05  0.6283185E+03  0.1000000E+03  0.0000000E+00
 P A R T I C I P A T I O N   F A C T O R S
 1    0.9999999E+00  0.0000000E+00  0.0000000E+00
""")
            self.assertEqual(parse_dat_frequencies(path), [0.0, 100.0])

    def test_frd_packed_negative_fields_and_normalized_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "modal.frd"
            path.write_text("""
 1PSTEP 1 1 1
 100CL 101 1.00000E+00 4 0 1 1
 -4 DISP 4 1
 -5 D1 1 2 1 0
 -5 D2 1 2 2 0
 -5 D3 1 2 3 0
 -1 10 1.00000E+00-1.00000E+00 0.00000E+00
 -1        20-1.00000E+00 1.00000E+00 0.00000E+00
 -1 30 0.00000E+00 0.00000E+00 1.00000E+00
 -1 40 0.00000E+00 0.00000E+00-1.00000E+00
 -3
""")
            shapes = parse_frd_displacements(path, [10, 20, 30, 40])
            self.assertEqual(len(shapes), 1)
            self.assertEqual(len(shapes[0]), 12)
            self.assertAlmostEqual(math.sqrt(sum(v*v for v in shapes[0])), 1)
            self.assertLess(shapes[0][1], 0)
            self.assertLess(shapes[0][3], 0)
            self.assertGreater(shapes[0][8], 0)

    def test_parser_rejects_missing_tip_and_missing_table(self):
        with tempfile.TemporaryDirectory() as directory:
            dat = Path(directory) / "bad.dat"
            dat.write_text("no frequency table\n")
            with self.assertRaises(ValueError):
                parse_dat_frequencies(dat)
            frd = Path(directory) / "bad.frd"
            frd.write_text("-4 DISP 4 1\n-1 10 1.0E+00 0.0E+00 0.0E+00\n-3\n")
            with self.assertRaisesRegex(ValueError, "missing tip nodes"):
                parse_frd_displacements(frd, [10, 20, 30, 40])

    def test_missing_ccx_fails_without_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("calculix_runner.shutil.which", return_value=None):
                with self.assertRaisesRegex(FileNotFoundError, "CalculiX executable not found"):
                    solve_calculix({}, directory)

    def test_native_runner_parses_solver_outputs_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)

            def write_deck(_design, out_dir, mesh_size_m, mode_count):
                self.assertEqual(mode_count, 12)
                inp = Path(out_dir) / "quad.inp"
                inp.write_text("*HEADING\n")
                dat = inp.with_suffix(".dat")
                rows = [" E I G E N V A L U E   O U T P U T", " MODE NO    EIGENVALUE FREQUENCY"]
                for mode in range(1, 13):
                    frequency = 0 if mode <= 6 else mode * 10
                    rows.append(f" {mode} 1.000000E+00 2.000000E+00 {frequency:.6E} 0.000000E+00")
                dat.write_text("\n".join(rows) + "\n")
                frd = inp.with_suffix(".frd")
                blocks = []
                for _mode in range(12):
                    blocks += [" -4 DISP 4 1"]
                    for node in (10, 20, 30, 40):
                        blocks.append(f" -1 {node} 1.00000E+00 0.00000E+00 1.00000E+00")
                    blocks += [" -3"]
                frd.write_text("\n".join(blocks) + "\n")
                return {"inp_path": str(inp), "tip_node_ids": [10, 20, 30, 40],
                        "mass_estimate_kg": 1.2, "mesh": {"elements": 100}}

            fake_geometry = types.SimpleNamespace(write_calculix_deck=write_deck)
            completed = types.SimpleNamespace(returncode=0, stdout="CalculiX version 2.22\n")
            with mock.patch.dict(sys.modules, {"calculix_mesh": fake_geometry}), \
                 mock.patch("calculix_runner.shutil.which", return_value="/fake/ccx"), \
                 mock.patch("calculix_runner.subprocess.run", return_value=completed):
                result = solve_calculix({}, output)
            self.assertEqual(result["provenance"], "calculix_ccx_native")
            self.assertEqual(result["solver_version"], "2.22")
            self.assertEqual(result["flexible_frequencies_hz"], [70, 80, 90, 100, 110, 120])
            self.assertEqual(result["total_mass_kg"], 1.2)
            self.assertEqual(len(result["mode_shapes_tip_xyz"]), 6)


if __name__ == "__main__":
    unittest.main()
