"""Meaningful dimensional, mesh-integrity and mechanics trend checks."""

import math
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geometry import arm_polygon, layout, load_design, part_meshes, write_svg
from mechanics import beam_estimate, sensitivity


class GeometryMechanicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design = load_design()

    def test_four_symmetric_motors_at_specified_span(self):
        loc = layout(self.design)
        points = [arm["motor_xy_m"] for arm in loc["arms"]]
        self.assertEqual(len(points), 4)
        for x, y in points:
            self.assertAlmostEqual(math.hypot(x, y) - loc["hub_edge_radius_m"],
                                   self.design["arm_length_m"])
            self.assertTrue(any(math.isclose(-x, a) and math.isclose(-y, b)
                                for a, b in points))

    def test_arm_width_and_component_meshes(self):
        loc = layout(self.design)
        poly = arm_polygon(self.design, loc["arms"][0])
        self.assertAlmostEqual(math.dist(poly[0], poly[3]), self.design["arm_width_m"])
        meshes = part_meshes(self.design)
        self.assertEqual(len(meshes), 10)  # 4 arms, 4 mounts, 2 plates
        self.assertTrue(all(len(triangles) >= 12 for triangles in meshes.values()))

    def test_svg_background_covers_negative_origin_viewbox(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preview.svg"
            write_svg(path, self.design)
            root = ET.parse(path).getroot()
            x, y, width, height = map(float, root.attrib["viewBox"].split())
            background = root.find("{http://www.w3.org/2000/svg}rect")
            self.assertIsNotNone(background)
            self.assertAlmostEqual(float(background.attrib["x"]), x)
            self.assertAlmostEqual(float(background.attrib["y"]), y)
            self.assertAlmostEqual(float(background.attrib["width"]), width)
            self.assertAlmostEqual(float(background.attrib["height"]), height)

    def test_bending_trends_and_sensitivities(self):
        base = beam_estimate(self.design)
        g = self.design["geometry"]
        expected_mount = (self.design["rho_kg_m3"] * math.pi
                          * (g["motor_mount_diameter_m"]/2)**2
                          * g["motor_mount_thickness_m"])
        self.assertAlmostEqual(base["mount_mass_kg"], expected_mount)
        self.assertAlmostEqual(base["effective_mass_kg"],
                               self.design["motor_mass_kg"] + expected_mount
                               + 0.236*base["arm_mass_kg"])
        self.assertGreater(base["modes"]["in_plane"]["frequency_Hz"],
                           base["modes"]["out_of_plane"]["frequency_Hz"])
        longer = dict(self.design, arm_length_m=1.1*self.design["arm_length_m"])
        thicker = dict(self.design, arm_thickness_m=1.1*self.design["arm_thickness_m"])
        heavier = dict(self.design, motor_mass_kg=1.1*self.design["motor_mass_kg"])
        f = lambda d: beam_estimate(d)["modes"]["out_of_plane"]["frequency_Hz"]
        self.assertLess(f(longer), f(self.design))
        self.assertGreater(f(thicker), f(self.design))
        self.assertLess(f(heavier), f(self.design))
        self.assertAlmostEqual(sensitivity(self.design)["out_of_plane"]["E_long_Pa"], 0.5, places=5)


if __name__ == "__main__":
    unittest.main()
