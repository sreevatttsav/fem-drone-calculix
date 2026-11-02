"""Deck-generation checks that do not require Gmsh or CalculiX."""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from calculix_mesh import write_calculix_deck
from geometry import load_design


class CalculixMeshTests(unittest.TestCase):
    def test_fallback_writes_connected_free_free_s8r_deck(self):
        design=load_design()
        with tempfile.TemporaryDirectory() as directory, patch('calculix_mesh.shutil.which',return_value=None):
            result=write_calculix_deck(design,directory,mesh_size_m=0.025)
            deck=Path(result['inp_path']).read_text()
            manifest=json.loads((Path(directory)/'calculix_manifest.json').read_text())
            self.assertEqual(result['mesh']['generator'],'deterministic_structured_fallback')
            self.assertEqual(result['mesh']['element_type'],'S8R')
            self.assertEqual(len(result['tip_node_ids']),4)
            self.assertEqual(len(set(result['tip_node_ids'])),4)
            self.assertGreater(result['mesh']['equation_count'],0)
            self.assertGreater(result['mesh']['clamped_nodes'],0)
            self.assertIn('*EQUATION',deck)
            self.assertNotIn('*TIE',deck)
            self.assertNotIn('*MPC',deck)
            self.assertEqual(deck.count('*ELEMENT, TYPE=S8R'),5)
            self.assertEqual(result['mesh']['hub_model'], 'coplanar_equivalent_two_plate_shell')
            self.assertIn('*ELASTIC, TYPE=ENGINEERING CONSTANTS',deck)
            self.assertIn('*NODE FILE, OUTPUT=2D\nU\n',deck)
            self.assertIn('*FREQUENCY\n18\n',deck)
            self.assertNotIn('*BOUNDARY',deck)
            self.assertEqual(manifest['tip_node_ids'],result['tip_node_ids'])
            self.assertTrue((Path(directory)/'x_quad_shells.geo').is_file())
            for location in result['tip_locations_m']:
                self.assertAlmostEqual(math.hypot(*location[:2]),
                                       design['hub_width_m']/math.sqrt(2)+design['arm_length_m'])

    def test_mass_accounting_and_invalid_controls(self):
        design=load_design()
        g=design['geometry']; rho=design['rho_kg_m3']
        expected=(2*rho*design['hub_width_m']**2*design['hub_plate_thickness_m']
                  +4*rho*design['arm_width_m']*design['arm_thickness_m']
                  *(design['arm_length_m']+g['arm_clamp_overlap_m'])
                  +4*design['motor_mass_kg']
                  +4*rho*math.pi*(g['motor_mount_diameter_m']/2)**2*g['motor_mount_thickness_m']
                  +design['payload_mass_kg'])
        with tempfile.TemporaryDirectory() as directory, patch('calculix_mesh.shutil.which',return_value=None):
            result=write_calculix_deck(design,directory,mesh_size_m=0.03)
            self.assertAlmostEqual(result['mass_estimate_kg'],expected)
            with self.assertRaises(ValueError):
                write_calculix_deck(design,directory,mesh_size_m=0)
            with self.assertRaises(ValueError):
                write_calculix_deck(design,directory,mode_count=6)


if __name__=='__main__':
    unittest.main()
