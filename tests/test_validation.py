import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from validation import impact_spectrum, compare_modal_peaks


class ImpactValidationTests(unittest.TestCase):
    def test_synthetic_tap_resonance(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tap.csv"
            t = np.arange(4096) / 2048.0
            y = np.exp(-2*t) * np.sin(2*np.pi*123*t)
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["time_s", "acceleration_m_s2"])
                writer.writerows(zip(t, y))
            result = impact_spectrum(path, min_hz=50, max_hz=300)
            self.assertAlmostEqual(result["peaks"][0]["frequency_hz"], 123, delta=1)
            match = compare_modal_peaks([120], result["peaks"])
            self.assertEqual(match[0]["status"], "matched")

    def test_no_reuse_of_peak(self):
        result = compare_modal_peaks([100, 101], [{"frequency_hz": 100}])
        self.assertEqual([item["status"] for item in result], ["matched", "unidentified"])


if __name__ == "__main__":
    unittest.main()
