"""Offline conversion rejects invalid inputs without overwriting evidence."""
import io
import pathlib
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SOURCE = HERE.parent / "app" / "calibration.npz"


class ExportCalibrationTest(unittest.TestCase):
    def test_valid_and_invalid_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            output = root / "calibration.yml"
            cmd = [sys.executable, "-B", str(HERE / "export_calibration.py")]
            subprocess.run([*cmd, str(SOURCE), str(output)], check=True)
            valid_bytes = output.read_bytes()
            self.assertIn(b"agtc-stereo-calibration-v1", valid_bytes)
            duplicate = subprocess.run([*cmd, str(SOURCE), str(output)], capture_output=True)
            self.assertNotEqual(duplicate.returncode, 0)
            self.assertEqual(output.read_bytes(), valid_bytes)
            with np.load(io.BytesIO(SOURCE.read_bytes()), allow_pickle=False) as data:
                model = {key: data[key].copy() for key in data.files}
            model["status"] = np.array("unverified")
            bad = root / "bad.npz"
            np.savez(bad, **model)
            rejected = root / "rejected.yml"
            result = subprocess.run([*cmd, str(bad), str(rejected)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(rejected.exists())


if __name__ == "__main__":
    unittest.main()
