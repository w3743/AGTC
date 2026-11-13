"""Offline appsrc -> native stereo -> HTTP response -> existing depth probe."""
import hashlib
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import unittest
from email.parser import BytesParser
from email.policy import default

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "app"))
import depth_wire
from probe_depth import DepthProbeCollector


class GstEndToEndTest(unittest.TestCase):
    def test_synthetic_nv12(self):
        executable = os.environ.get("AGTC_GST_FIXTURE")
        if not executable:
            self.skipTest("Set AGTC_GST_FIXTURE to the built test executable")
        calibration = HERE.parent / "app" / "calibration.npz"
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            yaml, output = root / "calibration.yml", root / "response.http"
            subprocess.run([sys.executable, "-B", str(HERE / "export_calibration.py"),
                            str(calibration), str(yaml)], check=True)
            subprocess.run([executable, str(yaml), str(output)], check=True, timeout=30)
            headers, body = output.read_bytes().split(b"\r\n\r\n", 1)
            status, raw_headers = headers.split(b"\r\n", 1)
            self.assertEqual(status, b"HTTP/1.1 200 OK")
            parsed = BytesParser(policy=default).parsebytes(raw_headers + b"\r\n\r\n")
            collector = DepthProbeCollector()
            try:
                now = time.perf_counter_ns()
                row = collector.ingest_response(status=200, headers=parsed.items(), body=body,
                    host_send_monotonic_ns=now, host_receive_monotonic_ns=now)
                self.assertEqual(row["validation_errors"], [])
            finally:
                collector.close_archive()
            metadata = depth_wire.parse_depth_metadata(parsed["X-Depth-Metadata"])
            self.assertEqual(metadata["seq"], 1)
            self.assertEqual((metadata["left_pts_ns"], metadata["right_pts_ns"]), (1_000_000_000, 1_010_000_000))
            self.assertEqual(metadata["calibration_id"], hashlib.sha256(calibration.read_bytes()).hexdigest()[:12])
            self.assertIsNone(metadata["source_mono_ns"])
            self.assertIsNone(metadata["extrinsics_id"])
            self.assertEqual(len(body), 131 * 240 * 2)


if __name__ == "__main__":
    unittest.main()
