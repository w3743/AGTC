"""Compare C++ producer output against the existing Python wire contract."""
import pathlib
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "app"))
import depth_wire


class PipelineContractTest(unittest.TestCase):
    def test_cpp_frame_matches_existing_wire_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = pathlib.Path(tmp) / ("depth_test.exe" if sys.platform == "win32" else "depth_test")
            payload_path = pathlib.Path(tmp) / "depth.bin"
            subprocess.run(["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-pthread",
                            *(str(HERE / name) for name in ("depth_encode.cpp", "depth_filter.cpp", "depth_store.cpp",
                               "depth_producer.cpp", "depth_request.cpp", "test_depth_pipeline.cpp")),
                            "-o", str(exe)], check=True)
            result = subprocess.run([str(exe), str(payload_path)], check=True,
                                    capture_output=True, text=True)
            metadata = depth_wire.parse_depth_metadata(result.stdout.strip())
            self.assertEqual(metadata["seq"], 1)
            self.assertEqual(metadata["pipeline_epoch"], 1)
            self.assertEqual(metadata["session_id"], 'unit-test"session')
            self.assertEqual(metadata["left_pts_ns"], 9007199254740993)
            self.assertIsNone(metadata["extrinsics_id"])
            pixels = np.full((240, 131), 1.25, dtype=np.float32)
            pixels[0, :4] = [0, .3, 5, np.nan]
            expected, _, _ = depth_wire.encode_depth_payload(pixels)
            self.assertEqual(payload_path.read_bytes(), expected)
            depth_wire.build_depth_record(payload_path.read_bytes(), metadata)


if __name__ == "__main__":
    unittest.main()
