"""Focused contracts for read-only board resource sampling."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import realtime


class NpuLoadParsingTests(unittest.TestCase):
    def test_parses_the_board_driver_format_and_labels_each_core(self):
        raw = "NPU load:  Core0:  0%, Core1:  12%, Core2:  7%,\r"

        parsed = realtime._parse_npu_load(raw)

        self.assertEqual(parsed["status"], "ok")
        self.assertEqual(parsed["load_percent"], [0, 12, 7])
        self.assertEqual(parsed["average_percent"], 6.3)
        self.assertEqual(parsed["max_percent"], 12)
        self.assertIsNone(parsed["error"])

    def test_reorders_cores_and_keeps_missing_core_unknown(self):
        parsed = realtime._parse_npu_load("NPU load: Core2: 31%, Core0: 9%,")

        self.assertEqual(parsed["status"], "partial")
        self.assertEqual(parsed["load_percent"], [9, None, 31])
        self.assertIsNone(parsed["average_percent"])
        self.assertIsNone(parsed["max_percent"])
        self.assertIn("missing Core1", parsed["error"])

    def test_invalid_duplicate_and_out_of_range_values_are_not_used(self):
        cases = (
            ("Core0: 101%, Core1: 20%, Core2: 30%", [None, 20, 30], "out-of-range"),
            ("Core0: 10%, Core0: 11%, Core1: 20%, Core2: 30%",
             [None, 20, 30], "duplicate Core0"),
            ("Core0: busy%, Core1: 20%, Core2: 30%",
             [None, 20, 30], "invalid Core0 value"),
        )
        for raw, expected, error in cases:
            with self.subTest(raw=raw):
                parsed = realtime._parse_npu_load(raw)
                self.assertEqual(parsed["status"], "partial")
                self.assertEqual(parsed["load_percent"], expected)
                self.assertIsNone(parsed["average_percent"])
                self.assertIsNone(parsed["max_percent"])
                self.assertIn(error, parsed["error"])

    def test_unreadable_and_unrecognized_counters_are_unavailable(self):
        unreadable = realtime._parse_npu_load(None)
        malformed = realtime._parse_npu_load("NPU load: Core0: ?, Core1: x%, Core2: 101%")

        self.assertEqual(unreadable["status"], "unavailable")
        self.assertEqual(unreadable["error"], "unreadable")
        self.assertEqual(malformed["status"], "unavailable")
        self.assertIsNone(malformed["average_percent"])
        self.assertIsNone(malformed["max_percent"])
        self.assertIn("invalid Core0 value", malformed["error"])


class NpuDevfreqTests(unittest.TestCase):
    def test_devfreq_frequency_does_not_treat_100_at_frequency_as_load(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "fdab0000.npu"
            base.mkdir()
            (base / "cur_freq").write_text("1000000000\n")
            (base / "load").write_text("100@1000000000Hz\n")
            original_glob = Path.glob

            def fake_glob(path, pattern):
                if str(path).replace("\\", "/") == "/sys/class/devfreq":
                    return [base]
                return original_glob(path, pattern)

            with patch.object(Path, "glob", fake_glob):
                result = realtime._devfreq("npu", include_load=False)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["frequency_hz"], 1_000_000_000)
        self.assertIsNone(result[0]["load_percent"])


class HardwareSamplerCacheTests(unittest.TestCase):
    def test_reuses_one_sample_for_about_a_second_and_reports_age(self):
        sampler = realtime.HardwareSampler()
        cpu_samples = iter((
            {"cpu0": (100, 10)},
            {"cpu0": (200, 20)},
        ))
        monotonic_samples = iter((100.0, 100.4, 101.01))
        raw_load = "NPU load: Core0: 1%, Core1: 2%, Core2: 3%,"

        def read_text(path):
            if str(path) == realtime._NPU_LOAD_PATH:
                return raw_load
            if str(path) == realtime._NPU_POWER_PATH:
                return "off\n"
            if str(path) == realtime._NPU_VERSION_PATH:
                return "v0.9.6\n"
            return None

        with patch.object(realtime.time, "monotonic", side_effect=lambda: next(monotonic_samples)), \
                patch.object(realtime.time, "time", return_value=1_800_000_000.0), \
                patch.object(realtime, "_read_cpu", side_effect=lambda: next(cpu_samples)) as read_cpu, \
                patch.object(realtime, "_memory_kb", return_value={
                    "MemTotal": 1000, "MemAvailable": 400,
                    "SwapTotal": 0, "SwapFree": 0,
                }), \
                patch.object(realtime, "_process_memory", return_value={"VmRSS": 10}), \
                patch.object(realtime, "_process_dmabuf", return_value={
                    "bytes": 0, "objects": 0, "exporters": {},
                }), \
                patch.object(realtime, "_read_text", side_effect=read_text) as read_counter, \
                patch.object(Path, "glob", return_value=[]):
            first = sampler.snapshot(123)
            cached = sampler.snapshot(123)
            refreshed = sampler.snapshot(123)

        self.assertEqual(first["sample_seq"], 1)
        self.assertEqual(first["sample_age_ms"], 0)
        self.assertEqual(cached["sample_seq"], 1)
        self.assertEqual(cached["sample_age_ms"], 400)
        self.assertEqual(refreshed["sample_seq"], 2)
        self.assertEqual(read_cpu.call_count, 2)
        self.assertEqual(sum(call.args[0] == realtime._NPU_LOAD_PATH
                             for call in read_counter.call_args_list), 2)
        self.assertEqual(first["sampled_monotonic"], 100.0)
        self.assertEqual(first["npu"]["status"], "ok")
        self.assertEqual(first["npu"]["raw"], raw_load)
        self.assertEqual(first["npu"]["source"], realtime._NPU_LOAD_PATH)
        self.assertIsNone(first["npu"]["error"])
        self.assertEqual(cached["npu"]["power"], "off")
        self.assertEqual(cached["npu"]["driver_version"], "v0.9.6")


if __name__ == "__main__":
    unittest.main()
