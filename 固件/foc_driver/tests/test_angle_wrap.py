"""Host regression for the real FOC angle helper; no target or hardware emulation.

Build test_angle_wrap.c with the include directories documented in the plan.
Usage: python test_angle_wrap.py AFTER_EXE [BEFORE_EXE]
"""
import math
import random
import struct
import subprocess
import sys


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def run(exe, value):
    result = subprocess.run([exe, value], capture_output=True, text=True,
                            timeout=2, check=True)
    return float.fromhex(result.stdout.strip())


def main():
    after = sys.argv[1]
    before = sys.argv[2] if len(sys.argv) > 2 else None
    period = f32(6.28318530718)
    unusual = ["nan", "inf", "-inf", "3.4028235e38", "-3.4028235e38",
               "1000000", "-1000000", "1e-45", "-1e-45",
               str(period), str(-period), "0", "-0"]
    for value in unusual:
        actual = run(after, value)
        assert math.isfinite(actual) and 0 <= actual < period, (value, actual)
        x = f32(float(value))
        if not math.isfinite(x):
            assert actual == 0
        else:
            expected = f32(math.fmod(x, period))
            if expected < 0:
                expected = f32(expected + period)
            if expected >= period:
                expected = 0.0
            assert actual == expected, (value, actual, expected)

    rng = random.Random(20261113)
    normal = [str(rng.uniform(-period, 2 * period)) for _ in range(100)]
    for value in normal:
        actual = run(after, value)
        assert 0 <= actual < period, (value, actual)
        if before:
            assert actual == run(before, value), (value, actual)
    print(f"PASS: {len(unusual)} boundary/non-finite and {len(normal)} normal inputs"
          + (", normal inputs match pre-change executable" if before else ""))


if __name__ == "__main__":
    main()
