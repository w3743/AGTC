"""Compare the C filter with an arbitrary-precision reference on full int32 inputs."""
import _ctypes
import ctypes
import os
import pathlib
import random
import sys
import tempfile

from test_filter_differential import SRC, build, load


class Reference:
    def __init__(self, window):
        self.window = window
        self.reset()

    def reset(self):
        self.values = []
        self.index = 0
        self.total = 0
        self.streak = 0
        self.last = 0

    def push(self, sample):
        if self.values:
            median = sorted(self.values)[len(self.values) // 2]
            if abs(sample - median) > max(500, abs(median) * 30 // 100):
                self.streak += 1
                if self.streak < 12:
                    return self.last, self.total
                self.values = []
                self.index = 0
        self.streak = 0
        if len(self.values) < self.window:
            self.values.append(sample)
        else:
            self.values[self.index] = sample
        self.index = (self.index + 1) % self.window
        self.total = (self.total + 1) & 0xffffffff
        if len(self.values) >= 3:
            self.last = sorted(self.values)[len(self.values) // 2]
        else:
            s = sum(self.values)
            self.last = (abs(s) // len(self.values)) * (-1 if s < 0 else 1)
        return self.last, self.total


def main():
    source = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else SRC / "range_link_filter.c"
    boundaries = [-2147483648, -2147483647, -100000000, -71582789, -1, 0, 1,
                  71582788, 71582789, 100000000, 2147483646, 2147483647]
    with tempfile.TemporaryDirectory() as directory:
        for window in (1, 2, 3, 8, 16, 32):
            path = pathlib.Path(directory) / f"filter-{window}.dll"
            build(source, path, window)
            lib = load(path)
            try:
                ref = Reference(window)
                lib.rl_filter_init()
                rng = random.Random(20261114 + window)
                samples = [2147483647] * 3 + [-2147483648] * 13
                samples += [x for x in boundaries for _ in range(14)]
                for i in range(30000):
                    if i % 27 < 14:
                        samples.append(boundaries[(i // 27) % len(boundaries)])
                    else:
                        samples.append(rng.randint(-2147483648, 2147483647))
                for i, value in enumerate(samples):
                    if i % 997 == 0:
                        lib.rl_filter_reset()
                        ref.reset()
                    count = ctypes.c_uint32()
                    actual = lib.rl_filter_push(value, ctypes.byref(count)), count.value
                    expected = ref.push(value)
                    assert actual == expected, (window, i, value, actual, expected)
                print(f"window={window}: {len(samples)} full-range samples match reference")
            finally:
                (_ctypes.FreeLibrary if os.name == "nt" else _ctypes.dlclose)(lib._handle)


if __name__ == "__main__":
    main()
