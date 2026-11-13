"""Compare current filter with an explicitly supplied pre-change source.

Run on a host with GCC and Python: python test_filter_differential.py BASELINE_C
The SDK common_def.h stub is test-only. This does not exercise the SLE device.
"""
import ctypes
import _ctypes
import os
import pathlib
import random
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parent


def build(source, output, window):
    subprocess.run(["gcc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                    "-shared", f"-DCONFIG_RANGE_LINK_FILTER_WIN={window}",
                    "-I", str(HERE / "stubs"), "-I", str(SRC),
                    str(source), "-o", str(output)], check=True)


def load(path):
    lib = ctypes.CDLL(str(path))
    lib.rl_filter_push.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_uint32)]
    lib.rl_filter_push.restype = ctypes.c_int32
    return lib


def compare(old, new, window):
    a, b = load(old), load(new)
    try:
        compare_loaded(a, b, window)
    finally:
        if os.name == "nt":
            _ctypes.FreeLibrary(a._handle)
            _ctypes.FreeLibrary(b._handle)
        else:
            _ctypes.dlclose(a._handle)
            _ctypes.dlclose(b._handle)


def compare_loaded(a, b, window):
    a.rl_filter_init()
    b.rl_filter_init()
    rng = random.Random(20261113 + window)
    values = [1000, 1001, 1002, 1003] + [50000] * 13 + [0] * 13
    values += [rng.randrange(0, 50001) if i % 31 < 14 else
               10000 + rng.randrange(-400, 401) for i in range(50000)]
    for i, value in enumerate(values):
        if i % 997 == 0:
            a.rl_filter_reset()
            b.rl_filter_reset()
        na, nb = ctypes.c_uint32(), ctypes.c_uint32()
        if i % 17 == 0:
            x, y = a.rl_filter_push(value, None), b.rl_filter_push(value, None)
        else:
            x, y = a.rl_filter_push(value, ctypes.byref(na)), b.rl_filter_push(value, ctypes.byref(nb))
        assert (x, na.value) == (y, nb.value), (window, i, value, x, y, na.value, nb.value)
    print(f"window={window}: {len(values)} inputs match")


def main():
    baseline = pathlib.Path(sys.argv[1]).resolve(strict=True)
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        for window in (1, 2, 3, 8, 16, 32):
            old, new = root / f"old-{window}.dll", root / f"new-{window}.dll"
            build(baseline, old, window)
            build(SRC / "range_link_filter.c", new, window)
            compare(old, new, window)


if __name__ == "__main__":
    main()
