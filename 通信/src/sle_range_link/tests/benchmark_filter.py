"""Build and measure a saved pre-change C source and current source on the host."""
import hashlib
import json
import pathlib
import platform
import statistics
import subprocess
import sys
import tempfile

from test_filter_differential import HERE, SRC


def build(source, exe, window):
    subprocess.run(["gcc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                    f"-DCONFIG_RANGE_LINK_FILTER_WIN={window}", "-I", str(HERE / "stubs"),
                    "-I", str(SRC), str(source), str(HERE / "filter_benchmark.c"),
                    "-o", str(exe)], check=True)


def main():
    baseline, output = pathlib.Path(sys.argv[1]).resolve(strict=True), pathlib.Path(sys.argv[2])
    current = SRC / "range_link_filter.c"
    result = {"platform": platform.platform(), "processor": platform.processor(),
              "compiler": subprocess.check_output(["gcc", "--version"], text=True).splitlines()[0],
              "flags": "-std=c11 -O2", "batch_pushes": 100000, "warmup_batches": 5,
              "measured_batches_per_run": 50, "runs_per_version": 3,
              "baseline_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
              "current_sha256": hashlib.sha256(current.read_bytes()).hexdigest(), "windows": {}}
    with tempfile.TemporaryDirectory() as directory:
        for window in (1, 16, 32):
            executables = {name: pathlib.Path(directory) / f"{name}-{window}.exe"
                           for name in ("baseline", "current")}
            build(baseline, executables["baseline"], window)
            build(current, executables["current"], window)
            runs = {name: [] for name in executables}
            checksums = set()
            for order in (("baseline", "current"), ("current", "baseline"), ("baseline", "current")):
                for name in order:
                    proc = subprocess.run([str(executables[name])], text=True, capture_output=True, check=True)
                    samples = [float(line) for line in proc.stdout.splitlines()]
                    assert len(samples) == 50
                    runs[name].append(samples)
                    checksums.add(proc.stderr.strip())
            assert len(checksums) == 1, checksums
            result["windows"][str(window)] = {"runs_seconds": runs, "checksum": checksums.pop(),
                "median_batch_seconds": {name: statistics.median(x for run in batches for x in run)
                                         for name, batches in runs.items()}}
            print(window, result["windows"][str(window)]["median_batch_seconds"])
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
