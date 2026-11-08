"""Run an ngspice switch-level check of the saved pump bridge's power paths.

Usage: python 功率路径模型.py <absolute path to ngspice_con.exe>
The fixed R/L/EMF load is a test fixture. No motor or IC macro-model is used.
"""

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory


CHECK = Path(__file__).resolve().parent
SOURCE = CHECK.parent / "eda工程文件/水泵驱动/sch/Schematic1/P1.esch2"


def run(exe, directory, name, vm, bus_source, hs_a, hs_b, ls_b, measurements):
    deck = f"""Pump H-bridge path fixture: {name}
{bus_source}
Cvm vm 0 470u IC={vm}
Vha ha 0 {hs_a}
Vhb hb 0 {hs_b}
Vlb lb 0 {ls_b}
Bha vm a I=V(vm,a)*(1e-7+40*V(ha))
Bhb vm b I=V(vm,b)*(1e-7+40*V(hb))
Blb b 0 I=V(b)*(1e-7+40*V(lb))
Dha a vm BODY
Dhb b vm BODY
Dla 0 a BODY
Dlb 0 b BODY
Ca a 0 1n
Cb b 0 1n
Rw a r 1
Lw r e 500u IC=3
Vemf e b 18
.model BODY D(Is=10n Rs=0.02)
.options method=gear
.tran 0.1u 2m UIC
.print tran v(vm)
{measurements}
.end
"""
    path = directory / f"{name}.cir"
    path.write_text(deck, encoding="ascii")
    log = subprocess.run([exe, "-b", str(path)], text=True, capture_output=True)
    values = dict(re.findall(r"^([a-z_]+)\s*=\s*([+\-\d.eE]+)", log.stdout, re.MULTILINE))
    expected = set(re.findall(r"^\.meas tran (\w+)", measurements, re.MULTILINE))
    if log.returncode or not expected <= values.keys():
        raise RuntimeError(log.stdout[:3000] + log.stderr)
    return {key: float(values[key]) for key in sorted(expected)}


def main(exe):
    with TemporaryDirectory(prefix="pump_power_paths_") as tmp:
        directory = Path(tmp)
        results = {}
        for name, vm in (("pwm_low", 22.5), ("pwm_high", 37.8)):
            on_us = 50 * 21 / vm
            results[name] = run(
                exe, directory, name, vm, f"Vbus vm 0 {vm}", "1",
                f"PULSE(0 1 {on_us + 0.15}u 100n 100n {49.7-on_us}u 50u)",
                f"PULSE(1 0 {on_us}u 100n 100n {49.8-on_us}u 50u)",
                """.meas tran drive_v FIND par('v(a)-v(b)') AT=1u
.meas tran recirc_v FIND par('v(a)-v(b)') AT=49u
.meas tran avg_v AVG par('v(a)-v(b)') FROM=1m TO=2m
.meas tran min_i MIN i(Lw) FROM=1m TO=2m
.meas tran max_i MAX i(Lw) FROM=1m TO=2m""",
            )
            assert abs(results[name]["drive_v"] - vm) < 0.3
            assert abs(results[name]["recirc_v"]) < 0.3
            assert 20.7 < results[name]["avg_v"] < 21
            assert 0 < results[name]["min_i"] < results[name]["max_i"]
        results["brake"] = run(
            exe, directory, "brake", 37.8, "Vbus vm 0 37.8", "1", "1", "0",
            """.meas tran min_i MIN i(Lw) FROM=1u TO=2m
.meas tran max_abs_bus_i MAX par('abs(i(Vbus))') FROM=1u TO=2m""",
        )
        assert results["brake"]["min_i"] < -5.19
        assert results["brake"]["max_abs_bus_i"] < 1e-5
        results["fault_coast_isolated_bus"] = run(
            exe, directory, "coast", 37.8, "Rleak vm 0 1G", "0", "0", "0",
            """.meas tran peak_vm MAX v(vm) FROM=1n TO=2m
.meas tran current_end FIND i(Lw) AT=2m""",
        )
        assert results["fault_coast_isolated_bus"]["peak_vm"] > 37.8
        assert abs(results["fault_coast_isolated_bus"]["current_end"]) < 1e-5

    report = {
        "p1_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixture": {"winding_ohm": 1, "winding_h": 0.0005, "emf_v": 18,
                    "initial_current_a": 3, "pwm_hz": 20000, "bulk_f": 0.00047,
                    "switch_on_ohm": 0.025},
        "coverage": "Path-level test. No regulation, OCP, delay, moving rotor, TVS, fuse, supply or harness transient model.",
        "measurements": results,
    }
    CHECK.joinpath("功率路径模型.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
