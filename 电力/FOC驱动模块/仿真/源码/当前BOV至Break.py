"""Probe the saved BOV circuit with the TI TLV6700 model and generic BJTs."""
import hashlib
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path

FOC = Path(__file__).resolve().parents[2]
SPICE = FOC / "仿真/tools/ngspice-47-portable/Spice64/bin/ngspice_con.exe"
PACKAGE = FOC.parent / "稳压模块/仿真/输入/厂家模型/snvmc71a.zip"
NETLIST = FOC / "eda工程文件/导出/当前交付/FOC_G431_P1.enet"

TEMPLATE = """* FOC U7 OUTB and existing two-transistor interface
.include tlv.lib
Vdump dump 0 {dump}
Vmcu mcu 0 {mcu}
Vbus bus 0 PWL(0 37.8 1m 37.8 2m 45 3m 45 4m 37.8 5m 37.8)
Rtop bus sense 1Meg
Rbot sense 0 10k
Csense sense 0 1n
XU7 0 sense dump 0 outa busov TLV6700
Rovpu dump busov 4.7k
Rbase busov pbase 10k
Qp pcol pbase dump QP
Rdrive pcol nbase 4.7k
Rpd nbase 0 10k
Qn break_n nbase 0 QN
Rbreak mcu break_n 2.2k
Cbreak break_n 0 100p
Vother other 0 PWL(0 0 0.6m 0 0.61m {other} 0.8m {other} 0.81m 0 5m 0)
Sother break_n 0 other 0 OPEN_DRAIN
.model OPEN_DRAIN SW(RON=40 ROFF=1G VT=0.5 VH=0.1)
.model QP PNP(IS=1e-14 BF={beta} VAF=70)
.model QN NPN(IS=1e-14 BF={beta} VAF=80)
.control
set noaskquit
set wr_singlescale
tran 0.5u 5m 0 0.5u
wrdata wave.txt v(bus) v(sense) v(busov) v(break_n) i(Vmcu)
quit
.endc
.end
"""


def run_case(folder, dump, mcu, beta, other=0):
    wave = folder / "wave.txt"
    wave.unlink(missing_ok=True)
    (folder / "case.cir").write_text(TEMPLATE.format(dump=dump, mcu=mcu, beta=beta, other=other),
                                     encoding="ascii")
    run = subprocess.run([SPICE, "-b", "case.cir"], cwd=folder, capture_output=True,
                         timeout=40, creationflags=subprocess.CREATE_NO_WINDOW)
    if run.returncode or not wave.exists():
        raise RuntimeError((run.stdout + run.stderr).decode("utf-8", errors="replace")[-2000:])
    rows = [tuple(map(float, line.split())) for line in wave.read_text().splitlines()]
    assert rows[-1][0] >= 0.00499
    def sample(t):
        row = min(rows, key=lambda x: abs(x[0] - t))
        return dict(zip(("t_s", "BUS_V", "sense_V", "BUSOV_N_V", "BREAK_N_V", "mcu_source_A"), row))
    values = {name: sample(t) for name, t in (("healthy", 0.0005), ("other_fault", 0.0007),
                                             ("overvoltage", 0.0025), ("recovered", 0.0048))}
    crossings = [r for r in rows if 0.001 <= r[0] <= 0.003 and r[4] < 0.3 * mcu]
    return {"DUMP_5V_V": dump, "MCU_3V3_V": mcu, "generic_bjt_BF": beta, "other_fault_injected": bool(other),
            "samples": values, "BUS_at_first_OV_Break_V": crossings[0][1] if crossings else None}


def main():
    with tempfile.TemporaryDirectory(prefix="foc_bov_spice_") as tmp:
        folder = Path(tmp)
        (folder / ".spiceinit").write_text("set ngbehavior=ps\n", encoding="ascii")
        with zipfile.ZipFile(PACKAGE) as package:
            model = package.read("tlv6700.lib")
        (folder / "tlv.lib").write_bytes(model)
        cases = [run_case(folder, dump, mcu, beta, other=1)
                 for dump, mcu in ((4.78, 3.0), (5.0, 3.3), (5.20, 3.6))
                 for beta in (30, 100, 420)]
        power_loss = [run_case(folder, 5, 0, 100), run_case(folder, 0, 3.3, 100)]
    for case in cases:
        v = {key: sample["BREAK_N_V"] for key, sample in case["samples"].items()}
        mcu = case["MCU_3V3_V"]
        assert v["healthy"] > .7 * mcu and v["recovered"] > .7 * mcu, case
        assert v["overvoltage"] < .3 * mcu and v["other_fault"] < .3 * mcu, case
    result = {
        "status": "NINE_POWERED_MODEL_CASES_PASS_POWER_LOSS_DIAGNOSTIC_ONLY",
        "netlistSha256": hashlib.sha256(NETLIST.read_bytes()).hexdigest(),
        "tiTLV6700MacroSha256": hashlib.sha256(model).hexdigest(),
        "scope": ["U7 OUTB in TI macro model and generic BJT models using the saved network.",
                  "Ideal supplies; generic BJT models and 100 pF node load are sensitivity assumptions.",
                  "The two fixed zero-rail runs only expose modeled behavior; no power-off leakage or ramp guarantee.",
                  "Does not include timer, six PWM outputs, MOS gate dynamics, motor, chopper, or shutdown energy."],
        "cases": cases, "fixedZeroRailDiagnostics": power_loss,
    }
    output = FOC / "仿真/结果/当前BOV至Break.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("9 powered circuit cases pass; 2 fixed zero-rail diagnostics")


if __name__ == "__main__":
    main()
