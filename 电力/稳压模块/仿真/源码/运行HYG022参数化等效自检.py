"""Run bounded HYG022 charge/diode/B2B fixture checks with bundled ngspice."""
from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / "仿真"
EXE = ROOT.parent / "FOC驱动模块/仿真/tools/ngspice-47-portable/Spice64/bin/ngspice_con.exe"
MODEL = SIM / "源码/HYG022_参数化等效模型.lib"
CIRCUIT = SIM / "源码/HYG022_参数化等效自检.cir"
STATIC_CIRCUIT = SIM / "源码/HYG022_参数化等效静态自检.cir"
OUTPUT = SIM / "结果/HYG022_参数化等效自检.json"
WAVE_OUTPUT = SIM / "结果/HYG022_参数化等效栅电荷波形.txt"
STATIC_LOG_OUTPUT = SIM / "结果/HYG022_参数化等效静态自检.log"
HYG_DATASHEET = ROOT.parent / "BMS模块/厂商资料/HYG022N10NS1TA.pdf"
IRLB_DATASHEET = SIM / "输入/厂家模型/IRLB4030PBF_Infineon_官方数据手册.pdf"
IPF042_DATASHEET = SIM / "输入/厂家模型/IPF042N10NF2S_Infineon_官方数据手册.pdf"
INFINEON_100V_ZIP = SIM / "输入/厂家模型/Infineon-OptiMOS_Power_MOSFET_Spice_100_V_N-Channel-SimulationModels-v04_00-EN.zip"
INFINEON_LIB_MEMBER = "StrongIRFET/StrongIRFET_100V_LTSpice.lib"


def qgs(v: float) -> float:
    if v < 2.0:
        return 9.026e-9 * v
    if v < 4.2:
        return 18.052e-9 + 14.97636364e-9 * (v - 2.0)
    if v < 4.4:
        return 51e-9 + 220e-9 * (v - 4.2)
    return 95e-9 + 11.69642857e-9 * (v - 4.4)


def trapz(y: list[float], x: list[float]) -> float:
    return sum((x[i + 1] - x[i]) * (y[i] + y[i + 1]) / 2 for i in range(len(x) - 1))


def printed_scalar(log: str, name: str) -> float:
    import re

    match = re.search(rf"(?im)^\s*{re.escape(name)}\s*=\s*([-+0-9.eE]+)\s*$", log)
    if not match:
        raise RuntimeError(f"static fixture did not print {name}:\n{log[-3000:]}")
    return float(match.group(1))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


with zipfile.ZipFile(INFINEON_100V_ZIP) as archive:
    library_text = archive.read(INFINEON_LIB_MEMBER).decode("latin1")
if "IPF042N10NF2S_L1" not in library_text:
    raise RuntimeError("IPF042N10NF2S_L1 is missing from the Infineon library member")


with tempfile.TemporaryDirectory(prefix="hyg022_eq_") as temp:
    work = Path(temp)
    shutil.copyfile(MODEL, work / MODEL.name)
    shutil.copyfile(CIRCUIT, work / CIRCUIT.name)
    proc = subprocess.run(
        [str(EXE.resolve()), "-b", CIRCUIT.name],
        cwd=work,
        capture_output=True,
        timeout=60,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    log = proc.stdout.decode("utf-8", "replace") + proc.stderr.decode("utf-8", "replace")
    wave_path = work / "hyg022_gatecharge_wave.txt"
    if proc.returncode or "Error:" in log or not wave_path.exists():
        raise RuntimeError(log[-5000:])
    rows = [list(map(float, line.split())) for line in wave_path.read_text().splitlines() if len(line.split()) == 4]
    if len(rows) < 100 or rows[-1][0] < 6.99e-6:
        raise RuntimeError(f"incomplete waveform: rows={len(rows)} last={rows[-1][0] if rows else None} tail={rows[-5:]}\n{log[-3000:]}")

    shutil.copyfile(STATIC_CIRCUIT, work / STATIC_CIRCUIT.name)
    static_proc = subprocess.run(
        [str(EXE.resolve()), "-b", STATIC_CIRCUIT.name],
        cwd=work,
        capture_output=True,
        timeout=60,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    static_log = static_proc.stdout.decode("utf-8", "replace") + static_proc.stderr.decode("utf-8", "replace")
    if static_proc.returncode or "Error:" in static_log:
        raise RuntimeError(static_log[-5000:])
    WAVE_OUTPUT.write_bytes(wave_path.read_bytes())
    STATIC_LOG_OUTPUT.write_text(static_log, encoding="utf-8")
    pair_i = printed_scalar(static_log, "i_on")
    pair_v = printed_scalar(static_log, "v_pair")
    diode_i = printed_scalar(static_log, "i_diode")
    diode_v = printed_scalar(static_log, "v_diode")
    leak_pos = printed_scalar(static_log, "leak_positive")
    leak_neg = printed_scalar(static_log, "leak_negative")

    # wrdata columns: time, V(gate), V(drv), I(VDRV).
    times = [r[0] for r in rows]
    vgate = [r[1] for r in rows]
    vdrv = [r[2] for r in rows]
    idrv_in = [-r[3] for r in rows]
    vgi = [vgate[i] - idrv_in[i] * 2.3 for i in range(len(rows))]
    qtotal = [qgs(vgi[i]) for i in range(len(rows))]
    q_in = trapz(idrv_in, times)
    q_model = qtotal[-1] - qtotal[0]
    e_driver = trapz([vdrv[i] * idrv_in[i] for i in range(len(rows))], times)
    e_rgate = trapz([idrv_in[i] ** 2 * 4.8 for i in range(len(rows))], times)
    e_gate_charge = sum(
        (vgi[i] + vgi[i + 1]) * (qtotal[i + 1] - qtotal[i]) / 2
        for i in range(len(rows) - 1)
    )

    final = rows[-1]
    charge_phase = [
        vgi[i] for i, r in enumerate(rows) if 1.0e-6 < r[0] < 6.0e-6 and 4.2 <= vgi[i] <= 4.4
    ]
    if not charge_phase:
        raise RuntimeError("the load did not produce a measurable gate-charge plateau interval")
    plateau_vgs = charge_phase
    expected = 160.5e-9
    result = {
        "status": "PARAMETERIZED_EQUIVALENT_NUMERICAL_FIXTURE_ONLY",
        "simulator": "ngspice 47 portable",
        "source_files": [MODEL.name, CIRCUIT.name],
        "official_model_search": {
            "HYG022N10NS1TA": {
                "exact_public_manufacturer_spice_model_found": False,
                "official_manufacturer_site_checked": "https://www.hymexa.com",
                "manufacturer_datasheet_url": "https://atta.szlcsc.com/upload/public/pdf/source/20211129/85A27B49583C97E874E50AC12D8AF79A.pdf",
                "result": "No exact public SPICE download found; manufacturer website fetch timed out. Local manufacturer datasheet remains the parameter source.",
            },
            "IRLB4030PBF": {
                "exact_manufacturer_spice_model_found": False,
                "official_product_page": "https://www.infineon.com/part/IRLB4030",
                "official_datasheet_url": "https://www.infineon.com/dgdl/irlb4030pbf.pdf?fileId=5546d462533600a4015356604640258d",
                "official_100V_library": "https://www.infineon.com/dgdl/Infineon-OptiMOS_Power_MOSFET_Spice_100_V_N-Channel-SimulationModels-v04_00-EN.zip?fileId=db3a30433d346a2d013d459a89bf5736",
                "library_inspection": "StrongIRFET/StrongIRFET_100V_LTSpice.lib has 218 subcircuits and no IRLB4030 model.",
            },
        },
        "infineon_neighbor_sensitivity_candidate": {
            "part_number": "IPF042N10NF2S",
            "official_library_subcircuit": "IPF042N10NF2S_L1",
            "official_datasheet_url": "https://www.infineon.com/assets/row/public/documents/24/49/infineon-ipf042n10nf2s-datasheet-en.pdf",
            "datasheet_package": "PG-TO263-7 (D2PAK 7-pin)",
            "IRLB4030PBF_package": "TO-220AB",
            "values": {
                "VDS_V": {"IPF042": 100, "IRLB4030": 100, "relative_difference_percent": 0.0},
                "RDSon_typ_mOhm": {"IPF042": 3.7, "IRLB4030": 3.4, "relative_difference_percent": 8.8235, "conditions": "IPF042: VGS=10 V, ID=80 A; IRLB4030: VGS=10 V, ID=110 A"},
                "RDSon_max_mOhm": {"IPF042": 4.25, "IRLB4030": 4.3, "relative_difference_percent": -1.1628, "conditions": "Different test currents; IPF042 80 A, IRLB4030 110 A"},
                "Qg_typ_nC": {"IPF042": 57, "IRLB4030": 87, "relative_difference_percent": -34.4828, "conditions": "IPF042: VDD=50 V, ID=80 A, VGS=0..10 V; IRLB4030 table/product specification uses a different gate-charge test point"},
                "Ciss_pF": {"IPF042": 4000, "IRLB4030": 11360, "relative_difference_percent": -64.7887, "conditions": "Both at VGS=0 V, VDS=50 V, f=1 MHz"},
                "Crss_pF": {"IPF042": 28, "IRLB4030": 290, "relative_difference_percent": -90.3448, "conditions": "Both at VGS=0 V, VDS=50 V, f=1 MHz"},
                "Coss_pF": {"IPF042": 630, "IRLB4030": 670, "relative_difference_percent": -5.9701, "conditions": "Both at VGS=0 V, VDS=50 V, f=1 MHz"},
            },
            "assessment": "Only a same-100-V-class sensitivity candidate. The notably lower Ciss/Crss and different Qg, diode behavior and package prevent treating it as an IRLB4030PBF equivalent or drop-in substitute.",
            "not_simulated_as_IRLB4030": True,
        },
        "datasheet_fit": {
            "device": "HYG022N10NS1TA",
            "charge_test_target_C": expected,
            "simulated_gate_charge_C": q_in,
            "charge_from_terminal_state_C": q_model,
            "terminal_charge_balance_error_percent": 100 * (q_in - q_model) / expected,
            "gate_driver_energy_J": e_driver,
            "gate_resistor_energy_J": e_rgate,
            "Q(VGS)_branch_work_J": e_gate_charge,
            "gate_energy_balance_error_percent": 100 * (e_driver - e_rgate - e_gate_charge) / max(e_driver, 1e-30),
            "final_vgs_V": vgi[-1],
            "gate_charge_plot_plateau_read_from_datasheet_V": 4.2,
            "finite_plateau_fit_range_V": [4.2, 4.4],
            "simulated_internal_vgs_inside_fitted_plateau_segment_V_min_max": [min(plateau_vgs), max(plateau_vgs)],
            "test_conditions": "single-device gate port; 10 V source through 2.5 ohm external + 2.3 ohm internal Rg; no drain/load dynamics. Curve interpolates datasheet Fig.10 and Qg table.",
        },
        "body_diode_forward_fit": {
            "datasheet_target_current_A": 50.0,
            "simulated_current_A": diode_i,
            "simulated_vsd_V": diode_v,
            "datasheet_typ_vsd_V": 0.83,
            "datasheet_max_vsd_V": 1.3,
            "test_temperature_C": 25,
            "reverse_recovery": "not modeled; no TT/Qrr behavior",
        },
        "b2b_static_screen": {
            "gate_vgs_V": 10.0,
            "target_current_A": 80.0,
            "simulated_current_A": pair_i,
            "simulated_pair_drop_V": pair_v,
            "computed_pair_rds_typ_ohm": 2 * 2.2e-3,
            "computed_pair_loss_W": pair_i * pair_v,
            "both_polarity_20V_gate_off_leakage_A": [leak_pos, leak_neg],
            "interpretation": "DC-only fixture matching the 10 V RDS(on) typical point; not the module's switching/load-current waveform",
        },
        "non_covered": [
            "No VDS-coupled Miller behavior or actual MOS turn-on/turn-off waveform. The 4.2..4.4 V finite plateau width is a curve-smoothing assumption; the datasheet plot itself is approximately flat.",
            "No verified reverse-recovery transient. Datasheet Qrr=239 nC and trr=93.3 ns are typical at ISD=50 A and dISD/dt=100 A/us, but the equivalent intentionally has no TT/Qrr recovery behavior.",
            "No DC/DC or vehicle load-current waveform, gate-driver current limiting, thermal coupling, avalanche, SOA, package/layout parasitics, Miller-induced turn-on margin, or fault turn-off guarantee.",
            "The Qg/Qgs/Qgd table uses VDS=80 V while Fig.10 uses VDS=50 V. Gate charge is a typical single-device curve fixture, not validation of the module's operating waveform or the two-device pair's real gate charge under its voltage sharing.",
        ],
    }

    with zipfile.ZipFile(INFINEON_100V_ZIP) as archive:
        library_member_sha256 = sha256_bytes(archive.read(INFINEON_LIB_MEMBER))
    result["source_sha256"] = {
        "files": {
            str(MODEL.relative_to(ROOT)).replace("\\", "/"): sha256_path(MODEL),
            str(CIRCUIT.relative_to(ROOT)).replace("\\", "/"): sha256_path(CIRCUIT),
            str(STATIC_CIRCUIT.relative_to(ROOT)).replace("\\", "/"): sha256_path(STATIC_CIRCUIT),
            str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"): sha256_path(Path(__file__).resolve()),
            "../BMS模块/厂商资料/HYG022N10NS1TA.pdf": sha256_path(HYG_DATASHEET),
            str(IRLB_DATASHEET.relative_to(ROOT)).replace("\\", "/"): sha256_path(IRLB_DATASHEET),
            str(IPF042_DATASHEET.relative_to(ROOT)).replace("\\", "/"): sha256_path(IPF042_DATASHEET),
            str(INFINEON_100V_ZIP.relative_to(ROOT)).replace("\\", "/"): sha256_path(INFINEON_100V_ZIP),
        },
        "zip_member_sha256": {
            INFINEON_LIB_MEMBER: library_member_sha256,
        },
    }
    result["result_artifact_sha256"] = {
        WAVE_OUTPUT.name: sha256_path(WAVE_OUTPUT),
        STATIC_LOG_OUTPUT.name: sha256_path(STATIC_LOG_OUTPUT),
    }

checks = {
    "single_device_gate_charge_within_1pct": abs(result["datasheet_fit"]["simulated_gate_charge_C"] / 160.5e-9 - 1) <= 0.01,
    "gate_terminal_charge_kcl_within_1pct": abs(result["datasheet_fit"]["terminal_charge_balance_error_percent"]) <= 1.0,
    "gate_network_energy_balance_within_1pct": abs(result["datasheet_fit"]["gate_energy_balance_error_percent"]) <= 1.0,
    "diode_forward_current_within_1pct": abs(result["body_diode_forward_fit"]["simulated_current_A"] / 50 - 1) <= 0.01,
    "diode_forward_voltage_within_0p01V": abs(result["body_diode_forward_fit"]["simulated_vsd_V"] - 0.83) <= 0.01,
    "b2b_static_path_within_1pct_of_80A": abs(result["b2b_static_screen"]["simulated_current_A"] / 80 - 1) <= 0.01,
    "both_polarity_off_leakage_below_2uA": max(result["b2b_static_screen"]["both_polarity_20V_gate_off_leakage_A"]) < 2e-6,
}
result["checks"] = checks
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
if not all(checks.values()):
    raise SystemExit("one or more bounded equivalent fixture checks failed")
