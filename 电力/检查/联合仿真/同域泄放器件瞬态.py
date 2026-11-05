"""Exercise the shared-supply candidate with vendor REF/comparator/driver models."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import 独立泄放启动建流 as previous


HERE = Path(__file__).resolve().parent
REFERENCE = """X_REF ANA TEMPO 0 TRIM REF REF5025
R_REFCAP REF REF_CAP 0.2
C_REF REF_CAP 0 1u
R_TRIM TRIM 0 1G
R_TEMPO TEMPO 0 1G"""
HELD_SUPPLY = """R_SLEW_A SRC12 SLEW_MID 22
R_SLEW_B SLEW_MID HW 22
C_HW HW 0 10u
D_HOLD HW ANA HOLD_D
.model HOLD_D D(Is=2n N=1.9 Rs=0.5 BV=100 Cjo=2p)
C_HOLD ANA 0 10u
R_ANALOG_EXTRA ANA 0 20k
"""


def run_case(work, executable, name, temperature, supply, permission, end_s,
             regen=False, reference_profile=None):
    supervisor = (
        f'V_CTL CTL 0 {permission}\n'
        'B_CTL READY 0 I={V(READY)*if(V(CTL)<0.5,1,0)}'
    )
    deck = previous.DECK.format(
        temp=temperature, supply=supply, old='PWL(0 12 27m 12 27.001m 0)',
        injection='PWL(0 0 27m 0 27.001m 450.03 38m 450.03 38.001m 0)' if regen else '0',
        bus=37.8, charger='PWL(0 1 26.99m 1 26.991m 0)' if regen else '1',
        reference=REFERENCE if reference_profile is None else f'V_REF REF 0 {reference_profile}\nI_REF_LOAD ANA 0 2m',
        r_high=143_600, ready_pullup='4.7k',
        ready_pulldown='100k', ready_cap='1p', supervisor=supervisor,
        resistor=1.0, inductor=100e-6, end=end_s,
    )
    deck = deck.replace('V_HW HW 0', 'V_HW SRC12 0')
    deck = deck.replace('V_OLD OLD 0', HELD_SUPPLY + 'V_OLD OLD 0')
    deck = deck.replace('X_COMP SENSE REF_SENSE HW', 'X_COMP SENSE REF_SENSE ANA')
    deck = deck.replace(f'tran 1u {end_s}', f'tran 1u {end_s} uic')
    deck = deck.replace('reltol=1e-4', 'reltol=1e-4 abstol=1e-10 vntol=1e-6 gmin=1e-10 rshunt=1e12')
    deck = deck.replace('method=gear', 'method=trap')
    deck = deck.replace('v(gate) i(l_dump)', 'v(gate) i(l_dump) v(ana)')
    (work / 'dump.cir').write_text(deck, encoding='ascii')
    wave = work / 'wave.txt'
    wave.unlink(missing_ok=True)
    process = subprocess.run(
        [str(executable), '-b', 'dump.cir'], cwd=work,
        capture_output=True, text=True, errors='replace', timeout=45,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
    )
    if process.returncode or not wave.is_file():
        raise RuntimeError((process.stdout + process.stderr)[-1800:])
    rows = [list(map(float, line.split())) for line in wave.read_text().splitlines()]
    if not rows or rows[-1][0] < end_s - 1e-6:
        raise RuntimeError(
            f'Incomplete transient at {rows[-1][0] if rows else None} s\n'
            + (process.stdout + process.stderr)[-1800:])
    ready_rows = [row for row in rows if row[6] >= 2.4]
    false_gate = [row for row in rows if row[6] <= 1.0 and row[8] > 1.0]
    injection_rows = [row for row in rows if 0.027001 <= row[0] <= 0.038]
    trip = next((row[0] for row in injection_rows if row[5] >= row[4]), None)
    building = next((row[0] for row in injection_rows
                     if trip is not None and row[0] >= trip and row[9] >= 450.03 / row[1]), None)
    return {
        'name': name, 'temperature_c': temperature, 'samples': len(rows),
        'reference_mode': 'vendor_macro' if reference_profile is None else 'imposed_sensitivity_waveform',
        'end_s': rows[-1][0], 'deck_sha256': sha256(deck.encode()).hexdigest(),
        'max_gate_while_ready_low_v': max((row[8] for row in rows if row[6] <= 1.0), default=0),
        'false_gate_samples_before_permission': len(false_gate),
        'max_gate_v': max(row[8] for row in rows),
        'first_ready_s': ready_rows[0][0] if ready_rows else None,
        'ref_at_first_ready_v': ready_rows[0][4] if ready_rows else None,
        'minimum_ref_while_ready_v': min((row[4] for row in ready_rows), default=None),
        'maximum_ref_while_ready_v': max((row[4] for row in ready_rows), default=None),
        'maximum_bus_v': max(row[1] for row in rows),
        'minimum_analogue_during_injected_sag_v':
            min((row[10] for row in rows if 0.027 <= row[0] <= 0.031), default=None),
        'threshold_to_net_current_s': building - trip if building is not None else None,
        'model_bounds': 'Supply and supervisor permission are imposed waveforms. Diode, BJT and power switch remain proxies. Vendor models are nominal and do not certify tolerances, physical rail faults or thermal capacity.',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-root', type=Path, default=HERE.parents[2])
    args = parser.parse_args()
    root = args.runtime_root.resolve()
    executable = root / '电力/FOC驱动模块/仿真/tools/ngspice-47-portable/Spice64/bin/ngspice_con.exe'
    diode = root / '电力/稳压模块/仿真/输入/厂家模型/ST_Schottky_PSpice_v6/ST_POWER_SCHOTTKY_V10.LIB'
    if not executable.is_file() or executable.stat().st_size < 1024:
        raise ValueError('A populated ngspice runtime is required; Git LFS pointers are not executable.')
    previous.DIODE = diode
    cases, failures, factorized_cases = [], [], []
    definitions = [
        ('cold_0c', 0, 'PWL(0 0 1m 0 1.1m 12.04875)',
         'PWL(0 0 24m 0 24.001m 1)', 0.030, False),
        ('cold_60c', 60, 'PWL(0 0 1m 0 1.1m 12.04875)',
         'PWL(0 0 24m 0 24.001m 1)', 0.030, False),
        ('warm_supply_retained_old_domain_withdrawn', 25,
         'PWL(0 0 1m 0 1.1m 11.66877)',
         'PWL(0 0 24m 0 24.001m 1)', 0.039, True),
        ('qualification_sag_and_full_reset', 25,
         'PWL(0 0 1m 0 1.1m 12.04875 27m 12.04875 27.001m 0 30m 0 30.001m 12.04875)',
         'PWL(0 0 24m 0 24.001m 1 27m 1 27.001m 0 69m 0 69.001m 1)',
         0.071, False),
        ('negative_control_permission_bypassed', 25,
         'PWL(0 0 1m 0 1.1m 12.04875)', '1', 0.005, False),
    ]
    with TemporaryDirectory(prefix='agtc-shared-dump-transient-') as temporary:
        work = Path(temporary)
        previous.prepare_models(work)
        for definition in definitions:
            try:
                case = run_case(work, executable, *definition)
                cases.append(case)
                print(definition[0], 'completed', flush=True)
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                failures.append({'name': definition[0], 'error': str(error)})
                print(definition[0], 'MODEL_FAILURE', flush=True)
        for definition in definitions:
            profile = 'PWL(0 0 1m 0 3m 2.5)'
            if definition[0] == 'negative_control_permission_bypassed':
                profile = 'PWL(0 0 1m 0 6m 2.5)'
            try:
                case = run_case(work, executable, *definition, reference_profile=profile)
                factorized_cases.append(case)
                print(definition[0], 'sensitivity completed', flush=True)
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                failures.append({'name': definition[0] + '_imposed_reference', 'error': str(error)})
    result = {
        'status': 'HOLD_NOMINAL_COMPONENT_TRANSIENT_ONLY',
        'source_models': previous.MODELS,
        'runtime_sha256': sha256(executable.read_bytes()).hexdigest(),
        'diode_model_sha256': sha256(diode.read_bytes()).hexdigest(),
        'cases': cases, 'model_failures': failures,
        'imposed_reference_sensitivity_cases': factorized_cases,
        'numerical_scope': 'Unconverged coupled REF5025 cases remain explicit. Separate imposed reference ramps exercise the comparator/driver and passive supply. They give waveform sensitivity and do not replace REF5025 qualification.',
        'scope': 'Local component cross-check, not the final all-board vehicle simulation. Real independent supply, supervisor, optocoupler, power MOS and measurements remain required.',
    }
    HERE.joinpath(Path(__file__).stem + '.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
