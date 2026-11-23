"""Run the six pre-EDA comparisons on the repository ngspice executable."""

from hashlib import sha256
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
POWER = HERE.parents[1]
SPICE = POWER / 'FOC驱动模块/仿真/tools/ngspice-47-portable/Spice64/bin/ngspice_con.exe'
OUT = HERE / '降本后典型故障联合仿真.json'
BEFORE = dict(foc_cap_f=.030, reg_cap_f=.0054, foc_esr_ohm=.023/3,
              reg_esr_ohm=.032/6, foc_mos_ohm=.0015, reg_mos_ohm=.0043,
              foc_qg_nc=211, cap_rating_v=80, foc_rating_v=100)
AFTER = dict(foc_cap_f=.030, reg_cap_f=.006, foc_esr_ohm=.025/3,
             reg_esr_ohm=.035/6, foc_mos_ohm=.0028, reg_mos_ohm=.0028,
             foc_qg_nc=160.5, cap_rating_v=63, foc_rating_v=63)


def spice(work, name, circuit, step, stop, measures):
    path = work / f'{name}.cir'
    path.write_text(f'{name}\n{circuit}\n.options method=gear\n'
                    f'.tran {step} {stop} 0 {step} uic\n.control\nrun\n'
                    f'{measures}\nquit\n.endc\n.end\n', encoding='ascii')
    run = subprocess.run([SPICE, '-b', path.name], cwd=work, capture_output=True,
                         encoding='utf-8', errors='replace', check=True, timeout=90)
    log = run.stdout + run.stderr
    expected = set(re.findall(r'^meas tran (\w+)', measures, re.M))
    values = {key: float(value) for key, value in re.findall(
        r'^([a-z][a-z0-9_]*)\s*=\s*([-+\d.eE]+)', log, re.M) if key in expected}
    if expected != values.keys() or not all(math.isfinite(v) for v in values.values()):
        raise RuntimeError(f'{name}: missing {expected - values.keys()}\n{log[-4300:]}')
    return values


def rails(parts, tolerance):
    return 2 * parts['foc_cap_f'] * tolerance + parts['reg_cap_f'] * tolerance + .00054


def regulator_two_phase(work, contract):
    """20 kHz ideal switched current and local BUS fault-tail screens."""
    spec = contract['p1_two_phase']
    period = 1 / spec['fsw_hz']
    inductance = spec['phase_inductance_uh'] * 1e-6
    total = spec['comparison_total_a']
    bus, sc = 38, 19
    pp = sc * period / (2 * inductance)
    valley, crest = (total - pp) / 2, (total + pp) / 2
    cases = {}
    for name, sign in (('charge', 1), ('discharge', -1)):
        # At t=0 phase A starts its rising ramp and B starts its falling ramp.
        a0, b0 = sign * total / 2 - pp / 2, sign * total / 2 + pp / 2
        current = spice(work, f'phase_{name}', f'''
Va ga 0 PULSE(0 1 0 1n 1n {period/2-1e-9} {period})
Vb gb 0 PULSE(1 0 0 1n 1n {period/2-1e-9} {period})
Ba swa 0 V={bus}*V(ga)
Bb swb 0 V={bus}*V(gb)
Vsa swa la 0
Vsb swb lb 0
La la sc {inductance} IC={a0}
Lb lb sc {inductance} IC={b0}
Vsc sc 0 {sc}
Bbus ia 0 V=V(ga)*I(Vsa)+V(gb)*I(Vsb)
Bsum total 0 V=I(Vsa)+I(Vsb)
''', '100n', '400.01u', '''meas tran a_pp PP i(Vsa) FROM=100u TO=400u
meas tran b_pp PP i(Vsb) FROM=100u TO=400u
meas tran a_mean AVG i(Vsa) FROM=100u TO=400u
meas tran b_mean AVG i(Vsb) FROM=100u TO=400u
meas tran sc_pp PP v(total) FROM=100u TO=400u
meas tran bus_mean AVG v(ia) FROM=100u TO=400u
meas tran bus_rms RMS v(ia) FROM=100u TO=400u''')
        current['bus_ac_rms_a'] = math.sqrt(max(0, current['bus_rms']**2 - current['bus_mean']**2))
        current['pass'] = (all(abs(current[key] - pp) < .15 for key in ('a_pp', 'b_pp'))
                           and all(abs(current[key] - sign * total/2) < .15 for key in ('a_mean', 'b_mean'))
                           and current['sc_pp'] < .15)
        cases[name] = current

    # Initial phase currents bound the nominal 38/19 V example. +20% L is
    # carried at those same initial currents to increase the stored energy.
    # Return 100% of both tail currents into the local worst-case C bank.
    fault_l = 1.2 * inductance
    edge = 40e-6
    tail = spice(work, 'regulator_local_tail', f'''
Cbank cap 0 {AFTER['reg_cap_f']*.8} IC=37.8
Resr bus cap {AFTER['reg_esr_ohm']}
Bfront 0 bus I={{time<{edge} ? {total} : 0}}
Ba 0 bus I={{time<{edge} ? 0 : max(0,{crest}-37.8*(time-{edge})/{fault_l})}}
Bb 0 bus I={{time<{edge} ? 0 : max(0,{valley}-37.8*(time-{edge})/{fault_l})}}
''', '50n', '100u', '''meas tran bus_peak_v MAX v(bus)
meas tran cap_peak_v MAX v(cap)''')
    tail.update(
        assumed_total_disable_us=edge*1e6, phase_initial_abs_a=[crest, valley],
        assumed_l_uh=1e6*fault_l, energy_j=fault_l*(crest**2+valley**2)/2,
        local_bus_limit_v=39.009, pass_screen=tail['bus_peak_v'] < 39.009,
        scope='理想同步关断与最慢线性尾流，40µs为既有总注入预算。未含换流寄生、单相短路或器件/驱动实际延迟。',
    )
    return {'ccm_38v_19v': cases, 'local_fault_tail': tail,
            'pass': all(case['pass'] for case in cases.values()) and tail['pass_screen']}


def braking(work, tag, parts):
    speed = 15 / 3.6
    delay, decel, mass = .050, 3.15, 30
    stop_time = delay + speed / decel
    stop_distance = speed * delay + speed**2 / (2 * decel)
    # 450 W is the existing electrical stress fixture; 30 kg * 3.15 * v = 393.75 W.
    cap = rails(parts, .8)
    esr = 1 / (2 / parts['foc_esr_ohm'] + 1 / parts['reg_esr_ohm'])
    ron = 2 * parts['reg_mos_ohm']
    rjc = .48 if tag == 'after' else .40
    result = spice(work, f'brake_{tag}', f'''
Vregen power 0 PWL(0 0 50m 0 50.001m 450 {stop_time} 0 1.5 0)
Bregen 0 bus I=V(power)/max(20,V(bus))
Rbank bus cap {esr}
Cbus cap 0 {cap} IC=37.8
Rdelay bus sense 500
Cdelay sense 0 1u IC=37.8
Sd ds drain sense 0 DUMPSW
.model DUMPSW SW(Ron=1u Roff=1e9 Vt=40.087 Vh=2.099)
Rdump bus dump_l .82
Ldump dump_l ds 1u IC=0
Dfw ds bus FREEWHEEL
.model FREEWHEEL D(Is=1u Rs=.01)
Rmos drain 0 {ron}
Vtemp case_t 0 85
Rjc tj case_t {rjc}
Cj tj case_t {{.02/{rjc}}}
Bheat 0 tj I=V(drain)^2/{ron}
Bnear near 0 V=V(bus)+.1
Bres pr 0 V=V(bus,dump_l)^2/.82
''', '10u', '1.5', '''meas tran bus_peak_v MAX v(bus)
meas tran foc_peak_v MAX v(near)
meas tran dump_current_peak_a MAX i(Ldump)
meas tran dump_junction_peak_c MAX v(tj)
meas tran dump_energy_j INTEG v(pr) FROM=0 TO=1.5''')
    result.update(cap_f=cap, esr_ohm=esr, stop_time_s=stop_time,
                  stop_distance_m=stop_distance, initial_mechanical_w=mass*decel*speed,
                  ideal_motor_brake_torque_each_nm=mass*decel*.03/(2*4),
                  cap_headroom_v=parts['cap_rating_v']-result['bus_peak_v'],
                  foc_threshold_margin_v=42.533-result['foc_peak_v'],
                  thermal_fixed_case_bound_c=85+rjc*ron*result['dump_current_peak_a']**2)
    result['pass'] = (result['foc_threshold_margin_v'] > 0 and result['cap_headroom_v'] > 0
                      and result['thermal_fixed_case_bound_c'] < 175 and stop_distance <= 3)
    return result


def precharge(work, tag, parts):
    # Preserve non-bank capacitance from BMS high-corner 79.787 mF.
    c_high = .079787 + 1.2*(parts['reg_cap_f'] - BEFORE['reg_cap_f'])
    r_slow = 9.41965
    leakage = .045736 if tag == 'before' else (6*.002381 + 6*.000630 + .003736)
    # Existing pump 5 V stays enabled. Local load is added after the measured-to-be 8 V knee.
    esr = 1 / (2 / parts['foc_esr_ohm'] + 1 / parts['reg_esr_ohm'])
    cases = {}
    for name, pump_w in [('inherited_without_pump_aux', 0), ('pump_aux_typical', .65)]:
        # A crossing beyond the transient is reported as absent, with the deadline voltage retained.
        final_delta = leakage*r_slow + pump_w*r_slow/37.2412
        crossing = '\nmeas tran charge_s WHEN v(bus)=37.2412 RISE=1' if final_delta < .5588 else ''
        values = spice(work, f'pre_{tag}_{name}', f'''
Vpack pack 0 37.8
Rpre pack bus {r_slow}
Resr bus cap {esr}
Cbus cap 0 {c_high} IC=0
Bload bus 0 I={leakage}*tanh(V(bus)/.1)+({pump_w}/max(8,V(bus)))*(.5+.5*tanh((V(bus)-8)/.3))
Bd diff 0 V=v(pack)-v(bus)
''', '500u', '7', 'meas tran delta_at_deadline_v FIND v(diff) AT=4.498' + crossing)
        charge = values.get('charge_s')
        values['charge_s'] = charge
        values['total_s'] = charge + .355 + 4.1 if charge is not None else None
        values['timer_margin_s'] = 8.953 - values['total_s'] if charge is not None else None
        values['pass'] = values['delta_at_deadline_v'] <= .5588
        cases[name] = values
    probe = spice(work, f'probe_{tag}', f'''
Cb bus 0 {c_high} IC=22.362
Vce ce 0 2
Rprobe bus ce 844.6
Bd diff 0 V=22.362-v(bus)
''', '200u', '4.3', '''meas tran detect_s WHEN v(bus)=21.2291 FALL=1
meas tran drop_at_4p1_v FIND v(diff) AT=4.1''')
    short = spice(work, f'preshort_{tag}', '''
Vpack pack 0 37.8
Rpre pack 0 7.284
Bpower p 0 V=v(pack)^2/7.284
''', '1m', '10.227', '''meas tran peak_w MAX v(p) FROM=0 TO=10.226
meas tran energy_j INTEG v(p) FROM=0 TO=10.226''')
    short['rx24_mount_and_pulse_qualification'] = '待核验，300 W 须按指定散热安装'
    return dict(c_high_f=c_high, assumed_pump_aux_w=.65, leakage_a=leakage, slow=cases,
                probe=probe, short=short,
                pass_timing=cases['pump_aux_typical']['pass'] and probe['detect_s'] < 4.1)


def hold(work):
    cases = {}
    for label, cap in [('retained_10uf_screen', '10u'), ('retained_30p08uf', '30.08u')]:
        result = spice(work, 'rail_' + label, f'''
Vprot prot 0 PWL(0 0 100u 0 101u 13.2 5m 13.2 5.01m 11.4 10m 11.4
+ 10.001m 0 12m 0 12.001m 11.4 15m 11.4 15.001m 0 39m 0)
Dhold prot src SCH
.model SCH D(IS=2e-5 N=1.08 RS=.15 BV=40 IBV=1m CJO=110p)
R1 src mid 9.9
R2 mid hw 9.9
Resr hw cap .08
Cb cap 0 {cap} IC=0
Bload hw 0 I=.016*tanh(V(hw)/.5)
Rtop hw sense 100k
Rbot sense 0 10k
''', '2u', '39m', '''meas tran steady_v FIND v(hw) AT=9.9m
meas tran dip_min_v MIN v(hw) FROM=10m TO=12m
meas tran recovered_v FIND v(hw) AT=14.9m
meas tran t_deep_s WHEN v(hw)=6.938 FALL=LAST
meas tran t_uvlo_s WHEN v(hw)=2.7 FALL=LAST
meas tran t_sense_recover_s WHEN v(sense)=0.83197 RISE=LAST''')
        result['ctr_available_us'] = (result['t_uvlo_s'] - result['t_deep_s']) * 1e6
        result['ctr_margin_us'] = result['ctr_available_us'] - 211.5
        result['crosses_latest_falling_threshold_during_dip'] = result['dip_min_v'] < 8.9285
        if result['t_sense_recover_s'] < .012:
            result['t_sense_recover_s'] = None
        result['pass'] = (result['dip_min_v'] > 7 and result['recovered_v'] > 9.1517
                          and result['ctr_margin_us'] > 0)
        cases[label] = result
    return cases


def island(work, tag, parts):
    result = spice(work, f'island_{tag}', f'''
Vinject p 0 PWL(0 500 1m 500 1.001m 0 2m 0)
Brein 0 bus I=V(p)/max(20,V(bus))
Rbank bus cb {parts['foc_esr_ohm']}
Cb cb 0 {parts['foc_cap_f']*.8} IC=37.8
Riso 0 can_reference 1e12
Ciso 0 can_reference 3p IC=0
Vremote can_reference 0 37.8
''', '1u', '2m', '''meas tran peak_v MAX v(bus)
meas tran energy_j INTEG v(p) FROM=0 TO=1.001m
meas tran modeled_dc_return_a FIND i(Vremote) AT=1.2m''')
    result['headroom_v'] = parts['foc_rating_v'] - result['peak_v']
    result['pass'] = result['peak_v'] < 42.533 and result['headroom_v'] > 0
    return result


def protection(work, tag, parts):
    balance = 2*.0028/4
    rsource = .015
    shunt = .0005
    start_load = 37.8/130 - rsource - balance - shunt
    cases = {}
    for name, fault_a, threshold, delay in [('ocd', 180, .08, .0099), ('scd', 250, .10, .0001)]:
        fault_load = 37.8/fault_a - rsource - balance - shunt
        values = spice(work, f'{name}_{tag}', f'''
Vpack pack 0 37.8
Rsource pack in {rsource}
Smain in fet permit 0 MAINSW
.model MAINSW SW(Ron=1u Roff=1e9)
Rmos fet bus {balance}
Bload bus sh I=v(bus,sh)*ternary_fcn(v(cmd)>.5,{1/fault_load},{1/start_load})
Rsh sh 0 {shunt}
Vfault cmd 0 PWL(0 0 2m 0 2.001m 1)
Btimer 0 timer I=ternary_fcn(v(sh)>{threshold},1m,0)
Ctimer timer 0 {delay*.002} IC=0
Btrip permit 0 V=ternary_fcn(v(timer)>.5,0,1)
Bd diff 0 V=v(fet)-v(bus)
Bh heat 0 V=v(diff)^2/{balance}
''', '1u', '15m', f'''meas tran start_shunt_v FIND v(sh) AT=1m
meas tran peak_shunt_v MAX v(sh)
meas tran threshold_at_s WHEN v(sh)={threshold} RISE=1
meas tran break_at_s WHEN v(permit)=.5 FALL=1
meas tran start_mos_drop_v FIND v(diff) AT=1m
meas tran fault_mos_energy_j INTEG v(heat) FROM=2.001m TO=15m
meas tran settled_current_a FIND i(Vpack) AT=14m''')
        values['fixture_break_delay_s'] = values['break_at_s'] - values['threshold_at_s']
        values['settled_current_a'] = abs(values['settled_current_a'])
        values['pass'] = (values['peak_shunt_v'] > threshold and values['settled_current_a'] < .001
                          and abs(values['fixture_break_delay_s'] - delay) < 3e-6)
        cases[name] = values
    result = {'main_ocd_and_scd_fixtures': cases}
    # Phase RMS and bus current are different. A 65 A per-bridge phase peak is a comparison input.
    result.update(bms_130a_cold_w=130**2*balance, bms_130a_hot_screen_w=2*130**2*balance,
                  bms_130a_shunt_w=130**2*.0005,
                  foc_phase_peak_a=65, foc_six_switch_pwm_conduction_w=3*(65/math.sqrt(2))**2*parts['foc_mos_ohm'],
                  foc_instantaneous_two_switch_w=2*65**2*parts['foc_mos_ohm'],
                  foc_12_gate_driver_w=12*parts['foc_qg_nc']*1e-9*10*20_000,
                  foc_92p4a_phase_peak_conduction_w=3*92.4**2/2*parts['foc_mos_ohm'])
    result['pass_thresholds'] = 130 < 142.2 and all(case['pass'] for case in cases.values())
    return result


def tap_fuse(work, contract):
    fuse = contract['localization_v1']['tap_fuse']
    terminals = next(interface['pins'] for interface in contract['drawing_components']
                     if interface['ref'] == 'J_CELL_SERVICE')
    source_assembly = fuse.get('source_assembly_requirements', {})
    source_placement_ok = (
        fuse['placement'] == 'off-board'
        and fuse['source_installation'] == 'each_fuse_before_harness_at_its_cell_tap'
        and fuse['b0'] == 'direct_unfused'
        and fuse['quantity'] == len(fuse['refs_and_nets']) == 9
    )
    interface_ok = (
        terminals['1'] == 'B0'
        and all(fuse['refs_and_nets'].get(f'F_CELL_B{tap}') ==
                {'1': f'CELL_B{tap}_RAW', '2': terminals[str(tap + 1)]}
                and terminals[str(tap + 1)] == f'TAP_B{tap}' for tap in range(1, 10))
    )
    topology_ok = source_placement_ok and interface_ok
    screen = fuse['fault_screen']
    voltage = screen['full_pack_v']
    cell_count = screen['series']
    parallel = screen['parallel']
    # EVE's 1 kHz AC upper limit is not a DC lower bound.
    measured_minimum = screen['verified_minimum_cell_dc_ohm']
    cell_screen_ohm = screen['screening_cell_dc_ohm'] if measured_minimum is None else measured_minimum
    wire_temp_c = screen['copper_temp_c']
    rho_copper = .01724 * (1 + .00393 * (wire_temp_c - 20))
    harness_section_mm2 = screen['wire_section_mm2']
    distal_loop_m = screen['board_end_loop_m']
    harness_ohm = rho_copper * distal_loop_m / harness_section_mm2
    fuse_ohm = fuse['typical_cold_resistance_ohm']
    prearc_i2t = fuse['typical_prearc_i2t_a2s']
    interrupt_a = fuse['interrupt_dc_a']
    cases = {}
    for label, loop_ohm in [('board_end_screen', harness_ohm), ('source_end_screen', 0.0)]:
        taps = []
        for series in range(1, cell_count + 1):
            tap_v = voltage * series / cell_count
            pack_ohm = series * cell_screen_ohm / parallel
            prospective_a = tap_v / (pack_ohm + loop_ohm)
            cold_fuse_a = tap_v / (pack_ohm + loop_ohm + fuse_ohm)
            taps.append(dict(series=series, voltage_v=tap_v, pack_ohm=pack_ohm,
                             prospective_a=prospective_a, cold_fuse_current_a=cold_fuse_a))
        high = taps[-1]
        # Prospective current excludes the fuse. The melting screen includes its
        # typical cold resistance and stops before arcing.
        spice_values = spice(work, label, f'''
Vcell cell 0 {voltage}
Rloop cell in {high['pack_ohm'] + loop_ohm}
Vmeter in fuse 0
Rfuse fuse 0 {fuse_ohm}
Bi2 0 joule I=i(Vmeter)^2
Ci2 joule 0 1 IC=0
''', '0.05u', '0.5m', f'''meas tran cold_current_a FIND i(Vmeter) AT=10u
meas tran typical_prearc_s WHEN v(joule)={prearc_i2t} RISE=1''')
        cases[label] = dict(
            harness_loop_ohm=loop_ohm, taps=taps,
            max_prospective_a=max(tap['prospective_a'] for tap in taps),
            below_interrupt_rating=all(tap['prospective_a'] < interrupt_a for tap in taps),
            typical_wire_prearc_energy_j=prearc_i2t * loop_ohm,
            typical_fuse_prearc_energy_j=prearc_i2t * fuse_ohm,
            agrees_with_ohmic_model=(
                math.isclose(spice_values['cold_current_a'], high['cold_fuse_current_a'], rel_tol=1e-5)
                and math.isclose(spice_values['typical_prearc_s'],
                                 prearc_i2t / high['cold_fuse_current_a']**2, rel_tol=1e-4)
            ),
            **spice_values,
        )

    board_end = cases['board_end_screen']
    source_end = cases['source_end_screen']
    holder_dc_v = source_assembly.get('holder_rated_dc_v')
    holder_rating_verified = (
        isinstance(holder_dc_v, (int, float)) and not isinstance(holder_dc_v, bool)
        and holder_dc_v >= voltage
    )
    assembly_description_complete = (
        source_assembly.get('double_insulated_short_input') is True
        and source_assembly.get('short_shielded_entry') is True
        and source_assembly.get('covers_entire_tap_harness') is True
        and holder_rating_verified
        and source_assembly.get('bench_required') is True
    )
    required_bench_checks = [
        'Measure 9S5P minimum dynamic resistance over the relevant fault pulse, including minimum cable and return-path resistance.',
        'Measure fuse-holder and tap-harness temperature during the fault and recovery tests.',
        'Verify total clearing I²t, B0 return path, holder DC rating, and physical fuse-to-hub fault coverage.',
        'Qualify the AWG22 XH crimp pigtail and splices under the existing 5 A fuse total clearing I²t. '
        'The 0.5 mm² harness screen omits those local parts and contact heating.',
    ]
    source_assembly_result = dict(
        double_insulated_short_input=source_assembly.get('double_insulated_short_input'),
        short_shielded_entry=source_assembly.get('short_shielded_entry'),
        covers_entire_tap_harness=source_assembly.get('covers_entire_tap_harness'),
        holder_rated_dc_v=holder_dc_v,
        holder_rating_verified_for_pack=holder_rating_verified,
        bench_required=source_assembly.get('bench_required'),
        required_bench_checks=required_bench_checks,
        description_complete=assembly_description_complete,
    )
    sensitivity = []
    for cell_mohm in (8, 12, 16, 18, 21, 24):
        pack_ohm = cell_count * cell_mohm / (1000 * parallel)
        sensitivity.append(dict(cell_dc_mohm=cell_mohm,
                                source_end_a=voltage / pack_ohm,
                                board_end_a=voltage / (pack_ohm + harness_ohm)))
    return dict(
        before='Eaton PV-5A10F + CHPV1U', withdrawn='2410FA-5A C2838938, 50 A DC',
        after=fuse['mpn'], dc_interrupt_a=interrupt_a, dc_rating_v=fuse['rated_dc_v'],
        model_inputs=dict(series=cell_count, parallel=parallel, cell_screen_ohm=cell_screen_ohm,
                          verified_minimum_cell_dc_ohm=measured_minimum, copper_temp_c=wire_temp_c,
                          wire_section_mm2=harness_section_mm2, board_end_loop_m=distal_loop_m,
                          minimum_loop_ohm_exclusive=voltage / interrupt_a,
                          minimum_cell_ohm_exclusive_at_near_zero_cable=voltage / interrupt_a * parallel / cell_count),
        cases=cases,
        board_end_harness_fault=dict(
            prospective_current_a=board_end['max_prospective_a'],
            interrupt_rating_a=interrupt_a,
            below_interrupt_rating=board_end['below_interrupt_rating'],
            typical_prearc_time_s=board_end['typical_prearc_s'],
            typical_prearc_time_us=board_end['typical_prearc_s'] * 1e6,
            typical_wire_prearc_energy_j=board_end['typical_wire_prearc_energy_j'],
            typical_fuse_prearc_energy_j=board_end['typical_fuse_prearc_energy_j'],
            model_agrees_with_ohmic_screen=board_end['agrees_with_ohmic_model'],
            basis='Prospective current is before fuse resistance; pre-arc time uses typical cold fuse resistance and typical I²t.'
        ),
        source_topology_pass=topology_ok,
        source_placement_pass=source_placement_ok,
        source_assembly_requirements=source_assembly_result,
        bench_and_assembly_audit=dict(
            near_source_ideal_prospective_a=source_end['max_prospective_a'],
            interrupt_rating_a=interrupt_a,
            exceeds_interrupt_rating=source_end['max_prospective_a'] > interrupt_a,
            required=True,
            disposition='Bench and assembly audit required; ideal near-source current above the published interrupt rating is not cleared by the harness screen.',
            checks=required_bench_checks,
        ),
        fuse_to_hub_fault_coverage=dict(
            status='not_proven',
            note='Physical routing, fault location, and return path from source fuse to hub are not established by this resistance screen.'
        ),
        cell_dc_resistance_sensitivity=sensitivity,
        typical_prearc_i2t_a2s=prearc_i2t,
        published_opening_time_max_s=fuse['published_opening_time_max_s'],
        pass_pack_tap_prospective_and_source_placement=(
            source_placement_ok
            and board_end['below_interrupt_rating']
            and board_end['agrees_with_ohmic_model']
            and assembly_description_complete
        ),
        physical_fuse_validation='pending; this is a screening model, not proof of fuse interruption or manufacturing pass.',
        scope='B1…B9对低位抽头的等效串数扫描。板端0.8 m、单芯0.5 mm²及12 mΩ筛查输入给出板端/线束故障模型；近源理想场景取零导线电阻并单列台架与装配审查。保险至hub的实际故障位置、回流路径和覆盖范围，最小动态电阻、总分断I²t、保险座/线束温升及双重绝缘与短屏蔽入口均待实物核验；不据此宣称保险已证实或制造通过。',
    )


def pump_path(work, tag, fault):
    # Existing fixture R/L. A filtered shunt threshold sets a one-way COMP/BREAK latch.
    # An assumed decaying EMF drives reverse brake current after normal stop.
    old = tag == 'before'
    ron, brake_ohm = (.05, .05) if old else (.085, .060)
    shunt = 1e-6 if old else .020
    ilim = 5.19 if old else 5.0
    if fault:
        raw_sense = '0.02*i(Lw)' if old else 'v(sns)'
        control = f'''
Bsense raw 0 V={raw_sense}
Rfilter raw sense 1k
Cfilter sense 0 1n IC=0
Btrip 0 latched I=ternary_fcn(v(sense)>{ilim*.02},1m,0)
Clatch latched 0 2n IC=0
Benable enable 0 V=ternary_fcn(v(latched)>.5,0,1)
'''
        mode_measures = 'meas tran command_off_s WHEN v(enable)=.5 FALL=1'
    else:
        control = 'Venable enable 0 PWL(0 1 1m 1 1.0001m 0)'
        mode_measures = '''meas tran command_off_s WHEN v(enable)=.5 FALL=1
meas tran motor_avg_v AVG v(motor) FROM=.5m TO=1m'''
    emfs = '0' if fault else 'PWL(0 18 1m 18 5m 16.62 20m 12.31 80m 3.70 100m 2.49)'
    brake = '0' if fault else 'PWL(0 0 1.0999m 0 1.1m 1)'
    # Existing H-bridge uses synchronous high-side and body diodes. New bridge uses same
    # terminal reference on positive and an explicit Schottky across the motor.
    loop = ('Dlo 0 apos BODY\nDhi apos bus BODY\n'
            f'Brh high 0 V={"v(enable)" if fault else "1"}\n'
            'Sha bus apos high 0 HIGH\n'
            '.model HIGH SW(Ron=.0001 Roff=1e8)') if old else 'Rlink bus apos 1u'
    diode = 'BODY' if old else 'SCH'
    pwm = 'Vpwm pwm 0 1' if fault else 'Vpwm pwm 0 PULSE(1 0 28u 100n 100n 21.8u 50u)'
    got = spice(work, f'pump_{tag}_{fault}', f'''
Vbus bat 0 37.8
Rsvc bat bus .025
Cbus bus 0 503u IC=37.8
Bi draw 0 V=abs(i(Vbus))
Bm motor 0 V=v(apos,mn)
{control}
{pwm}
Brun run 0 V=V(enable)*V(pwm)
Vbrk brk 0 {brake}
{loop}
Srun mn sns run 0 NCH
Rsh sns 0 {shunt}
Sbrk mn bus brk 0 PCH
Dpbody mn bus BODY
Lloop mn fw 50n IC=0
Dfw fw bus {diode}
Dlow sns mn BODY
Coss mn sns 300p
Rw apos w 1
Lw w e 500u IC={0 if fault else 3}
Vemf e mn {emfs}
.model SCH D(Is=4u Rs=.085 N=1.25 Cjo=250p)
.model BODY D(Is=100n Rs=.03 Cjo=300p)
.model NCH SW(Ron={ron} Roff=1e8)
.model PCH SW(Ron={brake_ohm} Roff=1e8)
''', '.15u' if fault else '.5u', '1m' if fault else '30m', f'''meas tran peak_i_a MAX i(Lw)
meas tran min_i_a MIN i(Lw)
meas tran switch_node_peak_v MAX v(mn)
meas tran bus_peak_v MAX v(bus)
meas tran bus_current_after_a MAX v(draw) FROM={.0008 if fault else .010} TO={.00099 if fault else .029}
{mode_measures}''')
    if fault:
        got['limit_a'] = ilim
        got['pass'] = (got['peak_i_a'] < 5.5 and got['bus_current_after_a'] < .05
                       and got['switch_node_peak_v'] < (70 if old else 100))
    else:
        got['pass'] = got['min_i_a'] < 0 and got['switch_node_peak_v'] < (70 if old else 100)
    return got


def shed(work, tag, parts):
    cap = rails(parts, .8)
    # First run the hot ADC gain corner with the pump drawing full power until trip.
    detect = spice(work, 'detect_' + tag, f'''
Cb bus 0 {cap} IC=22.5
Rfilter bus sense 15000
Cfilter sense 0 100n IC=22.5
Breport adc 0 V=1.01*V(sense)
Bload bus 0 I=75/max(12,V(bus))
''', '10u', '20m', 'meas tran sense_trip_s WHEN v(adc)=22.2 FALL=1')
    tail = .005 if tag == 'before' else .003
    deadline = detect['sense_trip_s'] + .0015 + tail
    # Existing 0.5 ms each for sample, decision and GPIO. Qualification targets a shorter tail.
    result = spice(work, 'shed_' + tag, f'''
Cb bus 0 {cap} IC=22.5
Vdeadline cmd 0 PWL(0 1 {deadline} 1 {deadline+.00001} 0 110m 0)
Bload bus 0 I=(10+65*V(cmd))/max(12,V(bus))
''', '10u', '110m', '''meas tran v_50ms FIND v(bus) AT=50m
meas tran v_100ms FIND v(bus) AT=100m
meas tran shed_s WHEN v(cmd)=.5 FALL=1''')
    result.update(detect, power_tail_budget_s=tail)
    result['foc_dq_margin_at_50ms_v'] = result['v_50ms'] - 21.10762
    result['foc_18v_margin_at_100ms_v'] = result['v_100ms'] - 18
    result['meets_15ms_target'] = result['shed_s'] < .015
    result['pass'] = (result['shed_s'] < .05 and result['foc_dq_margin_at_50ms_v'] > 0
                      and result['v_100ms'] > 18)
    return result


def main():
    inputs = [
        '稳压模块/检查/闭合候选_P5Rev2/稳压P5Rev2施工合同.json',
        'BMS模块/检查/闭合候选_9S5P/BMS_9S5P施工合同.json',
        'FOC驱动模块/检查/闭合候选_164Rev2/FOC_164Rev2施工合同.json',
        '水泵驱动模块/检查/闭合候选_隔离CAN/水泵闭合施工合同.json',
    ]
    contracts = [json.loads((POWER / name).read_text(encoding='utf-8')) for name in inputs]
    with TemporaryDirectory(prefix='vehicle_power_') as temporary:
        work = Path(temporary)
        scenarios = {}
        comparisons = (('before', BEFORE), ('after', AFTER))
        for tag, parts in comparisons:
            scenarios[tag] = {
                '1_braking': braking(work, tag, parts),
                '2_precharge': precharge(work, tag, parts),
                '4_island': island(work, tag, parts),
                '5_protection': protection(work, tag, parts),
                '6_pump': {'stall': pump_path(work, tag, True), 'normal_brake': pump_path(work, tag, False),
                           'shed': shed(work, tag, parts)},
            }
        hold_cases = hold(work)
        fuse_cases = tap_fuse(work, contracts[1])
        phase_cases = regulator_two_phase(work, contracts[0])
    after = scenarios['after']
    result = {
        'status': 'BEHAVIORAL_MODEL_ONLY',
        'engine': {'path': SPICE.relative_to(POWER.parent).as_posix(),
                   'sha256': sha256(SPICE.read_bytes()).hexdigest()},
        'script_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
        'contract_sha256': {name: sha256((POWER / name).read_bytes()).hexdigest() for name in inputs},
        'parts_for_models': {'before': BEFORE, 'after': AFTER},
        'comparison': scenarios,
        '3_hold_unchanged_before_and_after': hold_cases,
        '3_regulator_two_phase': phase_cases,
        '5_tap_fuse': fuse_cases,
        'conclusions': {
            '1': 'PASS' if after['1_braking']['pass'] else 'FAIL',
            '2': 'PASS' if after['2_precharge']['pass_timing'] else 'FAIL',
            '3': 'PASS' if (all(c['pass'] for c in hold_cases.values()) and phase_cases['pass']) else 'FAIL',
            '4': 'PASS' if after['4_island']['pass'] else 'FAIL',
            '5': 'PASS' if (after['5_protection']['pass_thresholds']
                            and fuse_cases['pass_pack_tap_prospective_and_source_placement']) else 'FAIL',
            '6': 'PASS' if all(c['pass'] for c in after['6_pump'].values()) else 'FAIL',
        },
        'scope': {
            '1': '断开动力主路，假设电池侧早保护仍有效。30 kg 平地、50 ms 建扭、净 450 W 峰值线性衰减、近桥+0.1 V、0.5 ms 判定滞后、1 µH 泄放线。停距基于给定减速度，需轮地/电机扭矩台架证实。85°C 固定壳温、20 ms 热时间常数为示意，固定壳温下另列 RθJC 上界。',
            '2': '预充 +20% 初始容量，额定电压漏电作恒流筛查，0.65 W 泵辅助取电为假设。理想硬短路用继承热角及最长定时，RX24 安装及连续/脉冲能力待核。',
            '3': '既有 100k/10k 门限及 211.5 µs CTR 预算。2 ms 短断与永久短路分开。另验双相理想开关两方向纹波和本地电容故障续流。40µs、80A是压力输入，实物止流和额定功率待核。',
            '4': '初始 37.8 V、每侧容量−20%、500 W→1 ms 归零。CAN 隔离屏障用假设电容和电阻表示；仅排除结构性直流回路，示波核对其他线束及共模瞬态。',
            '5': '以37.8V和电阻负载由130A阶跃到180/250A，使用行为阈值与定时。抽头门禁覆盖板端线束模型及源端装配合同，1750A近源理想短路另列装配与台架审计，不代表分断已证实。',
            '6': '泵用 1 Ω、0.5 mH 旧夹具，正常运行 20 kHz PWM。卸载用 22.2 V 校准阈值、+1% 增益、1.5 ms RC，从稳态 22.5 V 起算模型 crossing 后加 1.5 ms 控制及旧5/新3 ms功率尾段预算，按 21.10762 V/50 ms 与 18 V/100 ms 筛查。COMP 采用1µs输入RC及1µs行为级BREAK锁存，旧芯片用等效单次响应对照。实泵堵转、续流寄生、制动电流、开关实际时序及 FOC 建扭待核。',
        },
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(result['conclusions']))
    return 0 if all(status == 'PASS' for status in result['conclusions'].values()) else 1


if __name__ == '__main__':
    sys.exit(main())
