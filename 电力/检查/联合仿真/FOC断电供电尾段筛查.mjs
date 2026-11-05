import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';

const root = new URL('../../../', import.meta.url);
const netlistPath = new URL('电力/FOC驱动模块/eda工程文件/导出/当前交付/FOC_G431_P1.enet', root);
const outPath = new URL('./FOC断电供电尾段筛查.json', import.meta.url);
const bytes = await readFile(netlistPath);
const net = JSON.parse(bytes.toString('utf8').replace(/^\uFEFF/, ''));
const comps = Object.values(net.components);
const byRef = Object.fromEntries(comps.map(c => [c.props.Designator, c]));
const pins = ref => Object.fromEntries(Object.values(byRef[ref].pinInfoMap).map(p => [p.name, p.net]));
const val = ref => byRef[ref]?.props?.Value ?? null;
const requireNets = (ref, expected) => {
  const actual = pins(ref);
  for (const [pin, expectedNet] of Object.entries(expected)) {
    if (actual[pin] !== expectedNet) throw new Error(`${ref}.${pin}: expected ${expectedNet}, got ${actual[pin]}`);
  }
};

// Native-source topology anchors: stop if the official export changes.
const u1Nets = Object.values(byRef.U1.pinInfoMap).map(p => p.net);
for (const n of ['BUS', 'DVDD_5V', 'BUCK_RT_SD', 'BUCK_SW', 'BUCK_FB']) if (!u1Nets.includes(n)) throw new Error(`U1 missing expected native net ${n}`);
requireNets('U5', { VCC: '5V_BUCK', VIO: 'MCU_3V3' });
requireNets('U2', { IN: 'BUS', OUT: 'SAFE_5V' });
requireNets('U6', { IN: 'BUS', OUT: 'DUMP_5V' });
requireNets('U7', { VDD: 'DUMP_5V', OUTB: 'BUSOV_N' });
requireNets('Q_BUCK_CLAMP', { B: 'BUCK_CLAMP_BASE', E: 'GND_SYS', C: 'BUCK_RT_SD' });
requireNets('Q_BUCK_RELEASE', { B: 'BUCK_RELEASE_BASE', E: 'GND_SYS', C: 'BUCK_CLAMP_BASE' });
requireNets('R_BUCK_VALID_BASE', { 1: 'BUCK_RELEASE_BASE', 2: 'LOAD_POWER_VALID' });
requireNets('R_BUCK_CLAMP_BUS', { 1: 'BUS', 2: 'BUCK_CLAMP_BASE' });

const namedCap = (ref, expectedNet) => {
  const c = byRef[ref];
  if (!c) throw new Error(`Missing ${ref}`);
  const ns = Object.values(c.pinInfoMap).map(p => p.net);
  if (!ns.includes(expectedNet)) throw new Error(`${ref} not on ${expectedNet}: ${ns}`);
  const s = String(val(ref)).toLowerCase().replaceAll(' ', '');
  const m = s.match(/([0-9.]+)(uf|nf|pf)/);
  if (!m) return { ref, value: val(ref), capacitance_f: null, note: 'value not machine-parsable; check source annotation' };
  const scale = { uf: 1e-6, nf: 1e-9, pf: 1e-12 }[m[2]];
  return { ref, value: val(ref), capacitance_f: Number(m[1]) * scale };
};
const cap5 = ['U1_BUCK_COUT1','U1_BUCK_COUT2','C_HALL_VCC1','C_HALL_VCC2','U5_C'].map(r => namedCap(r, r === 'U1_BUCK_COUT1' ? 'BUCK_RIPPLE_CAP' : '5V_BUCK'));
const capSafe = ['U2_COUT'].map(r => namedCap(r, 'SAFE_5V'));
const capDump = ['U6_COUT','U3_C1','U3_C2','C_U7'].map(r => namedCap(r, 'DUMP_5V'));
const total = xs => xs.reduce((a, x) => a + (x.capacitance_f ?? 0), 0);
// R_BUCK_RRIP (0.82 ohm) separates the 22 uF ripple capacitor from the 5V rail.
const c5Direct = total(cap5.filter(x => x.ref !== 'U1_BUCK_COUT1'));
const c5Reservoir = total(cap5.filter(x => x.ref === 'U1_BUCK_COUT1'));
const c5EffectiveUpper = c5Direct + c5Reservoir;
const eBetween = (c, hi, lo) => 0.5 * c * (hi ** 2 - lo ** 2);
const loadCases = [
  { name: 'normal budget (existing design note)', current_a: 0.240, basis: '5V buck load budget: 60mA core board equivalent + 80mA CAN + 50mA Hall + 50mA margin; estimate, not measured' },
  { name: 'CAN dominant budget (existing design note)', current_a: 0.270, basis: 'existing design note estimate' },
  { name: 'transient model peak stress only (not continuous load)', current_a: 0.366554, basis: 'existing model inductor peak; stress case only, not 5V continuous load or steady hold budget' },
];
const holdPower = [0.1, 1, 3, 10].map(seconds => ({
  hold_s: seconds,
  load_cases: loadCases.map(l => ({ name: l.name, current_a: l.current_a, energy_j_at_5v: 5 * l.current_a * seconds,
    equivalent_capacitance_f_for_5v_to_4v5: (5 * l.current_a * seconds) / eBetween(1, 5, 4.5) }))
}));
const capHold = loadCases.map(l => ({ name: l.name, current_a: l.current_a,
  conservative_direct_5v_capacitance_f: c5Direct,
  direct_cap_energy_5v_to_4v5_j: eBetween(c5Direct, 5, 4.5),
  ideal_hold_to_4v5_s_direct_only: eBetween(c5Direct, 5, 4.5) / (5 * l.current_a),
  optimistic_with_ripple_capacitance_f: c5EffectiveUpper,
  ideal_hold_to_4v5_s_instant_parallel_upper_bound: eBetween(c5EffectiveUpper, 5, 4.5) / (5 * l.current_a),
  caveat: '5V rail loss threshold is not measured; 22uF cap is behind 0.82ohm ripple resistor, so parallel-cap result is an optimistic upper bound. Capacitor derating, ESR, regulator dynamics, MCU 3V3 core-board input/load, Hall sink current, and 5V startup/load transient are not modeled.'
}));
const lpvLoadTime = [0, 0.005, 0.02, 0.05, 0.1, 1].map(t => ({
  input_lpv_loss_delay_after_brake_request_s: t,
  boards_expected_to_receive_direct_native_RT_SD_clamp_on_signal_loss: ['A','B'],
  rt_sd_delay_s: null,
  result: 'parameter only: no actual input event timing or netlist-backed threshold/propagation guarantee is assumed'
}));
const result = {
  qualification: 'HOLD',
  schema: 'agtc-foc-local-power-tail-screen-v1',
  purpose: 'Native-netlist anchored repeatable screen for dual FOC local-rail loss/hold through a parameterized parking tail; this is not gate-drive or vehicle stop validation.',
  source: { path: '电力/FOC驱动模块/eda工程文件/导出/当前交付/FOC_G431_P1.enet', sha256: createHash('sha256').update(bytes).digest('hex') },
  verified_native_paths: {
    drv8353rs: { ref: 'U1', VM: 'BUS', '5v_buck_output': '5V_BUCK', buck_architecture: 'TI DRV835x official datasheet identifies the integrated buck as LM5008A; internal packaged implementation is not interchangeable with standalone LM5008A external specifications without qualification', integrated_buck_inductor: val('U1_BUCK_L'), input_bus_capacitors: [val('U1_BUCK_CIN1'),val('U1_BUCK_CIN2')], output_feedback: [val('U1_BUCK_RFB1'),val('U1_BUCK_RFB2')], series_ripple_resistor: val('U1_BUCK_RRIP'), rcl: val('U1_BUCK_RCL') },
    local_rails: { '5V_BUCK': 'DRV8353RS integrated Buck from BUS; its output currently disabled by native RT/SD clamp when LOAD_POWER_VALID path requests clamp', 'MCU_3V3': 'WeAct core-board rail; native board provides 100nF CAN VIO decoupling only, core-board capacitance and load unknown', 'SAFE_5V': 'BUS -> U2 TPS7A4001 -> SAFE_5V -> MCU ADC/COMP divider; no independent stored-energy source', 'DUMP_5V': 'BUS -> U6 TPS7A4001 -> comparator rails U3/U7 and BOV chain; no independent stored-energy source' },
    hall: { connector_supply: '5V_BUCK', signals: 'HALL_A/B/C have direct and divider paths; pullups to MCU_3V3; Hall device current and whether its VCC is powered during tail are not established' },
    bms_loss_trigger: { signal: 'LOAD_POWER_VALID', path: 'R_BUCK_VALID_BASE -> Q_BUCK_RELEASE -> Q_BUCK_CLAMP -> BUCK_RT_SD', 'actual_physical_logic_and_timing': 'native topology verified; delivered waveform, voltage threshold, optocoupler transfer, transistor storage, and system loss ordering are not measured; no GPIO or firmware trigger assumed' },
    bov: { path: 'BUS -> U6 DUMP_5V -> U7 BUSOV_N -> Q_BOV_PNP/Q_BOV_NPN -> BREAK_N', meaning: 'BOV commands hardware Break; it does not keep 5V_BUCK/MCU_3V3 alive and does not provide absorption power' },
    candidate_pc6: { status: 'not connected in current native netlist', evidence: 'J_MCU_L pin 9 has empty net; historical footprint evidence maps pin 9 to PC6. Candidate only pending exact WeAct core-board revision/pinout and isolated-use verification. No firmware or wiring is presumed.' }
  },
  capacitor_inventory: { '5V_BUCK_direct': cap5.filter(x => x.ref !== 'U1_BUCK_COUT1'), '5V_BUCK_ripple_node_behind_0p82ohm': cap5.filter(x => x.ref === 'U1_BUCK_COUT1'), 'SAFE_5V': capSafe, 'DUMP_5V': capDump, '5V_direct_total_f': c5Direct, '5V_optimistic_total_f': c5EffectiveUpper, 'SAFE_total_f': total(capSafe), 'DUMP_total_f': total(capDump) },
  load_inputs: loadCases,
  cap_only_hold_estimates: capHold,
  hold_energy_to_supply_5v_for_parking_tail: holdPower,
  lpv_loss_delay_sweep: lpvLoadTime,
  parking_tail_interface: {
    consumes: 'per-board BUS(t), LPV(t), 5V_BUCK(t), MCU_3V3(t), SAFE_5V(t), DUMP_5V(t), and actual standstill time from parent coupled vehicle model; feed parent BUS trace/stop time when available',
    present_data_limit: 'existing vehicle co-simulation reports aggregate rows over a 5 s horizon but does not export BUS(t), local rail waveforms, MCU reset, Hall validity, or actual motor body-diode tail; this screen therefore exposes an input contract and capacitor/power bounds instead of inventing a shared time trace',
    board_interference: 'run cases A hold only, B hold only, both hold, neither hold. Hardware input is shared LOAD_POWER_VALID; two boards share BUS/GND and one A-side 1k bias per population contract. Independent local holds must not source LOAD_POWER_VALID or each other; cross-current and ground displacement need bench measurement.'
  },
  limits: [
    'No code or wiring maps PC6 or any other GPIO to local hold; candidate stuck-high can defeat cold-start/fault default-off unless a separate hardware actual-main-path interlock is proven.',
    '366.554mA is a modeled inductor transient peak used only as a stress case; it is not a continuous load or steady hold budget. Buck startup current upper bound previously screened at 0.685A exceeds 0.610A maximum limit point; cold-start qualification remains open.',
    'No 3V3 budget/capacitance, MCU BOR waveform, Hall sensor supply/current, SAFE/DUMP dropout trajectory, DRV8353 UVLO/shutdown behavior, gate VGS, torque, or regenerative diode current inferred here.',
    'Local control power alone is insufficient to credit stopping torque: integrated vehicle model gives zero actual installed-FOC braking credit, and brake continuation after main path opens is counterfactual.'
  ],
  official_sources: {
    TI_DRV8353: 'https://www.ti.com/lit/ds/symlink/drv8353.pdf',
    TI_LM5008A: 'https://www.ti.com/lit/ds/symlink/lm5008a.pdf',
    note: 'TI DRV835x datasheet identifies the integrated buck architecture as LM5008A. Standalone LM5008A electrical limits are not automatically transferable to the integrated driver implementation.'
  }
};
await writeFile(outPath, JSON.stringify(result, null, 2) + '\n', 'utf8');
console.log(`FOC local supply tail screen: native anchors verified; direct 5V C=${(c5Direct*1e6).toFixed(2)}uF; HOLD`);
