// Reproduce the dual-rejection, 15 km/h, 3 m whole-vehicle bus comparison.
// Run from the repository root with:
// node 电力/检查/联合仿真/15kmh_3m_双拒充早泄放.mjs
// This remains a reduced-order behavior model, not a switched-circuit or vehicle qualification.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '../../..');
const INPUT_PATH = path.join(HERE, '15kmh_3m_双拒充早泄放输入.json');
const input = JSON.parse(fs.readFileSync(INPUT_PATH, 'utf8'));
const sha256 = (buf) => crypto.createHash('sha256').update(buf).digest('hex');

const vehicle = input.confirmed_vehicle_request;
const assumption = input.engineering_assumptions_only;
const bus = input.bus_comparison;
const pb = input.current_pb0_pb7_behavior_candidate;
const v0 = vehicle.initial_speed_kmh / 3.6;
const mass = assumption.mass_kg;
const stopDistance = vehicle.stop_distance_m_from_brake_request;
const cBus = bus.effective_common_bus_capacitance_f;
const vBus0 = bus.initial_bus_v;
const compareV = bus.bus_comparison_limit_v;

function stopDemand(delayS, extraBusSourceW = 0) {
  const remaining = stopDistance - v0 * delayS;
  if (remaining <= 0) {
    return { delay_s: delayS, feasible_distance: false, remaining_distance_m: remaining };
  }
  const decel = v0 * v0 / (2 * remaining);
  const activeStopS = v0 / decel;
  const forceN = mass * decel;
  const pPeakMechanicalW = forceN * v0;
  return {
    feasible_distance: true,
    delay_s: delayS,
    delay_status: 'sensitivity_only_not_a_released_timing_limit',
    remaining_distance_m: remaining,
    required_constant_deceleration_m_s2: decel,
    total_stop_time_s: delayS + activeStopS,
    total_stop_distance_m: stopDistance,
    translation_only_brake_force_n: forceN,
    ideal_shaft_torque_each_motor_nm_comparison: forceN * 0.03 / 8,
    initial_mechanical_regen_w: pPeakMechanicalW,
    initial_mechanical_regen_per_foc_w: pPeakMechanicalW / 2,
    extra_bus_source_w_sensitivity: extraBusSourceW,
    initial_net_bus_injection_w: pPeakMechanicalW + extraBusSourceW,
    mechanical_kinetic_energy_j: 0.5 * mass * v0 * v0,
    brake_active_time_s: activeStopS,
    net_stop_energy_to_bus_j_ideal_efficiency: 0.5 * mass * v0 * v0 +
      extraBusSourceW * activeStopS
  };
}

function busEnergyWindow(fromV, toV = compareV) {
  return toV > fromV ? 0.5 * cBus * (toV * toV - fromV * fromV) : 0;
}

function noAbsorptionCrossTime(targetV, delayS, extraBusSourceW) {
  const required = busEnergyWindow(vBus0, targetV);
  const d = stopDemand(delayS, extraBusSourceW);
  if (!d.feasible_distance) return null;
  const stopS = d.total_stop_time_s;
  const sourceEnergyUntil = (t) => {
    const activeT = Math.max(0, Math.min(d.brake_active_time_s, t - delayS));
    const v = Math.max(0, v0 - (v0 / d.brake_active_time_s) * activeT);
    const eMech = 0.5 * mass * (v0 * v0 - v * v);
    const eOther = extraBusSourceW * activeT;
    return eMech + eOther;
  };
  if (sourceEnergyUntil(stopS) < required) return null;
  let lo = 0, hi = stopS;
  for (let i = 0; i < 80; i++) {
    const mid = 0.5 * (lo + hi);
    if (sourceEnergyUntil(mid) >= required) hi = mid;
    else lo = mid;
  }
  return hi;
}

function rlStep(i0, voltage, dt, switchOn, r, l, vf) {
  const tau = l / r;
  let activeDt = dt;
  let u = switchOn ? voltage : -vf;
  let a = u / r;
  let b = i0 - a;
  let decay = Math.exp(-activeDt / tau);
  let i1 = a + b * decay;
  if (!switchOn && i1 < 0) {
    activeDt = i0 <= 0 ? 0 : tau * Math.log((i0 + vf / r) / (vf / r));
    activeDt = Math.min(dt, Math.max(0, activeDt));
    a = -vf / r;
    b = i0 - a;
    decay = activeDt > 0 ? Math.exp(-activeDt / tau) : 1;
    i1 = activeDt > 0 ? Math.max(0, a + b * decay) : 0;
  }
  const charge = activeDt > 0
    ? a * activeDt + b * tau * (1 - decay)
    : 0;
  const decay2 = activeDt > 0 ? Math.exp(-2 * activeDt / tau) : 1;
  const currentSquaredIntegral = activeDt > 0
    ? a * a * activeDt +
      2 * a * b * tau * (1 - decay) +
      b * b * tau * 0.5 * (1 - decay2)
    : 0;
  return {
    i1: Math.max(0, i1),
    busEnergyJ: switchOn ? voltage * charge : 0,
    resistorEnergyJ: r * currentSquaredIntegral,
    currentChargeC: charge
  };
}

function simulatePb0Pb7Stop(delayS, extraBusSourceW, capture = false) {
  const d = stopDemand(delayS, extraBusSourceW);
  const dt = 1e-6;
  const vOn = pb.modeled_worst_common_offset_on_v;
  const vOff = pb.modeled_worst_common_offset_off_v;
  const r = pb.hot_total_series_resistance_ohm;
  const l = pb.loop_inductance_h;
  const vf = pb.freewheel_drop_v;
  const period = 1 / pb.pb7_frequency_hz;
  const onWidth = pb.effective_on_time_per_period_s;
  const firstCycle = pb.first_request_phase_delay_s;
  const requestToPwm = firstCycle + pb.per_period_gate_opening_loss_s;
  const endT = d.total_stop_time_s + 0.01;
  const steps = Math.ceil(endT / dt);
  let energy = 0.5 * cBus * vBus0 * vBus0;
  let vBus = vBus0, iDump = pb.initial_branch_current_a;
  let dumpPendingAt = null, pwmEnableAt = null;
  let firstThreshold = null, firstEffective = null, firstCompare = null, firstFocBov = null;
  let maxBus = vBus, maxCurrent = 0, maxBusAbsorbW = 0, maxResistorPowerW = 0;
  const absorbWindowN = Math.round(period / dt);
  const absorbWindow = Array(absorbWindowN).fill(0);
  let absorbWindowJ = 0, absorbWindowIndex = 0, max50usBusAbsorbW = 0;
  let eMech = 0, eOther = 0, eBusAbsorb = 0, eResistor = 0;
  let maxNetEnergyAboveTrigger = 0, netEnergyAt39 = null;
  let pwmSwitchOn = false;
  const trace = [];
  const traceEvery = Math.round(0.001 / dt);

  for (let k = 0; k <= steps; k++) {
    const t = k * dt;
    if (t > endT) break;
    const activeT = Math.max(0, Math.min(d.brake_active_time_s, t - delayS));
    const vVehicle = Math.max(0, v0 - (v0 / d.brake_active_time_s) * activeT);
    const pMechanical = t >= delayS && t <= d.total_stop_time_s
      ? mass * (v0 / d.brake_active_time_s) * vVehicle
      : 0;
    const pOther = t >= delayS && t <= d.total_stop_time_s ? extraBusSourceW : 0;
    const pSource = pMechanical + pOther;
    if (k === steps) break;

    if (pwmEnableAt === null && dumpPendingAt === null && vBus >= vOn) {
      if (firstThreshold === null) firstThreshold = t;
      dumpPendingAt = t + firstCycle;
    }
    if (dumpPendingAt !== null && t >= dumpPendingAt) {
      pwmEnableAt = dumpPendingAt;
      dumpPendingAt = null;
      if (firstEffective === null)
        firstEffective = pwmEnableAt + pb.per_period_gate_opening_loss_s;
    }
    if (pwmEnableAt !== null && vBus <= vOff) {
      pwmEnableAt = null;
      dumpPendingAt = null;
    }
    pwmSwitchOn = false;
    if (pwmEnableAt !== null) {
      const phase = ((t - pwmEnableAt) % period + period) % period;
      pwmSwitchOn = phase >= pb.per_period_gate_opening_loss_s &&
        phase < pb.per_period_gate_opening_loss_s + onWidth;
    }

    const rl = rlStep(iDump, vBus, dt, pwmSwitchOn, r, l, vf);
    const pBusAbsorb = rl.busEnergyJ / dt;
    const pResistor = rl.resistorEnergyJ / dt;
    const nextEnergy = Math.max(0, energy + (pSource * dt) - rl.busEnergyJ);
    const oldEnergy = energy;
    energy = nextEnergy;
    vBus = Math.sqrt(2 * energy / cBus);
    iDump = rl.i1;

    eMech += pMechanical * dt;
    eOther += pOther * dt;
    eBusAbsorb += rl.busEnergyJ;
    eResistor += rl.resistorEnergyJ;
    maxBus = Math.max(maxBus, vBus);
    maxCurrent = Math.max(maxCurrent, iDump);
    maxBusAbsorbW = Math.max(maxBusAbsorbW, pBusAbsorb);
    maxResistorPowerW = Math.max(maxResistorPowerW, pResistor);
    absorbWindowJ += rl.busEnergyJ - absorbWindow[absorbWindowIndex];
    absorbWindow[absorbWindowIndex] = rl.busEnergyJ;
    absorbWindowIndex = (absorbWindowIndex + 1) % absorbWindowN;
    max50usBusAbsorbW = Math.max(max50usBusAbsorbW,
      absorbWindowJ / (absorbWindowN * dt));
    if (firstCompare === null && vBus >= compareV) firstCompare = t + dt;
    if (firstFocBov === null && vBus >= bus.foc_earliest_paper_bov_rising_v) firstFocBov = t + dt;
    if (firstThreshold !== null) {
      const netAbove = energy - 0.5 * cBus * vOn * vOn;
      maxNetEnergyAboveTrigger = Math.max(maxNetEnergyAboveTrigger, netAbove);
      if (firstCompare !== null && netEnergyAt39 === null)
        netEnergyAt39 = netAbove;
    }
    if (capture && k % traceEvery === 0) {
      trace.push([
        t,
        vBus,
        vVehicle,
        pMechanical / 2,
        pMechanical / 2,
        pOther,
        pwmEnableAt === null ? 0 : 1,
        pwmSwitchOn ? 1 : 0,
        iDump,
        pBusAbsorb,
        pResistor,
        eResistor
      ]);
    }
  }
  const totalInput = eMech + eOther;
  const finalBusEnergy = energy;
  return {
    lpv_core_alive: true,
    battery_absorption_w: 0,
    supercap_absorption_w: 0,
    unconfirmed_auxiliary_sink_w: 0,
    delay_s: delayS,
    extra_bus_source_w_sensitivity: extraBusSourceW,
    model_pwm_threshold_on_v: vOn,
    model_pwm_threshold_off_v: vOff,
    first_pb0_threshold_s: firstThreshold,
    first_pb7_effective_current_s: firstEffective,
    first_bus_limit_crossing_s: firstCompare,
    first_foc_paper_bov_crossing_s: firstFocBov,
    maximum_bus_v: maxBus,
    passed_39p009_model_comparison: maxBus < compareV,
    branch_peak_current_a: maxCurrent,
    bus_absorption_1us_step_peak_w_model_only: maxBusAbsorbW,
    bus_absorption_50us_average_peak_w_model_only: max50usBusAbsorbW,
    resistor_instantaneous_power_peak_w_model_only: maxResistorPowerW,
    resistor_energy_j_model_only: eResistor,
    mechanical_regen_energy_j_model_only: eMech,
    extra_source_energy_j_model_only: eOther,
    bus_energy_change_j: finalBusEnergy - 0.5 * cBus * vBus0 * vBus0,
    bus_energy_balance_error_j: totalInput - eBusAbsorb -
      (finalBusEnergy - 0.5 * cBus * vBus0 * vBus0),
    maximum_bus_energy_excursion_above_trigger_j: maxNetEnergyAboveTrigger,
    physical_status: 'HOLD: this model assumes PB0 calibration, PB7 command, the stated hot R/L loop and a motor torque profile; none is a full-temperature installed guarantee.'
  , trace };
}

function simulateHardTlvCounterfactual(delayS, extraBusSourceW) {
  const d = stopDemand(delayS, extraBusSourceW);
  const e40 = busEnergyWindow(vBus0, input.legacy_hard_tlv_candidate.on_v_nominal);
  const eFoc = busEnergyWindow(vBus0, bus.foc_earliest_paper_bov_rising_v);
  const eCompare = busEnergyWindow(vBus0, compareV);
  const noAbsorb = (v) => noAbsorptionCrossTime(v, delayS, extraBusSourceW);
  return {
    lpv_core_alive: false,
    central_12v_alive: true,
    software_pb0_pb7_available: false,
    hard_tlv_nominal_on_v: input.legacy_hard_tlv_candidate.on_v_nominal,
    hard_tlv_nominal_off_v: input.legacy_hard_tlv_candidate.off_v_nominal,
    energy_from_initial_to_39p009_j: eCompare,
    energy_from_initial_to_foc_earliest_bov_j: eFoc,
    energy_from_initial_to_old_tlv_on_j: e40,
    time_to_39p009_if_no_sink_s: noAbsorb(compareV),
    time_to_foc_bov_if_no_sink_s: noAbsorb(bus.foc_earliest_paper_bov_rising_v),
    time_to_old_tlv_if_foc_keeps_injecting_counterfactual_s: noAbsorb(input.legacy_hard_tlv_candidate.on_v_nominal),
    central_hard_dump_preempts_foc_bov: input.legacy_hard_tlv_candidate.on_v_nominal <
      bus.foc_earliest_paper_bov_rising_v,
    finding: 'The paper FOC BOV lower corner is reached before the legacy central 40.32 V nominal threshold. Once FOC Break changes torque, the remaining stop and body-diode energy are outside this reduced model; the 40.32 V counterfactual is not a physical stop result.',
    physical_status: 'HOLD: with LPV/CORE revoked, PB0/PB7 is not credited; the 40.32 V backup is later than the FOC paper BOV lower corner, and neither energy disposal nor 3 m stopping is proven.'
  };
}

function earlyCandidateTiming(delayS, extraBusSourceW, candidateTripV) {
  const d = stopDemand(delayS, extraBusSourceW);
  const pPeak = d.initial_net_bus_injection_w;
  const eWindow = busEnergyWindow(candidateTripV, compareV);
  const windowS = pPeak > 0 ? eWindow / pPeak : null;
  const r = input.independent_hard_early_candidate.comparison_proxy.resistance_ohm;
  const l = input.independent_hard_early_candidate.comparison_proxy.inductance_h;
  const vTrip = candidateTripV;
  const iNeeded = pPeak / vTrip;
  const tau = l / r;
  const rampToBalanceS = iNeeded >= vTrip / r
    ? Infinity
    : -tau * Math.log(1 - r * iNeeded / vTrip);
  const delays = [0, 0.00016, 0.0003, 0.001, 0.0012];
  return {
    lpv_core_alive: null,
    candidate_latest_on_v: candidateTripV,
    delay_sensitivity_only: delays.map(delayToCurrent => ({
      delay_to_first_switch_current_s: delayToCurrent,
      remaining_energy_before_bus_limit_j: eWindow - pPeak * delayToCurrent,
      time_to_required_bus_absorption_after_threshold_s: delayToCurrent + rampToBalanceS,
      within_voltage_window_model_only: delayToCurrent + rampToBalanceS < windowS,
      assumed_hot_loop_r_ohm: r,
      assumed_hot_loop_l_h: l,
      initial_branch_current_a: 0,
      branch_current_needed_to_match_source_a: iNeeded,
      branch_current_needed_resistor_power_w: iNeeded * iNeeded * r,
      continuous_full_on_current_at_trip_a: vTrip / r,
      continuous_full_on_resistor_power_at_trip_w: vTrip * vTrip / r
    })),
    energy_window_j: eWindow,
    net_injection_peak_w: pPeak,
    no_absorption_window_s: windowS,
    current_ramp_to_match_s_after_switching: rampToBalanceS,
    maximum_delay_to_first_effective_current_s: Math.max(0, windowS - rampToBalanceS),
    maximum_total_delay_to_source_matching_absorption_s: windowS,
    candidate_release_and_rearm_behavior: 'NOT MODELED: no release threshold or latch/rearm details supplied.',
    status: 'HOLD: the corrected 38.738 V latest threshold is a conditional threshold input. Candidate R/L, powered domain, actual comparator/driver delay, MOS current waveform, and release/fault behavior are absent.'
  };
}

function thresholdTable() {
  const points = [
    { name: 'PB0/PB7 latest common-offset threshold', onV: pb.modeled_worst_common_offset_on_v,
      offV: pb.modeled_worst_common_offset_off_v, status: 'MODEL INPUT; PB0 calibration static FAIL' },
    { name: 'independent hard early candidate latest', onV: input.independent_hard_early_candidate.latest_on_v_reported_v,
      offV: input.independent_hard_early_candidate.release_min_v_reported_v,
      status: 'CONDITIONAL; HOLD pending circuit and delay' },
    { name: 'older 38.742 V threshold sensitivity', onV: input.independent_hard_early_candidate.older_threshold_sensitivity_v,
      offV: input.independent_hard_early_candidate.release_min_v_reported_v,
      status: 'Prior sensitivity only; corrected candidate is 38.738 V' },
    { name: 'LTC2965 nominal', onV: input.ltc2965_candidate.nominal_on_v,
      offV: input.ltc2965_candidate.nominal_off_v, status: 'CANDIDATE; static range FAIL' },
    { name: 'LTC2965 latest initial-corner action', onV: input.ltc2965_candidate.on_range_v[1],
      offV: input.ltc2965_candidate.off_range_v[0], status: 'FAIL against 39.009 V and FOC low BOV corner' },
    { name: 'legacy TLV hard BOV nominal', onV: input.legacy_hard_tlv_candidate.on_v_nominal,
      offV: input.legacy_hard_tlv_candidate.off_v_nominal, status: 'LATE for early takeover' }
  ];
  return points.map(x => {
    const energyJ = busEnergyWindow(x.onV, compareV);
    const noSinkP = stopDemand(0.1, 30).initial_net_bus_injection_w;
    return {
      ...x,
      energy_to_39p009_j: energyJ,
      margin_time_at_100ms_delay_plus_30w_sensitivity_s: energyJ > 0 ? energyJ / noSinkP : 0,
      false_dump_possible_at_37p8_v: x.onV <= vBus0 || (x.offV !== null && x.offV < vBus0),
      precedes_foc_paper_bov_min: x.onV < bus.foc_earliest_paper_bov_rising_v
    };
  });
}

const ledger = input.sources.map(s => {
  const abs = path.resolve(ROOT, s.path);
  if (!fs.existsSync(abs)) return { ...s, exists: false, sha256: null };
  return { ...s, exists: true, sha256: sha256(fs.readFileSync(abs)) };
});

const delays = assumption.brake_build_delay_sweep;
const primaryCases = [];
const waveTraces = [];
for (const delayS of delays) {
  for (const extraSourceW of [0, assumption.mppt_injection_sensitivity_w]) {
    const capture = delayS === Math.max(...delays) &&
      extraSourceW === assumption.mppt_injection_sensitivity_w;
    const sim = simulatePb0Pb7Stop(delayS, extraSourceW, capture);
    const trace = sim.trace;
    delete sim.trace;
    const d = stopDemand(delayS, extraSourceW);
    const noSinkCompare = noAbsorptionCrossTime(compareV, delayS, extraSourceW);
    const noSinkBov = noAbsorptionCrossTime(bus.foc_earliest_paper_bov_rising_v, delayS, extraSourceW);
    primaryCases.push({
      stop_demand: d,
      no_sink_time_to_39p009_s_from_brake_request: noSinkCompare,
      no_sink_time_to_foc_39p209_s_from_brake_request: noSinkBov,
      pb0_pb7_system_behavior: sim,
      lpv_revoked_fallback: simulateHardTlvCounterfactual(delayS, extraSourceW),
      independent_early_candidate_window: earlyCandidateTiming(
        delayS, extraSourceW, input.independent_hard_early_candidate.latest_on_v_reported_v)
    });
    if (capture) waveTraces.push(...trace.map(row => [delayS, extraSourceW, ...row]));
  }
}

const falseDump = {
  normal_full_battery_bus_v: vBus0,
  legacy_tlv_nominal_is_above_normal_bus: input.legacy_hard_tlv_candidate.on_v_nominal > vBus0,
  pb0_calibrated_release_can_be_held_above_normal_bus: false,
  pb0_note: 'The modeled 38.35 V release is a common-offset sensitivity only. PB0 reference/ADC threshold is not currently qualified; false-on at the normal 37.8 V bus cannot be ruled out from current evidence.',
  ltc2965_low_corner_action_v: input.ltc2965_candidate.on_range_v[0],
  ltc2965_low_corner_release_v: input.ltc2965_candidate.off_range_v[0],
  ltc2965_false_dump_possible_at_normal_bus: input.ltc2965_candidate.on_range_v[0] < vBus0 ||
    input.ltc2965_candidate.off_range_v[0] < vBus0,
  full_on_current_a_at_37p8v_and_minimum_resistance:
    vBus0 / bus.dump_resistor_range_ohm[0],
  full_on_resistor_power_w_at_37p8v_and_minimum_resistance:
    vBus0 * vBus0 / bus.dump_resistor_range_ohm[0],
  early_hard_candidate_minimum_action_v: input.independent_hard_early_candidate.on_range_v_reported[0],
  early_hard_candidate_minimum_release_v: input.independent_hard_early_candidate.release_min_v_reported_v,
  early_hard_candidate_static_false_dump_at_37p8_possible:
    input.independent_hard_early_candidate.on_range_v_reported[0] <= vBus0 ||
    input.independent_hard_early_candidate.release_min_v_reported_v < vBus0,
  early_hard_candidate_false_dump_status:
    'The supplied static corners are above 37.8 V; a static threshold false dump is not indicated in this screen. Startup, brownout, signal loss, and dynamic rearm states remain HOLD.'
};

const result = {
  schema: 'agtc-joint-regeneration-15kmh-3m-result-v1',
  qualification: 'HOLD',
  produced_utc: new Date().toISOString(),
  input_file: path.basename(INPUT_PATH),
  source_ledger: ledger,
  confirmed_request: vehicle,
  assumption_boundary: assumption,
  energy_window_comparisons: {
    bus_cap_f: cBus,
    initial_bus_v: vBus0,
    comparison_limit_v: compareV,
    initial_to_comparison_energy_j: busEnergyWindow(vBus0, compareV),
    pb0_latest_38p50_to_limit_energy_j: busEnergyWindow(pb.modeled_worst_common_offset_on_v, compareV),
    early_latest_38p738_to_limit_energy_j: busEnergyWindow(
      input.independent_hard_early_candidate.latest_on_v_reported_v, compareV),
    early_old_38p742_sensitivity_to_limit_energy_j: busEnergyWindow(
      input.independent_hard_early_candidate.older_threshold_sensitivity_v, compareV),
    foc_paper_bov_earliest_rising_v: bus.foc_earliest_paper_bov_rising_v,
    note: 'Capacitance and thresholds are conditional comparison points. Effective local bus capacitance, bus-to-near-bridge offset, device delays and line/winding energy are not bounded here.'
  },
  threshold_comparison: thresholdTable(),
  lpv_core_alive_dual_rejection_cases: primaryCases.map(x => ({
    delay_s: x.stop_demand.delay_s,
    extra_bus_source_w_sensitivity: x.stop_demand.extra_bus_source_w_sensitivity,
    stop_demand: x.stop_demand,
    no_sink_time_to_39p009_s_from_brake_request: x.no_sink_time_to_39p009_s_from_brake_request,
    no_sink_time_to_foc_39p209_s_from_brake_request: x.no_sink_time_to_foc_39p209_s_from_brake_request,
    pb0_pb7_system_behavior: x.pb0_pb7_system_behavior,
    independent_early_candidate_window: x.independent_early_candidate_window
  })),
  lpv_or_core_revoked_central_12v_alive: {
    interpretation: 'R1 software PB0/PB7 is disabled by the LPV/CORE hard gate. Only legacy 40.32 V TLV hard backup is counted, and the FOC local BOV/Break threshold is earlier on paper.',
    cases: primaryCases.map(x => ({
      delay_s: x.stop_demand.delay_s,
      extra_bus_source_w_sensitivity: x.stop_demand.extra_bus_source_w_sensitivity,
      fallback: x.lpv_revoked_fallback
    })),
    status: 'HOLD: fallback does not establish continuous 3 m braking after FOC BOV/Break.'
  },
  central_12v_lost: {
    software_pb0_pb7_available: false,
    central_hard_tlv_dump_available: false,
    central_absorption_credit_w: 0,
    no_sink_time_to_foc_39p209_s: primaryCases.find(x =>
      x.stop_demand.delay_s === 0.1 && x.stop_demand.extra_bus_source_w_sensitivity === 30)
      ?.no_sink_time_to_foc_39p209_s_from_brake_request ?? null,
    cases: primaryCases.map(x => ({
      delay_s: x.stop_demand.delay_s,
      extra_bus_source_w_sensitivity: x.stop_demand.extra_bus_source_w_sensitivity,
      time_to_39p009_s: x.no_sink_time_to_39p009_s_from_brake_request,
      time_to_foc_39p209_s: x.no_sink_time_to_foc_39p209_s_from_brake_request,
      motor_speed_at_foc_bov_m_s: x.no_sink_time_to_foc_39p209_s_from_brake_request === null
        ? null
        : Math.max(0, v0 - (v0 / x.stop_demand.brake_active_time_s) *
          Math.max(0, x.no_sink_time_to_foc_39p209_s_from_brake_request - x.stop_demand.delay_s)),
      stop_distance_remaining_at_foc_bov_m: x.no_sink_time_to_foc_39p209_s_from_brake_request === null
        ? null
        : Math.max(0, stopDistance - (v0 * Math.min(x.stop_demand.delay_s,
          x.no_sink_time_to_foc_39p209_s_from_brake_request) +
          (v0 * Math.max(0, x.no_sink_time_to_foc_39p209_s_from_brake_request -
            x.stop_demand.delay_s) -
            0.5 * (v0 / x.stop_demand.brake_active_time_s) *
            Math.max(0, x.no_sink_time_to_foc_39p209_s_from_brake_request -
              x.stop_demand.delay_s) ** 2)))
    }))
      .map(x => ({ ...x, status: 'post-Break vehicle distance is unresolved; no mechanical brake or qualified passive torque model' })),
    at_foc_bov: 'The motor torque path may be cut. Break is not an energy sink; post-Break torque, body-diode flow and vehicle stop distance are unmodeled.',
    status: 'HOLD: no qualified central absorption or post-Break stopping behavior.'
  },
  communication_loss: {
    dual_rejection_default: 'Battery and supercapacitor absorption are both forced to 0 W immediately in this model.',
    pb0_local_bus_sense: 'The PB0 bus-voltage loop is modeled as local and does not need CAN while LPV/CORE and its reference are healthy.',
    software_default_on_can_timeout: 'NOT ESTABLISHED: firmware timeout, permit revocation, and PWM command default were not imported as guaranteed state transitions.',
    if_lpv_core_remains_alive: 'Use the PB0/PB7 behavior-model cases above; their electrical result is MODEL PASS ONLY.',
    if_lpv_core_is_revoked: 'PB0/PB7 is blocked and the old 40.32 V TLV fallback remains later than the FOC 39.209 V paper lower corner.',
    status: 'HOLD: the CAN-loss to LPV/CORE/PB7 state path has no end-to-end timing or default-state proof.'
  },
  central_12v_sag: {
    active_dump_credit_w: 0,
    condition: 'Once 12V_DRV falls below the real driver/comparator operating guarantee, neither PB7 nor the old TLV path is credited.',
    measurement_boundary: '12V_DRV, DUMP_5V, DUMP_OFF, Q_DUMP VGS/VDS, resistor current, BREAK_N, six PWM/VGS and motor phase currents must be time-aligned.',
    post_foc_bov_behavior: 'The FOC local BOV may request Break, but post-Break back-EMF/body-diode current and vehicle stopping are not modeled.',
    status: 'HOLD: no guaranteed brownout threshold, response delay, dump hold-up or post-Break energy envelope is supplied.'
  },
  dual_refusal_trigger: {
    battery_ccl_a_credited: 0,
    battery_pack_net_charge_hard_limit_a: assumption.battery_pack_net_charge_hard_limit_a,
    supercapacitor_absorption_w_credited: 0,
    timing: 'Both absorption quotas are removed at the start of the modeled brake event, with no stale-CAN or in-flight absorption credit.',
    handoff: 'Central PB0/PB7 is bus-voltage-triggered in the LPV/CORE-alive model. Actual CCL/SC refusal recognition and transition delay are not measured.',
    status: 'HOLD for physical refusal recognition and takeover; the behavior model applies the zero-absorption state directly.'
  },
  false_dump_and_startup: falseDump,
  independent_hard_candidate_threshold_ranges: {
    candidate_type: input.independent_hard_early_candidate.candidate_type,
    rising_v: input.independent_hard_early_candidate.on_range_v_reported,
    falling_v: input.independent_hard_early_candidate.release_range_v_reported,
    minimum_action_and_release_above_full_battery_bus:
      input.independent_hard_early_candidate.on_range_v_reported[0] > vBus0 &&
      input.independent_hard_early_candidate.release_min_v_reported_v > vBus0,
    zero_regeneration_release_window_us:
      input.independent_hard_early_candidate.zero_regeneration_release_window_s_reported * 1e6,
    note: 'This static screen places the lowest action and release above 37.8 V. Dynamic false-on, hysteresis under startup/power loss, and current decay remain unverified.'
  },
  independent_spice_cross_check: {
    source_path: '电力/检查/联合仿真/双拒充泄放电路复跑.json',
    status: 'Read-only cross-check from parent; script does not execute or require this file.',
    given_case: '390 W constant, 38.50 V trigger, 1 ohm, 40 uH, 160 us first effective current, 30 us every 50 us.',
    cases: [
      {
        trigger_v: 38.5, constant_net_injection_w: 390,
        reported_bus_peak_v: 38.5337387, reported_bus_at_1ms_v: 38.476256,
        reported_wire_current_peak_a: 28.2846029
      },
      {
        trigger_v: 38.738, constant_net_injection_w: 390,
        reported_bus_peak_v: 38.7714518, reported_bus_at_1ms_v: 38.7117498,
        reported_wire_current_peak_a: 28.4594172
      }
    ],
    comparison_limit_v: compareV,
    use_limit: 'Supports only the stated first-millisecond SPICE case; it does not prove the 420–450 W sensitivity, candidate hardware delay, long stop, MOS SOA, resistor thermal rating or full vehicle stop.'
  },
  result_conclusions: [
    'With LPV/CORE available, the reduced PB0/PB7 model keeps the bus below the 39.009 V comparison line across the 0–100 ms delay sensitivities and 0/30 W concurrent source sensitivities; this is MODEL PASS ONLY.',
    'The corrected independent hard candidate uses the 38.7383 V latest rising corner (38.742 V is retained only as an older sensitivity), giving about 0.556 J to 39.009 V. Under the 30 kg, 100 ms, +30 W comparison, peak net injection is about 450 W and the zero-absorption window is about 1.24 ms. With the reused 1 ohm/40 uH screening loop, matching this input takes about 14 us after effective switch current starts; the maximum total threshold-to-matching-absorption time is about 1.22 ms, leaving roughly 1.22 ms for comparator/gate delay before first current. A 0.30 ms first-current delay consumes about 0.135 J and leaves about 0.421 J before current begins. Candidate-specific gate path, delay, release behavior and thermal limits are still absent, so status is HOLD.',
    'The legacy 40.32 V central threshold is later than both 39.009 V and the FOC paper BOV lower corner of 39.209 V. Under LPV/CORE revocation, PB0/PB7 is not credited and the 3 m stop cannot be demonstrated after the FOC Break path acts.',
    'The LTC2965 candidate initial-corner action range extends to 39.704 V and release range down to 37.243 V. This fails early threshold ordering and can false-dump at a 37.8 V full-battery bus; its specified 80 us response point is not this 40x low-overdrive circuit.',
    'A 30 kg mass, flat tire contact, constant deceleration after a zero-force delay, equal FOC split, ideal mechanical-to-bus conversion, and 52.86 mF common capacitance are comparison assumptions. They do not qualify actual motor torque, 3 m stopping, resistor pulse/repetition heat, MOS SOA, local BOV behavior or physical wiring.'
  ],
  remaining_hold_items: [
    'PB0 calibrated threshold, code conversion fix, actual sampling/filter/update latency and independent hysteresis corners.',
    'PB7 command and LPV/CORE hard-gate behavior under BMS disconnect, CAN loss, MCU reset, and partial rail loss.',
    'Current 0.82 ohm resistor exact MPN, hot resistance and L/R/diode path; PWM pulse, continuous and repeated thermal/SOA qualification.',
    'Independent early candidate actual schematic nets, minimum and release thresholds, supply domain, comparator/output delay, MOS gate/current waveforms and latching/reset behavior.',
    'The separate AO3400A PB7 candidate is not directly electrically linked in this model and is not credited to the LPV/CORE-revoked case.',
    'FOC-A/B measured torque build profile, back-EMF, phase current, local BUS, near-bridge/BUS offset and Break-to-current-tail waveform.',
    'Central 12 V brownout and complete loss; active dump becomes 0 W, while post-Break body-diode energy and vehicle coasting remain unresolved.',
    'Actual LPV behavior and BMS physical open path, including which capacitance stays connected after the fault.',
    '30 W MPPT is only a sensitivity; its current topology and behavior after rejection are not bound.'
  ],
  csv_waveform: '15kmh_3m_双拒充PB0_PB7_100ms_30W波形.csv',
  standalone_execution: 'node 电力/检查/联合仿真/15kmh_3m_双拒充早泄放.mjs'
};

const numericalChecks = {
  all_pb0_pb7_cases_below_bus_line: primaryCases.every(x =>
    Number.isFinite(x.pb0_pb7_system_behavior.maximum_bus_v) &&
    x.pb0_pb7_system_behavior.maximum_bus_v < compareV),
  all_no_sink_cases_reach_bus_line_before_the_modeled_stop: primaryCases.every(x =>
    Number.isFinite(x.no_sink_time_to_39p009_s_from_brake_request)),
  all_no_sink_cases_reach_foc_bov_before_the_modeled_stop: primaryCases.every(x =>
    Number.isFinite(x.no_sink_time_to_foc_39p209_s_from_brake_request)),
  each_case_has_finite_resistor_energy: primaryCases.every(x =>
    Number.isFinite(x.pb0_pb7_system_behavior.resistor_energy_j_model_only)),
  all_electrical_energy_residuals_below_0p01_j: primaryCases.every(x =>
    Math.abs(x.pb0_pb7_system_behavior.bus_energy_balance_error_j) < 0.01)
};
if (Object.values(numericalChecks).some(x => !x))
  throw new Error(`Joint model output check failed: ${JSON.stringify(numericalChecks)}`);
result.numerical_checks = numericalChecks;

const csv = [
  ['delay_s', 'extra_bus_source_w', 'time_s', 'bus_v', 'vehicle_speed_m_s',
    'foc_a_mechanical_bus_injection_w', 'foc_b_mechanical_bus_injection_w',
    'other_source_w', 'pb7_pwm_enabled', 'pb7_switch_on', 'dump_current_a',
    'dump_bus_absorption_w', 'resistor_instantaneous_power_w', 'resistor_cumulative_energy_j'],
  ...waveTraces
].map(row => row.map(value => {
  const s = String(value ?? '');
  return /[",\r\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
}).join(',')).join('\n') + '\n';
fs.writeFileSync(path.join(HERE, result.csv_waveform), csv, 'utf8');
fs.writeFileSync(path.join(HERE, '15kmh_3m_双拒充早泄放结果.json'),
  JSON.stringify(result, null, 2) + '\n', 'utf8');
console.log(JSON.stringify({
  qualification: result.qualification,
  cases: result.lpv_core_alive_dual_rejection_cases.length,
  latest_bus_v: Math.max(...result.lpv_core_alive_dual_rejection_cases.map(x =>
    x.pb0_pb7_system_behavior.maximum_bus_v)),
  threshold_window_j_early: result.energy_window_comparisons.early_latest_38p738_to_limit_energy_j,
  source_missing_count: ledger.filter(x => !x.exists).length,
  result: path.join(HERE, '15kmh_3m_双拒充早泄放结果.json')
}, null, 2));
