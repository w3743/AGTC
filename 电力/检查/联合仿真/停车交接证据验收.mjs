#!/usr/bin/env node
// Validate the measured stop-handoff evidence package. This program does not simulate.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const CASES = [
  'cold_start',
  'brownout',
  'low_bus_dual_foc',
  'low_speed_stop',
  'dual_refusal_main_first',
  'dual_refusal_old_rails_first',
  'dual_refusal_simultaneous',
  'pump_thermal_required',
  'load_disturbance',
  'upstream_cut_no_load',
];
const STOP_CASES = new Set([
  'low_speed_stop',
  'dual_refusal_main_first',
  'dual_refusal_old_rails_first',
  'dual_refusal_simultaneous',
]);
const BRAKING_WINDOW_CASES = new Set([
  'low_bus_dual_foc',
  'low_speed_stop',
  'dual_refusal_main_first',
  'dual_refusal_old_rails_first',
  'dual_refusal_simultaneous',
]);
const COLUMNS = [
  'time_s', 'distance_m', 'speed_mps', 'bus_v',
  'near_bridge_a_v', 'near_bridge_b_v', 'foc_a_rail_v', 'foc_b_rail_v',
  'torque_a_nm', 'torque_b_nm', 'id_a_a', 'id_b_a', 'iq_a_a', 'iq_b_a',
  'phase_current_a_rms', 'phase_current_b_rms', 'battery_current_a',
  'supercap_current_a', 'dump_current_a', 'dump_resistor_temp_c',
  'pump_state', 'central_12v_v', 'brake_request_signal', 'event_marker_signal',
  'battery_refusal_signal', 'supercap_refusal_signal', 'upstream_disconnect_signal',
  'old_rails_lost_signal', 'brownout_signal', 'load_step_signal',
  'main_path_current_a', 'foc_local_rail_v', 'bq_alert_active', 'main_clr_active',
  'fet_release_active', 'load_power_valid_active', 'pc10_v',
  'software_fault_latched', 'traction_inhibited', 'local_reset_active',
  'pc6_hold_active', 'pwm_enabled',
];
const CORE_COLUMNS = [
  'time_s', 'bus_v', 'near_bridge_a_v', 'near_bridge_b_v',
  'foc_a_rail_v', 'foc_b_rail_v', 'central_12v_v', 'event_marker_signal',
];
const CURRENT_COLUMNS = ['battery_current_a', 'supercap_current_a', 'dump_current_a'];
const MOTION_COLUMNS = [
  'distance_m', 'speed_mps', 'torque_a_nm', 'torque_b_nm',
  'id_a_a', 'id_b_a', 'iq_a_a', 'iq_b_a',
  'phase_current_a_rms', 'phase_current_b_rms', 'brake_request_signal',
];
const MOTION_SAMPLE_GAP_S = 0.005;
const BROWNOUT_SAMPLE_GAP_S = 0.0001;
const NO_LOAD_CUT_SAMPLE_GAP_S = 0.001;
const SLOW_THERMAL_SAMPLE_GAP_S = 1.0;
function requiredColumns(name) {
  const columns = [...CORE_COLUMNS];
  if (BRAKING_WINDOW_CASES.has(name)) {
    columns.push(...CURRENT_COLUMNS, ...MOTION_COLUMNS);
    if (name.startsWith('dual_refusal_')) {
      columns.push('battery_refusal_signal', 'supercap_refusal_signal', 'upstream_disconnect_signal', 'old_rails_lost_signal');
    }
    return columns;
  }
  return [...columns, ...({
    cold_start: [],
    brownout: ['brownout_signal'],
    pump_thermal_required: ['dump_current_a', 'pump_state', 'dump_resistor_temp_c'],
    load_disturbance: [...CURRENT_COLUMNS, 'load_step_signal'],
    upstream_cut_no_load: [...CURRENT_COLUMNS, 'upstream_disconnect_signal'],
  }[name] ?? [])];
}
const DIGITAL_COLUMNS = new Set([
  'pump_state', 'brake_request_signal', 'event_marker_signal',
  'battery_refusal_signal', 'supercap_refusal_signal',
  'upstream_disconnect_signal', 'old_rails_lost_signal',
  'brownout_signal', 'load_step_signal',
  'bq_alert_active', 'main_clr_active', 'fet_release_active',
  'load_power_valid_active', 'software_fault_latched', 'traction_inhibited',
  'local_reset_active', 'pc6_hold_active', 'pwm_enabled',
]);
const isFiniteNumber = value => typeof value === 'number' && Number.isFinite(value);
const hasSha256 = value => typeof value === 'string' && /^[0-9a-f]{64}$/i.test(value);

function hold(errors, caseSummaries = {}) {
  return {
    status: 'HOLD',
    qualification: 'HOLD',
    errors,
    case_summaries: caseSummaries,
    scope: 'Intake validation only; no simulation or release approval.',
  };
}

function parseCsv(text, requireResistorTemperature, columns) {
  const lines = text.replace(/^\uFEFF/, '').trim().split(/\r?\n/);
  if (lines.length < 2) throw new Error('needs at least two samples');

  const headers = lines[0].split(',').map(value => value.trim());
  const missing = columns.filter(column => !headers.includes(column));
  if (missing.length) throw new Error(`columns missing: ${missing.join(', ')}`);
  if (new Set(headers).size !== headers.length) throw new Error('duplicate CSV columns');

  return lines.slice(1).map(line => {
    const cells = line.split(',');
    const row = {};
    headers.forEach((header, index) => { row[header] = cells[index]?.trim() ?? ''; });

    for (const column of COLUMNS.filter(column => headers.includes(column))) {
      if (DIGITAL_COLUMNS.has(column)) {
        if (row[column] !== '0' && row[column] !== '1') {
          throw new Error(`${column} must be a sampled digital 0/1 channel`);
        }
        row[column] = Number(row[column]);
      } else if (column === 'dump_resistor_temp_c' && row[column] === '' && !requireResistorTemperature) {
        row[column] = null;
      } else {
        if (row[column] === '' || !Number.isFinite(Number(row[column]))) {
          throw new Error(`${column} must contain finite measured numeric data`);
        }
        row[column] = Number(row[column]);
      }
    }
    return row;
  });
}

function firstRisingEdge(rows, column) {
  const index = rows.findIndex((row, i) => row[column] === 1 && (i === 0 || rows[i - 1][column] === 0));
  return index < 0 ? null : { index, time_s: rows[index].time_s };
}

function observedRisingEdge(rows, column, minTimeS = -Infinity) {
  const index = rows.findIndex((row, i) => i > 0 && row.time_s >= minTimeS && row[column] === 1 && rows[i - 1][column] === 0);
  return index < 0 ? null : { index, time_s: rows[index].time_s };
}

function observedFallingEdge(rows, column, minTimeS = -Infinity) {
  const index = rows.findIndex((row, i) => i > 0 && row.time_s >= minTimeS && row[column] === 0 && rows[i - 1][column] === 1);
  return index < 0 ? null : { index, time_s: rows[index].time_s };
}

function observedThresholdRisingEdge(rows, column, threshold, minTimeS = -Infinity) {
  const index = rows.findIndex((row, i) => i > 0 && row.time_s >= minTimeS && row[column] >= threshold && rows[i - 1][column] < threshold);
  return index < 0 ? null : { index, time_s: rows[index].time_s };
}

function zeroTimeIndex(rows) {
  const indices = rows.flatMap((row, index) => Math.abs(row.time_s) <= 1e-9 ? [index] : []);
  return indices.length === 1 ? indices[0] : -1;
}

function maxSampleGap(rows) {
  return rows.slice(1).reduce((max, row, index) => Math.max(max, row.time_s - rows[index].time_s), 0);
}

function sampleGapLimit(name) {
  if (name === 'pump_thermal_required') return SLOW_THERMAL_SAMPLE_GAP_S;
  if (name === 'brownout' || name.startsWith('dual_refusal_')) return BROWNOUT_SAMPLE_GAP_S;
  if (name === 'upstream_cut_no_load') return NO_LOAD_CUT_SAMPLE_GAP_S;
  return MOTION_SAMPLE_GAP_S;
}

function validateSetup(manifest, errors) {
  const setup = manifest.setup ?? {};
  const required = [
    'vehicle_id', 'mass_kg', 'tire_radius_m', 'gear_ratio',
    'gear_efficiency_lower_bound', 'ground_condition', 'grade_percent',
    'sensor_calibration_ids', 'sensor_location_map', 'firmware_build_hash',
    'board_revision_ids', 'motor_serials',
  ];
  for (const key of required) {
    const value = setup[key];
    if (value === undefined || value === null || value === '' || (Array.isArray(value) && value.length === 0)) {
      errors.push(`setup.${key} missing`);
    }
  }
  if (setup.measurement_units !== 'SI') errors.push('setup.measurement_units must be SI');
  if (!hasSha256(setup.firmware_build_hash)) errors.push('setup.firmware_build_hash must be SHA-256');
  if (!isFiniteNumber(setup.grade_percent)) errors.push('setup.grade_percent must be measured numeric grade');
  if (setup.braking_torque_sign !== 1 && setup.braking_torque_sign !== -1) {
    errors.push('setup.braking_torque_sign must identify the measured forward-braking torque sign');
  }
  for (const key of ['mass_kg', 'tire_radius_m', 'gear_ratio', 'gear_efficiency_lower_bound']) {
    if (!isFiniteNumber(setup[key]) || setup[key] <= 0) errors.push(`setup.${key} must be positive and measured or bounded`);
  }
  if (isFiniteNumber(setup.gear_efficiency_lower_bound) && setup.gear_efficiency_lower_bound > 1) {
    errors.push('setup.gear_efficiency_lower_bound cannot exceed 1');
  }

  const instrumentation = manifest.instrumentation ?? {};
  if (!isFiniteNumber(instrumentation.clock_sync_uncertainty_ms) || instrumentation.clock_sync_uncertainty_ms < 0 || instrumentation.clock_sync_uncertainty_ms > 1) {
    errors.push('instrumentation.clock_sync_uncertainty_ms must be measured and <= 1 ms');
  }
  if (!Array.isArray(instrumentation.calibration_certificates) || instrumentation.calibration_certificates.length === 0) {
    errors.push('instrumentation.calibration_certificates required');
  }
  const sources = manifest.configuration_sources;
  if (!Array.isArray(sources) || sources.length === 0 || sources.some(source => !source.path || !source.role || !hasSha256(source.sha256))) {
    errors.push('configuration_sources need path, role, and SHA-256');
  }
}

function validateLimits(manifest, errors) {
  const limits = manifest.acceptance_limits ?? {};
  const positiveLimits = [
    'bus_max_v', 'near_bridge_a_max_v', 'near_bridge_b_max_v',
    'foc_a_rail_min_v', 'foc_a_rail_max_v', 'foc_b_rail_min_v', 'foc_b_rail_max_v',
    'id_abs_max_a', 'id_abs_max_b', 'torque_min_abs_nm',
    'iq_abs_max_a', 'iq_abs_max_b', 'phase_rms_max_a', 'phase_rms_max_b',
    'dump_temp_max_c',
    'stopped_speed_max_abs_mps', 'required_stop_dwell_s',
    'no_load_abs_current_max_a', 'brownout_rail_trip_max_v', 'brownout_rail_recover_min_v',
  ];
  for (const key of positiveLimits) {
    if (!isFiniteNumber(limits[key]) || limits[key] <= 0) errors.push(`acceptance_limits.${key} must be positive`);
  }
  if (isFiniteNumber(limits.stopped_speed_max_abs_mps) && limits.stopped_speed_max_abs_mps > 0.05) {
    errors.push('acceptance_limits.stopped_speed_max_abs_mps must be <= 0.05 m/s');
  }
  if (isFiniteNumber(limits.foc_a_rail_min_v) && isFiniteNumber(limits.foc_a_rail_max_v) && limits.foc_a_rail_min_v >= limits.foc_a_rail_max_v) {
    errors.push('FOC-A rail minimum must be below its maximum');
  }
  if (isFiniteNumber(limits.foc_b_rail_min_v) && isFiniteNumber(limits.foc_b_rail_max_v) && limits.foc_b_rail_min_v >= limits.foc_b_rail_max_v) {
    errors.push('FOC-B rail minimum must be below its maximum');
  }
  if (isFiniteNumber(limits.brownout_rail_trip_max_v) && isFiniteNumber(limits.brownout_rail_recover_min_v) &&
      limits.brownout_rail_trip_max_v >= limits.brownout_rail_recover_min_v) {
    errors.push('brownout rail trip threshold must be below its recovery threshold');
  }
  if (!Array.isArray(limits.sources) || limits.sources.length === 0 || limits.sources.some(source => !source.path || !source.claim || !hasSha256(source.sha256))) {
    errors.push('acceptance limit source documents need path, claim, and SHA-256');
  }
  for (const key of ['no_load_abs_current_max_a', 'brownout_rail_trip_max_v', 'brownout_rail_recover_min_v']) {
    if (Array.isArray(limits.sources) && !limits.sources.some(source => typeof source.claim === 'string' && source.claim.includes(key))) {
      errors.push(`acceptance limit sources must explicitly cover ${key}`);
    }
  }
  return limits;
}

function validateCase(name, evidence, manifest, evidenceDirectory, limits, errors) {
  if (!evidence || typeof evidence !== 'object') {
    errors.push(`case ${name} missing`);
    return null;
  }
  if (evidence.test_kind !== name) errors.push(`${name}: wrong test_kind`);
  if (evidence.evidence_type !== 'measured' || evidence.synthetic !== false) {
    errors.push(`${name}: must be measured with synthetic=false`);
  }
  if (!evidence.trace_csv || !hasSha256(evidence.sha256)) {
    errors.push(`${name}: CSV path and SHA-256 required`);
    return null;
  }

  const tracePath = path.resolve(evidenceDirectory, evidence.trace_csv);
  if (!tracePath.startsWith(`${path.resolve(evidenceDirectory)}${path.sep}`)) {
    errors.push(`${name}: trace must stay inside evidence folder`);
    return null;
  }
  if (!fs.existsSync(tracePath)) {
    errors.push(`${name}: trace file missing`);
    return null;
  }
  const traceBytes = fs.readFileSync(tracePath);
  const actualHash = crypto.createHash('sha256').update(traceBytes).digest('hex');
  if (actualHash !== evidence.sha256.toLowerCase()) {
    errors.push(`${name}: CSV SHA-256 mismatch`);
    return null;
  }

  try {
    const rows = parseCsv(traceBytes.toString('utf8'), name === 'pump_thermal_required', requiredColumns(name));
    const times = rows.map(row => row.time_s);
    const maxGapS = maxSampleGap(rows);
    const gapLimitS = sampleGapLimit(name);
    if (maxGapS > gapLimitS + 1e-12) errors.push(`${name}: maximum sample gap ${maxGapS}s exceeds ${gapLimitS}s`);
    const hasMotion = BRAKING_WINDOW_CASES.has(name);
    const zeroIndex = hasMotion ? zeroTimeIndex(rows) : -1;
    if (hasMotion && zeroIndex < 1) errors.push(`${name}: requires a pre-trigger sample before one unique t=0 sample`);
    if (hasMotion && zeroIndex >= 1) {
      if (times[0] < -0.010 || times[zeroIndex - 1] >= 0 || Math.abs(rows[zeroIndex].distance_m) > 0.005) {
        errors.push(`${name}: pre-trigger must be within 10 ms and distance at t=0 must be 0±5 mm`);
      }
      if (firstRisingEdge(rows, 'event_marker_signal')?.index !== zeroIndex ||
          observedRisingEdge(rows, 'event_marker_signal')?.index !== zeroIndex) {
        errors.push(`${name}: event marker must have a sampled 0→1 edge at t=0`);
      }
      if (observedRisingEdge(rows, 'brake_request_signal')?.index !== zeroIndex) {
        errors.push(`${name}: brake request must have a sampled 0→1 edge at t=0`);
      }
    }
    if (times.some((time, i) => i > 0 && time <= times[i - 1])) errors.push(`${name}: timestamps must strictly increase`);
    if (rows.some(row => row.bus_v < 0 ||
        row.near_bridge_a_v < 0 || row.near_bridge_b_v < 0 ||
        row.phase_current_a_rms < 0 || row.phase_current_b_rms < 0)) {
      errors.push(`${name}: supply voltages and RMS currents must be nonnegative`);
    }
    if (hasMotion && zeroIndex >= 0) {
      const distanceAtZero = rows[zeroIndex].distance_m;
      const activeRows = rows.slice(zeroIndex);
      if (activeRows.some((row, i) => row.distance_m - distanceAtZero < -1e-9 ||
          (i > 0 && row.distance_m < activeRows[i - 1].distance_m))) {
        errors.push(`${name}: t>=0 distance must be nonnegative and cannot decrease`);
      }
      if (rows.slice(0, zeroIndex).some(row => {
        const relativeDistance = row.distance_m - distanceAtZero;
        return relativeDistance < -0.05 || relativeDistance > 0.005;
      })) {
        errors.push(`${name}: pre-trigger relative distance must remain within -0.05..0.005 m`);
      }
    }
    if (hasMotion && zeroIndex >= 0 && rows.slice(zeroIndex).some(row => row.speed_mps < -0.01)) {
      errors.push(`${name}: reverse travel is outside this forward-stop scope`);
    }

    const maximumAbs = column => rows.reduce((value, row) => Math.max(value, Math.abs(row[column])), 0);
    const maximum = column => rows.reduce((value, row) => Math.max(value, row[column]), -Infinity);
    const minimum = column => rows.reduce((value, row) => Math.min(value, row[column]), Infinity);
    for (const [column, limit] of [
      ['bus_v', limits.bus_max_v],
      ['near_bridge_a_v', limits.near_bridge_a_max_v],
      ['near_bridge_b_v', limits.near_bridge_b_max_v],
    ]) {
      if (isFiniteNumber(limit) && maximum(column) > limit) errors.push(`${name}: ${column} exceeds sourced limit`);
    }

    if (BRAKING_WINDOW_CASES.has(name)) {
      for (const [column, min, max] of [
        ['foc_a_rail_v', limits.foc_a_rail_min_v, limits.foc_a_rail_max_v],
        ['foc_b_rail_v', limits.foc_b_rail_min_v, limits.foc_b_rail_max_v],
      ]) {
        if (isFiniteNumber(min) && isFiniteNumber(max) && rows.some(row => row[column] < min || row[column] > max)) {
          errors.push(`${name}: ${column} outside sourced braking-window range`);
        }
      }
      if (maximumAbs('id_a_a') > limits.id_abs_max_a || maximumAbs('id_b_a') > limits.id_abs_max_b) {
        errors.push(`${name}: absolute Id exceeds per-FOC sourced limit`);
      }
      if (maximumAbs('iq_a_a') > limits.iq_abs_max_a || maximumAbs('iq_b_a') > limits.iq_abs_max_b ||
          maximum('phase_current_a_rms') > limits.phase_rms_max_a ||
          maximum('phase_current_b_rms') > limits.phase_rms_max_b) {
        errors.push(`${name}: Iq or phase RMS exceeds the per-FOC sourced limit`);
      }
    }

    if (name === 'cold_start') {
      if (!isFiniteNumber(evidence.measured_ambient_temp_c) || evidence.measured_ambient_temp_c > 5 ||
          !isFiniteNumber(evidence.cold_soak_hours) || evidence.cold_soak_hours <= 0 || !evidence.power_cycle_record) {
        errors.push('cold_start: measured <=5 C soak and power-cycle record required');
      }
    }
    if (name === 'brownout') {
      const edge = observedRisingEdge(rows, 'brownout_signal', 0);
      if (!edge) {
        errors.push('brownout: sampled post-start 0→1 brownout_signal edge missing');
      } else {
        const rail = evidence.browned_out_rail;
        if (!['foc_a_rail_v', 'foc_b_rail_v'].includes(rail)) {
          errors.push('brownout: browned_out_rail must select a measured FOC rail');
        } else {
        const beforeEdge = rows[edge.index - 1];
        if (!beforeEdge || beforeEdge[rail] < limits.brownout_rail_recover_min_v) {
          errors.push('brownout: selected measured rail must be above the sourced recovery threshold immediately before the event edge');
        }
          const afterEdge = rows.slice(edge.index);
        const assertedSamples = afterEdge.flatMap((row, index) => row.brownout_signal === 1 ? [{ ...row, index: edge.index + index }] : []);
        const dip = assertedSamples.reduce((best, row) => row[rail] < best.value ? { value: row[rail], index: row.index } : best,
          { value: Infinity, index: -1 });
        if (dip.index < 0 || dip.value > limits.brownout_rail_trip_max_v || rows[dip.index]?.brownout_signal !== 1) {
            errors.push('brownout: selected measured rail did not cross the sourced trip threshold during the asserted event');
        } else if (!rows.slice(dip.index + 1).some(row => row[rail] >= limits.brownout_rail_recover_min_v)) {
            errors.push('brownout: selected measured rail did not recover above the sourced recovery threshold');
          }
        }
      }
    }

    if (name === 'low_bus_dual_foc' || STOP_CASES.has(name)) {
      validateBrakingWindow(name, evidence, manifest, limits, rows, errors);
    }
    if (name.startsWith('dual_refusal_')) validateDualRefusal(name, evidence, rows, manifest, errors);
    if (name === 'pump_thermal_required') {
      if (!evidence.matched_pump_on_off_trace_ids || !evidence.thermal_basis_source) {
        errors.push('pump thermal case needs matched trace IDs and traceable thermal basis');
      }
      if (rows.some(row => !isFiniteNumber(row.dump_resistor_temp_c))) {
        errors.push('pump thermal case requires measured resistor temperature in every sample');
      }
      if (maximum('dump_resistor_temp_c') > limits.dump_temp_max_c) {
        errors.push('pump thermal case exceeds the sourced resistor temperature limit');
      }
    }
    if (name === 'load_disturbance' && observedRisingEdge(rows, 'load_step_signal', 0) === null) {
      errors.push('load disturbance needs observed post-start 0→1 load_step_signal edge');
    }
    if (name === 'upstream_cut_no_load') {
      const edge = observedRisingEdge(rows, 'upstream_disconnect_signal', 0);
      if (!edge) {
        errors.push('no-load cut guard needs an observed post-start 0→1 upstream disconnect edge');
      } else {
        const preCutRows = rows.filter(row => row.time_s >= edge.time_s - 0.010 && row.time_s < edge.time_s);
        if (!preCutRows.length || edge.time_s - preCutRows[0].time_s < 0.010 - sampleGapLimit(name) - 1e-12) {
          errors.push('no-load cut needs a sampled 10 ms pre-cut current window');
        } else if (preCutRows.some(row => CURRENT_COLUMNS.some(column => Math.abs(row[column]) > limits.no_load_abs_current_max_a))) {
          errors.push('no-load cut has a measured branch current above the sourced no-load threshold before cut');
        }
      }
    }

    return {
      samples: rows.length,
      duration_s: times.at(-1) - times[0],
      max_sample_gap_s: maxGapS,
      distance_m: hasMotion && zeroIndex >= 0 ? rows.at(-1).distance_m - rows[zeroIndex].distance_m : null,
      terminal_speed_mps: hasMotion ? rows.at(-1).speed_mps : null,
      bus_peak_v: maximum('bus_v'),
      near_bridge_a_peak_v: maximum('near_bridge_a_v'),
      near_bridge_b_peak_v: maximum('near_bridge_b_v'),
      foc_a_rail_min_v: minimum('foc_a_rail_v'),
      foc_b_rail_min_v: minimum('foc_b_rail_v'),
      id_a_abs_peak_a: maximumAbs('id_a_a'),
      id_b_abs_peak_a: maximumAbs('id_b_a'),
      torque_a_abs_peak_nm: maximumAbs('torque_a_nm'),
      torque_b_abs_peak_nm: maximumAbs('torque_b_nm'),
      torque_a_range_nm: [minimum('torque_a_nm'), maximum('torque_a_nm')],
      torque_b_range_nm: [minimum('torque_b_nm'), maximum('torque_b_nm')],
    };
  } catch (error) {
    errors.push(`${name}: CSV parse error: ${error.message}`);
    return null;
  }
}

function validateBrakingWindow(name, evidence, manifest, limits, rows, errors) {
  const sign = manifest.setup?.braking_torque_sign;
  const stopSpeed = Math.min(limits.stopped_speed_max_abs_mps ?? 0.05, 0.05);
  const zeroIndex = zeroTimeIndex(rows);
  if (zeroIndex < 1) return;
  const activeRows = rows.slice(zeroIndex);
  const firstActive = activeRows[0];
  if (!isFiniteNumber(evidence.buildup_end_s) || evidence.buildup_end_s < 0) {
    errors.push(`${name}: measured buildup_end_s must be finite and >= 0`);
    return;
  }
  const movingRows = activeRows.filter(row => Math.abs(row.speed_mps) > stopSpeed);
  if (movingRows.length === 0 || evidence.buildup_end_s > movingRows.at(-1).time_s) {
    errors.push(`${name}: buildup_end_s must fall within the recorded moving interval`);
    return;
  }

  if (name === 'low_bus_dual_foc') {
    if (!isFiniteNumber(evidence.bus_start_tolerance_v) || evidence.bus_start_tolerance_v < 0 || evidence.bus_start_tolerance_v > 0.2) {
      errors.push('low_bus_dual_foc: bus_start_tolerance_v must be measured and between 0 and 0.2 V');
    } else if (Math.abs(firstActive.bus_v - 22.5) > evidence.bus_start_tolerance_v) {
      errors.push('low_bus_dual_foc: must start at measured 22.5 V within documented tolerance');
    }
  }

  if (STOP_CASES.has(name)) {
    if (firstActive.speed_mps < 15 / 3.6) errors.push(`${name}: measured initial forward speed must be at least 15 km/h`);
    if (!isFiniteNumber(evidence.distance_uncertainty_m) || evidence.distance_uncertainty_m < 0 ||
        rows.at(-1).distance_m - firstActive.distance_m + evidence.distance_uncertainty_m > 3) {
      errors.push(`${name}: measured stopping distance plus uncertainty must be <= 3.000 m`);
    }
    let firstSettledIndex = activeRows.length - 1;
    while (firstSettledIndex > 0 && Math.abs(activeRows[firstSettledIndex - 1].speed_mps) <= stopSpeed) firstSettledIndex--;
    const dwell = activeRows.at(-1).time_s - activeRows[firstSettledIndex].time_s;
    if (Math.abs(activeRows.at(-1).speed_mps) > stopSpeed || dwell < limits.required_stop_dwell_s) {
      errors.push(`${name}: must stay continuously at |speed| <= ${stopSpeed} m/s for ${limits.required_stop_dwell_s} s`);
    }
  }

  const measuredBrakeSamples = activeRows.filter(row => row.time_s > 0 && row.time_s >= evidence.buildup_end_s && Math.abs(row.speed_mps) > stopSpeed);
  if (measuredBrakeSamples.length === 0) errors.push(`${name}: no post-buildup moving samples`);
  for (const row of measuredBrakeSamples) {
    if (sign * row.torque_a_nm < limits.torque_min_abs_nm || sign * row.torque_b_nm < limits.torque_min_abs_nm) {
      errors.push(`${name}: signed A/B braking torque below sourced minimum during moving interval`);
      break;
    }
  }
}

function validateDualRefusal(name, evidence, rows, manifest, errors) {
  const stopSpeed = Math.min(manifest.acceptance_limits?.stopped_speed_max_abs_mps ?? 0.05, 0.05);
  const zeroIndex = zeroTimeIndex(rows);
  if (zeroIndex < 1 || !isFiniteNumber(evidence.buildup_end_s)) return;
  const activeRows = rows.slice(zeroIndex);
  let firstSettledIndex = activeRows.length - 1;
  while (firstSettledIndex > 0 && Math.abs(activeRows[firstSettledIndex - 1].speed_mps) <= stopSpeed) firstSettledIndex--;
  const effectiveStart = activeRows.findIndex(row => row.time_s >= evidence.buildup_end_s);
  if (effectiveStart < 0) {
    errors.push(`${name}: no samples in effective braking interval`);
    return;
  }

  const main = observedRisingEdge(rows, 'upstream_disconnect_signal', 0);
  const oldRails = observedRisingEdge(rows, 'old_rails_lost_signal', 0);
  const firstDisconnectIndex = [main, oldRails]
    .filter(Boolean)
    .reduce((first, edge) => Math.min(first, edge.index - zeroIndex), firstSettledIndex);
  const refusalStart = Math.min(effectiveStart, firstDisconnectIndex);

  for (const channel of ['battery_refusal_signal', 'supercap_refusal_signal']) {
    const refusalRows = activeRows.slice(refusalStart, firstSettledIndex + 1);
    if (!refusalRows.length || refusalRows.some(row => row[channel] !== 1)) {
      errors.push(`${name}: ${channel} must stay asserted from the first effective-braking/disconnect boundary through stop`);
    }
  }

  for (const [channel, edge] of [['upstream_disconnect_signal', main], ['old_rails_lost_signal', oldRails]]) {
    if (!edge) {
      errors.push(`${name}: observed post-request 0→1 ${channel} edge missing`);
      continue;
    }
    if (edge.index < zeroIndex || edge.index > zeroIndex + firstSettledIndex || Math.abs(rows[edge.index].speed_mps) <= stopSpeed) {
      errors.push(`${name}: ${channel} must rise at/after t=0 while still moving`);
    }
    if (activeRows.slice(edge.index - zeroIndex, firstSettledIndex + 1).some(row => row[channel] !== 1)) {
      errors.push(`${name}: ${channel} must remain asserted through stop`);
    }
  }
  if (!main || !oldRails) return;

  // clock_sync_uncertainty_ms is the maximum pairwise relative event-time error.
  // Add one measured sample interval for edge quantization.
  const timingUncertaintyS = (manifest.instrumentation?.clock_sync_uncertainty_ms ?? 0) / 1000 + maxSampleGap(rows);
  const separationS = Math.abs(main.time_s - oldRails.time_s);
  if (name === 'dual_refusal_main_first' && oldRails.time_s - main.time_s <= timingUncertaintyS) {
    errors.push(`${name}: upstream disconnect must precede old-rails loss by more than timing uncertainty`);
  } else if (name === 'dual_refusal_old_rails_first' && main.time_s - oldRails.time_s <= timingUncertaintyS) {
    errors.push(`${name}: old-rails loss must precede upstream disconnect by more than timing uncertainty`);
  } else if (name === 'dual_refusal_simultaneous' && separationS > timingUncertaintyS) {
    errors.push(`${name}: event-time difference exceeds clock uncertainty plus sample quantization`);
  }
}

function validateSourceFiles(sources, directory, errors) {
  for (const source of sources ?? []) {
    if (!source?.path || !hasSha256(source.sha256)) continue;
    const sourcePath = path.resolve(directory, source.path);
    if (!sourcePath.startsWith(`${path.resolve(directory)}${path.sep}`)) {
      errors.push(`source must stay in evidence folder: ${source.path}`);
      continue;
    }
    try {
      const digest = crypto.createHash('sha256').update(fs.readFileSync(sourcePath)).digest('hex');
      if (digest !== source.sha256.toLowerCase()) errors.push(`source SHA-256 mismatch: ${source.path}`);
    } catch (error) {
      errors.push(`source unreadable: ${source.path}: ${error.message}`);
    }
  }
}

function readVerifiedEvidenceFile(relativePath, sha256, directory, label, errors) {
  if (typeof relativePath !== 'string' || !relativePath || !hasSha256(sha256)) {
    errors.push(`${label}: relative path and SHA-256 required`);
    return null;
  }
  const sourcePath = path.resolve(directory, relativePath);
  if (!sourcePath.startsWith(`${path.resolve(directory)}${path.sep}`)) {
    errors.push(`${label}: file must stay inside evidence folder`);
    return null;
  }
  try {
    const bytes = fs.readFileSync(sourcePath);
    const digest = crypto.createHash('sha256').update(bytes).digest('hex');
    if (digest !== sha256.toLowerCase()) {
      errors.push(`${label}: SHA-256 mismatch`);
      return null;
    }
    return bytes;
  } catch (error) {
    errors.push(`${label}: file unreadable: ${error.message}`);
    return null;
  }
}

function validatePropagationAttachment(sources, directory, limits, errors) {
  const role = 'foc_bms_propagation_local_rail_reset';
  const matches = (sources ?? []).filter(source => source?.role === role);
  if (matches.length !== 1) {
    errors.push(`configuration_sources must contain exactly one ${role} attachment`);
    return null;
  }
  const attachmentBytes = readVerifiedEvidenceFile(matches[0].path, matches[0].sha256, directory, role, errors);
  if (!attachmentBytes) return null;
  let attachment;
  try {
    attachment = JSON.parse(attachmentBytes.toString('utf8'));
  } catch (error) {
    errors.push(`${role}: attachment JSON parse error: ${error.message}`);
    return null;
  }
  if (attachment.schema !== 'agtc-foc-bms-propagation-local-reset-v1' ||
      attachment.evidence_type !== 'measured' || attachment.synthetic !== false) {
    errors.push(`${role}: measured v1 attachment with synthetic=false required`);
    return null;
  }
  if (!isFiniteNumber(attachment.clock_sync_uncertainty_ms) || attachment.clock_sync_uncertainty_ms < 0 ||
      attachment.clock_sync_uncertainty_ms > 1) {
    errors.push(`${role}: clock_sync_uncertainty_ms must be measured and <=1 ms`);
    return null;
  }
  if (!isFiniteNumber(attachment.main_current_active_min_a) ||
      !isFiniteNumber(attachment.main_current_open_max_a) ||
      attachment.main_current_active_min_a <= attachment.main_current_open_max_a ||
      attachment.main_current_open_max_a < 0 ||
      !isFiniteNumber(attachment.pc10_high_min_v) || attachment.pc10_high_min_v <= 0) {
    errors.push(`${role}: measured main-current and PC10 thresholds required`);
    return null;
  }
  if (!Array.isArray(attachment.runs)) {
    errors.push(`${role}: runs array required`);
    return null;
  }
  const runsByTemperature = new Map();
  for (const run of attachment.runs) {
    if (!isFiniteNumber(run.temperature_c) || run.temperature_c < 0 || run.temperature_c > 60 ||
        runsByTemperature.has(run.temperature_c)) {
      errors.push(`${role}: each run needs a unique measured temperature in 0..60 C`);
      continue;
    }
    runsByTemperature.set(run.temperature_c, run);
  }
  for (const temperature of [0, 60]) {
    if (!runsByTemperature.has(temperature)) errors.push(`${role}: measured ${temperature} C run required`);
  }

  const summaries = [];
  for (const [temperature, run] of runsByTemperature) {
    const label = `${role} ${temperature} C`;
    const mainBytes = readVerifiedEvidenceFile(run.main_cut_trace_csv, run.main_cut_trace_sha256, directory, `${label} main-cut trace`, errors);
    const resetBytes = readVerifiedEvidenceFile(run.local_reset_trace_csv, run.local_reset_trace_sha256, directory, `${label} local-reset trace`, errors);
    if (!mainBytes || !resetBytes) continue;
    let mainRows;
    let resetRows;
    try {
      mainRows = parseCsv(mainBytes.toString('utf8'), false, [
        'time_s', 'main_path_current_a', 'bq_alert_active', 'main_clr_active',
        'fet_release_active', 'load_power_valid_active', 'pc10_v',
        'software_fault_latched', 'traction_inhibited', 'foc_local_rail_v',
      ]);
      resetRows = parseCsv(resetBytes.toString('utf8'), false, [
        'time_s', 'local_reset_active', 'pc6_hold_active', 'pwm_enabled',
      ]);
    } catch (error) {
      errors.push(`${label}: trace CSV parse error: ${error.message}`);
      continue;
    }
    for (const [kind, rows] of [['main-cut', mainRows], ['local-reset', resetRows]]) {
      if (rows.some((row, index) => index > 0 && row.time_s <= rows[index - 1].time_s)) {
        errors.push(`${label} ${kind}: timestamps must strictly increase`);
      }
      if (maxSampleGap(rows) > BROWNOUT_SAMPLE_GAP_S + 1e-12) {
        errors.push(`${label} ${kind}: sample gap must be <=100 us for a 1 ms timing claim`);
      }
    }
    const mainCutIndex = mainRows.findIndex((row, index) => index > 0 &&
      Math.abs(row.main_path_current_a) <= attachment.main_current_open_max_a &&
      mainRows.slice(0, index).some(previous => Math.abs(previous.main_path_current_a) >= attachment.main_current_active_min_a));
    if (mainCutIndex < 0) {
      errors.push(`${label}: measured main-path current does not transition from loaded to interrupted`);
      continue;
    }
    if (mainRows.slice(mainCutIndex).some(row => Math.abs(row.main_path_current_a) > attachment.main_current_open_max_a)) {
      errors.push(`${label}: main-path current rises above its open threshold after interruption`);
    }
    const mainCutTime = mainRows[mainCutIndex].time_s;
    const afterCutRows = mainRows.slice(mainCutIndex);
    if (isFiniteNumber(limits.foc_a_rail_min_v) && isFiniteNumber(limits.foc_a_rail_max_v) &&
        afterCutRows.some(row => row.foc_local_rail_v < limits.foc_a_rail_min_v || row.foc_local_rail_v > limits.foc_a_rail_max_v)) {
      errors.push(`${label}: local FOC rail leaves the sourced operating window after main-current interruption`);
    }
    const edges = [
      ['bq_alert_active', observedRisingEdge(mainRows, 'bq_alert_active')],
      ['main_clr_active', observedRisingEdge(mainRows, 'main_clr_active')],
      ['fet_release_active', observedFallingEdge(mainRows, 'fet_release_active')],
      ['load_power_valid_active', observedFallingEdge(mainRows, 'load_power_valid_active')],
      ['pc10_v', observedThresholdRisingEdge(mainRows, 'pc10_v', attachment.pc10_high_min_v)],
      ['software_fault_latched', observedRisingEdge(mainRows, 'software_fault_latched')],
      ['traction_inhibited', observedRisingEdge(mainRows, 'traction_inhibited')],
    ];
    for (const [channel, edge] of edges) if (!edge) errors.push(`${label}: observed transition missing for ${channel}`);
    const edgeList = edges.map(([, edge]) => edge);
    if (edgeList.every(Boolean)) {
      if (edgeList.some((edge, index) => index > 0 && edge.time_s < edgeList[index - 1].time_s)) {
        errors.push(`${label}: BQ alert→CLR→FET_RELEASE→LPV→PC10→software timing order is invalid`);
      }
      if (mainRows.slice(edgeList[4].index).some(row => row.pc10_v < attachment.pc10_high_min_v) ||
          mainRows.slice(edgeList[5].index).some(row => row.software_fault_latched !== 1) ||
          mainRows.slice(edgeList[6].index).some(row => row.traction_inhibited !== 1) ||
          mainRows.slice(edgeList[2].index).some(row => row.fet_release_active !== 0) ||
          mainRows.slice(edgeList[3].index).some(row => row.load_power_valid_active !== 0)) {
        errors.push(`${label}: fault and revoke states must persist after measured transitions`);
      }
    }
    const uncertaintyMs = attachment.clock_sync_uncertainty_ms + maxSampleGap(mainRows) * 1000;
    const pc10DelayMs = edgeList[4] ? (edgeList[4].time_s - mainCutTime) * 1000 : Infinity;
    const latchDelayMs = edgeList[4] && edgeList[5] ? (edgeList[5].time_s - edgeList[4].time_s) * 1000 : Infinity;
    const inhibitDelayMs = edgeList[4] && edgeList[6] ? (edgeList[6].time_s - edgeList[4].time_s) * 1000 : Infinity;
    if (pc10DelayMs < 0 || pc10DelayMs + uncertaintyMs > 1) errors.push(`${label}: main-current interruption→PC10 rise plus uncertainty must be <=1 ms`);
    if (latchDelayMs < 0 || latchDelayMs + uncertaintyMs > 1) errors.push(`${label}: PC10 rise→software fault latch plus uncertainty must be <=1 ms`);
    if (inhibitDelayMs < 0 || inhibitDelayMs + uncertaintyMs > 1) errors.push(`${label}: PC10 rise→traction inhibit plus uncertainty must be <=1 ms`);

    const resetEdge = observedRisingEdge(resetRows, 'local_reset_active');
    const pc6Fall = resetEdge ? observedFallingEdge(resetRows, 'pc6_hold_active', resetEdge.time_s) : null;
    const pwmFall = resetEdge ? observedFallingEdge(resetRows, 'pwm_enabled', resetEdge.time_s) : null;
    if (!resetEdge || !resetRows.slice(0, resetEdge.index).some(row => row.pc6_hold_active === 1 && row.pwm_enabled === 1)) {
      errors.push(`${label}: local-reset trace must show an armed state before a measured reset edge`);
    }
    if (!pc6Fall || !pwmFall) {
      errors.push(`${label}: local reset must show PC6 hold and PWM falling low`);
    } else {
      const resetUncertaintyMs = attachment.clock_sync_uncertainty_ms + maxSampleGap(resetRows) * 1000;
      if ((pc6Fall.time_s - resetEdge.time_s) * 1000 + resetUncertaintyMs > 1 ||
          (pwmFall.time_s - resetEdge.time_s) * 1000 + resetUncertaintyMs > 1) {
        errors.push(`${label}: reset→PC6-low and PWM-safe plus uncertainty must be <=1 ms`);
      }
      if (resetRows.slice(pc6Fall.index).some(row => row.pc6_hold_active !== 0) ||
          resetRows.slice(pwmFall.index).some(row => row.pwm_enabled !== 0)) {
        errors.push(`${label}: PC6 hold and PWM must remain low after reset through the recorded interval`);
      }
    }
    summaries.push({
      temperature_c: temperature,
      main_cut_max_sample_gap_s: maxSampleGap(mainRows),
      local_reset_max_sample_gap_s: maxSampleGap(resetRows),
      main_to_pc10_ms: pc10DelayMs,
      pc10_to_fault_latch_ms: latchDelayMs,
      pc10_to_traction_inhibit_ms: inhibitDelayMs,
      uncertainty_ms: uncertaintyMs,
    });
  }
  return summaries;
}

function validateManifest(file) {
  let manifest;
  try {
    manifest = JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch (error) {
    return hold([`cannot read evidence JSON: ${error.message}`]);
  }
  if (manifest?.schema !== 'agtc-parking-handoff-evidence-v3') return hold(['wrong or missing schema']);

  const errors = [];
  const summaries = {};
  let electricalAttachmentSummary = null;
  if (manifest.evidence_class !== 'measured_vehicle_evidence') errors.push('evidence_class must be measured_vehicle_evidence');
  if (manifest.status !== 'SUBMITTED_FOR_REVIEW') errors.push('manifest status must be SUBMITTED_FOR_REVIEW');
  validateSetup(manifest, errors);
  const limits = validateLimits(manifest, errors);
  if (Array.isArray(manifest.configuration_sources)) {
    validateSourceFiles(manifest.configuration_sources, path.dirname(file), errors);
    electricalAttachmentSummary = validatePropagationAttachment(manifest.configuration_sources, path.dirname(file), limits, errors);
  } else {
    validatePropagationAttachment([], path.dirname(file), limits, errors);
  }
  if (Array.isArray(limits.sources)) validateSourceFiles(limits.sources, path.dirname(file), errors);

  for (const name of CASES) {
    const summary = validateCase(name, manifest.cases?.[name], manifest, path.dirname(file), limits, errors);
    if (summary) summaries[name] = summary;
  }
  if (errors.length) {
    const result = hold(errors, summaries);
    result.electrical_attachment_summary = electricalAttachmentSummary;
    return result;
  }
  return {
    status: 'EVIDENCE_READY_FOR_REPLAY',
    qualification: 'EVIDENCE_READY_FOR_REPLAY',
    errors: [],
    case_summaries: summaries,
    electrical_attachment_summary: electricalAttachmentSummary,
    scope: 'Evidence intake checks only. No model is executed, no PCB or vehicle PASS is issued, and no configuration is released.',
  };
}

const args = process.argv.slice(2);
let inputPath = path.join(HERE, '停车交接证据模板.json');
let outputPath = null;
for (let index = 0; index < args.length; index++) {
  if (args[index] === '--output') outputPath = args[++index];
  else inputPath = path.resolve(args[index]);
}

let result;
try {
  result = validateManifest(inputPath);
} catch (error) {
  result = hold([`invalid evidence structure: ${error.message}`]);
}
const output = `${JSON.stringify(result, null, 2)}\n`;
if (outputPath) fs.writeFileSync(path.resolve(outputPath), output);
else process.stdout.write(output);
process.exitCode = result.status === 'HOLD' ? 2 : 0;
