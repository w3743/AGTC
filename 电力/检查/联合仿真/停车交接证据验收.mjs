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
];
const DIGITAL_COLUMNS = new Set([
  'pump_state', 'brake_request_signal', 'event_marker_signal',
  'battery_refusal_signal', 'supercap_refusal_signal',
  'upstream_disconnect_signal', 'old_rails_lost_signal',
  'brownout_signal', 'load_step_signal',
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

function parseCsv(text, requireResistorTemperature) {
  const lines = text.replace(/^\uFEFF/, '').trim().split(/\r?\n/);
  if (lines.length < 2) throw new Error('needs at least two samples');

  const headers = lines[0].split(',').map(value => value.trim());
  const missing = COLUMNS.filter(column => !headers.includes(column));
  if (missing.length) throw new Error(`columns missing: ${missing.join(', ')}`);

  return lines.slice(1).map(line => {
    const cells = line.split(',');
    const row = {};
    headers.forEach((header, index) => { row[header] = cells[index]?.trim() ?? ''; });

    for (const column of COLUMNS) {
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
  if (!Array.isArray(limits.sources) || limits.sources.length === 0 || limits.sources.some(source => !source.path || !source.claim || !hasSha256(source.sha256))) {
    errors.push('acceptance limit source documents need path, claim, and SHA-256');
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
    const rows = parseCsv(traceBytes.toString('utf8'), name === 'pump_thermal_required');
    const times = rows.map(row => row.time_s);
    if (Math.abs(times[0]) > 1e-6 || Math.abs(rows[0].distance_m) > 0.005) {
      errors.push(`${name}: t=0 and distance=0 must align to the measured event marker`);
    }
    if (times.some((time, i) => i > 0 && time <= times[i - 1])) errors.push(`${name}: timestamps must strictly increase`);
    if (rows.some(row => row.distance_m < 0 || row.bus_v < 0 ||
        row.near_bridge_a_v < 0 || row.near_bridge_b_v < 0 ||
        row.phase_current_a_rms < 0 || row.phase_current_b_rms < 0)) {
      errors.push(`${name}: distance, supply voltages and RMS currents must be nonnegative`);
    }
    if (rows.some((row, i) => i > 0 && row.distance_m < rows[i - 1].distance_m)) {
      errors.push(`${name}: measured forward travel distance cannot decrease`);
    }
    if (rows.some(row => row.speed_mps < -0.01)) errors.push(`${name}: reverse travel is outside this forward-stop scope`);
    if (firstRisingEdge(rows, 'event_marker_signal')?.time_s !== 0) errors.push(`${name}: event_marker_signal must rise at t=0`);

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
    if (name === 'brownout' && firstRisingEdge(rows, 'brownout_signal') === null) {
      errors.push('brownout: timestamped brownout_signal edge missing');
    }

    if (name === 'low_bus_dual_foc' || STOP_CASES.has(name)) {
      validateBrakingWindow(name, evidence, manifest, limits, rows, errors);
    }
    if (name.startsWith('dual_refusal_')) validateDualRefusal(name, rows, manifest, errors);
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
    if (name === 'load_disturbance' && firstRisingEdge(rows, 'load_step_signal') === null) {
      errors.push('load disturbance needs timestamped load_step_signal edge');
    }
    if (name === 'upstream_cut_no_load' && firstRisingEdge(rows, 'upstream_disconnect_signal') === null) {
      errors.push('no-load cut guard needs timestamped upstream disconnect edge');
    }

    return {
      samples: rows.length,
      duration_s: times.at(-1) - times[0],
      distance_m: rows.at(-1).distance_m,
      terminal_speed_mps: rows.at(-1).speed_mps,
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
  if (!isFiniteNumber(evidence.buildup_end_s) || evidence.buildup_end_s < 0) {
    errors.push(`${name}: measured buildup_end_s must be finite and >= 0`);
    return;
  }
  const movingRows = rows.filter(row => Math.abs(row.speed_mps) > stopSpeed);
  if (movingRows.length === 0 || evidence.buildup_end_s > movingRows.at(-1).time_s) {
    errors.push(`${name}: buildup_end_s must fall within the recorded moving interval`);
    return;
  }

  if (name === 'low_bus_dual_foc') {
    if (!isFiniteNumber(evidence.bus_start_tolerance_v) || evidence.bus_start_tolerance_v < 0 || evidence.bus_start_tolerance_v > 0.2) {
      errors.push('low_bus_dual_foc: bus_start_tolerance_v must be measured and between 0 and 0.2 V');
    } else if (Math.abs(rows[0].bus_v - 22.5) > evidence.bus_start_tolerance_v) {
      errors.push('low_bus_dual_foc: must start at measured 22.5 V within documented tolerance');
    }
  }

  if (STOP_CASES.has(name)) {
    if (firstRisingEdge(rows, 'brake_request_signal')?.time_s !== 0) errors.push(`${name}: brake request must rise at t=0`);
    if (rows[0].speed_mps < 4.166) errors.push(`${name}: measured initial forward speed must be at least 15 km/h`);
    if (!isFiniteNumber(evidence.distance_uncertainty_m) || evidence.distance_uncertainty_m < 0 ||
        rows.at(-1).distance_m + evidence.distance_uncertainty_m > 3) {
      errors.push(`${name}: measured stopping distance plus uncertainty must be <= 3.000 m`);
    }
    let firstSettledIndex = rows.length - 1;
    while (firstSettledIndex > 0 && Math.abs(rows[firstSettledIndex - 1].speed_mps) <= stopSpeed) firstSettledIndex--;
    const dwell = rows.at(-1).time_s - rows[firstSettledIndex].time_s;
    if (Math.abs(rows.at(-1).speed_mps) > stopSpeed || dwell < limits.required_stop_dwell_s) {
      errors.push(`${name}: must stay continuously at |speed| <= ${stopSpeed} m/s for ${limits.required_stop_dwell_s} s`);
    }
  }

  const measuredBrakeSamples = rows.filter(row => row.time_s > 0 && row.time_s >= evidence.buildup_end_s && Math.abs(row.speed_mps) > stopSpeed);
  if (measuredBrakeSamples.length === 0) errors.push(`${name}: no post-buildup moving samples`);
  for (const row of measuredBrakeSamples) {
    if (sign * row.torque_a_nm < limits.torque_min_abs_nm || sign * row.torque_b_nm < limits.torque_min_abs_nm) {
      errors.push(`${name}: signed A/B braking torque below sourced minimum during moving interval`);
      break;
    }
  }
}

function validateDualRefusal(name, rows, manifest, errors) {
  const stopSpeed = Math.min(manifest.acceptance_limits?.stopped_speed_max_abs_mps ?? 0.05, 0.05);
  for (const channel of ['battery_refusal_signal', 'supercap_refusal_signal', 'upstream_disconnect_signal', 'old_rails_lost_signal']) {
    const edge = firstRisingEdge(rows, channel);
    if (!edge) {
      errors.push(`${name}: timestamped ${channel} edge missing`);
    } else if (channel.endsWith('_refusal_signal') &&
        (rows[edge.index].brake_request_signal !== 1 || Math.abs(rows[edge.index].speed_mps) <= stopSpeed)) {
      errors.push(`${name}: ${channel} must be observed during requested braking while moving`);
    }
  }

  const main = firstRisingEdge(rows, 'upstream_disconnect_signal');
  const oldRails = firstRisingEdge(rows, 'old_rails_lost_signal');
  if (!main || !oldRails) return;
  const simultaneityWindowS = (manifest.instrumentation?.clock_sync_uncertainty_ms ?? 0) / 1000;
  if (name === 'dual_refusal_main_first' && main.time_s >= oldRails.time_s) {
    errors.push(`${name}: upstream disconnect must precede old-rails loss`);
  } else if (name === 'dual_refusal_old_rails_first' && oldRails.time_s >= main.time_s) {
    errors.push(`${name}: old-rails loss must precede upstream disconnect`);
  } else if (name === 'dual_refusal_simultaneous' && Math.abs(main.time_s - oldRails.time_s) > simultaneityWindowS) {
    errors.push(`${name}: upstream disconnect and old-rails loss must be simultaneous within clock uncertainty`);
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

function validateManifest(file) {
  let manifest;
  try {
    manifest = JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch (error) {
    return hold([`cannot read evidence JSON: ${error.message}`]);
  }
  if (manifest?.schema !== 'agtc-parking-handoff-evidence-v2') return hold(['wrong or missing schema']);

  const errors = [];
  const summaries = {};
  if (manifest.evidence_class !== 'measured_vehicle_evidence') errors.push('evidence_class must be measured_vehicle_evidence');
  if (manifest.status !== 'SUBMITTED_FOR_REVIEW') errors.push('manifest status must be SUBMITTED_FOR_REVIEW');
  validateSetup(manifest, errors);
  const limits = validateLimits(manifest, errors);
  if (Array.isArray(manifest.configuration_sources)) {
    validateSourceFiles(manifest.configuration_sources, path.dirname(file), errors);
  }
  if (Array.isArray(limits.sources)) validateSourceFiles(limits.sources, path.dirname(file), errors);

  for (const name of CASES) {
    const summary = validateCase(name, manifest.cases?.[name], manifest, path.dirname(file), limits, errors);
    if (summary) summaries[name] = summary;
  }
  if (errors.length) return hold(errors, summaries);
  return {
    status: 'EVIDENCE_READY_FOR_REPLAY',
    qualification: 'EVIDENCE_READY_FOR_REPLAY',
    errors: [],
    case_summaries: summaries,
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
