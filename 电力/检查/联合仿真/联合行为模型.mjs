// Four-board reduced-order behavior model. Run from the repository root with:
// node 电力/检查/联合仿真/联合行为模型.mjs
// This is not a switched circuit or hardware qualification simulation.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '../../..');
const inputPath = path.join(HERE, '联合行为模型输入.json');
const input = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
const p = input.parameters;
const sha = (b) => crypto.createHash('sha256').update(b).digest('hex');
const sourceLedger = input.sources.map((s) => {
  const abs = path.resolve(ROOT, s.path);
  if (!fs.existsSync(abs)) return { ...s, exists: false, sha256: null,
    source_status: 'MISSING_OR_HISTORICAL_PATH' };
  const h = sha(fs.readFileSync(abs));
  const baselineMatch = s.baseline_sha256 ? h.toLowerCase() === s.baseline_sha256.toLowerCase() : null;
  return { ...s, exists: true, sha256: h, baseline_match: baselineMatch,
    source_status: baselineMatch === false ? 'SOURCE_CHANGED_SINCE_OLD_BINDING' : 'READ_AND_HASHED' };
});
const writeCsv = (name, header, rows) => {
  const esc = (v) => {
    if (v === null || v === undefined) return '';
    const s = String(v);
    return /[",\r\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
  };
  fs.writeFileSync(path.join(HERE, name), [header.map(esc).join(','), ...rows.map(r => r.map(esc).join(','))].join('\n') + '\n', 'utf8');
};
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

function rcConstantLoad(packV, r, c, loadA, t, startV = 0) {
  const gapSteady = r * loadA;
  const gap = gapSteady + (packV - startV - gapSteady) * Math.exp(-t / (r * c));
  const targetTime = gapSteady >= p.precharge.legacy_gap_comparison_v ? null :
    r * c * Math.log((packV - startV - gapSteady) /
      (p.precharge.legacy_gap_comparison_v - gapSteady));
  return { t_s: t, bus_v: packV - gap, gap_v: gap, time_to_0p5_v_s: targetTime };
}

function simulateRcAux({ packV, r, c, backgroundA, regW, pumpW, duration, dt = 0.0001 }) {
  let v = 0;
  let targetTime = null;
  const rows = [];
  const steps = Math.ceil(duration / dt);
  dt = duration / steps;
  const loadA = (bus) => {
    const frac = clamp(bus - p.precharge.aux_start_threshold_v, 0, p.precharge.aux_ramp_width_v);
    const power = (regW + pumpW) * frac;
    return backgroundA + power / Math.max(1, bus);
  };
  const dv = (bus) => ((packV - bus) / r - loadA(bus)) / c;
  for (let k = 0; k <= steps; k++) {
    const t = Math.min(k * dt, duration);
    if (targetTime === null && packV - v <= p.precharge.legacy_gap_comparison_v) targetTime = t;
    if (k === steps || k % Math.max(1, Math.round(0.001 / dt)) === 0) {
      rows.push([t, v, packV - v, (packV - v) / r, loadA(v), regW > 0 && v >= 19 ? 1 : 0, pumpW > 0 && v >= 19 ? 1 : 0]);
    }
    if (k === steps) break;
    const a = dv(v), b = dv(v + a * dt / 2), c1 = dv(v + b * dt / 2), d = dv(v + c1 * dt);
    v += dt * (a + 2*b + 2*c1 + d) / 6;
  }
  return { end_bus_v: v, end_gap_v: packV - v, time_to_0p5_v_s: targetTime,
    peak_noncapacitive_load_a: Math.max(...rows.map(r => r[4])), waveform: rows };
}

function simulateTps({ loadA, regW = 0, pumpW = 0, bgA = 0, c, targetGap, dt = 0.0001 }) {
  const packV = 37.8, r = p.precharge.tps2490_candidate_series_resistance_ohm;
  const ilim = p.precharge.tps2490_candidate_current_min_a;
  let gap = packV, time = 0, peakLoad = 0, peakCharge = 0;
  const limit = 10;
  const load = (g) => {
    const bus = Math.max(1, packV - g);
    const frac = clamp(bus - p.precharge.aux_start_threshold_v, 0, p.precharge.aux_ramp_width_v);
    return loadA + bgA + (regW + pumpW) * frac / bus;
  };
  const dg = (g) => (load(g) - Math.min(ilim, Math.max(0, g / r))) / c;
  while (time < limit && gap > targetGap) {
    const h = Math.min(dt, limit - time);
    peakLoad = Math.max(peakLoad, load(gap));
    peakCharge = Math.max(peakCharge, Math.min(ilim, gap / r));
    const a = dg(gap), b = dg(gap + a*h/2), c1 = dg(gap + b*h/2), d = dg(gap + c1*h);
    const next = gap + h * (a + 2*b + 2*c1 + d) / 6;
    if (gap > targetGap && next <= targetGap) {
      const frac = (gap - targetGap) / (gap - next);
      time += h * frac;
      gap = targetGap;
      break;
    }
    gap = next;
    time += h;
  }
  const reached = gap <= targetGap + 1e-9;
  const timing = p.precharge.tps2490_startup_reserve_s +
    p.precharge.main_proof_time_components_s.stability +
    p.precharge.main_proof_time_components_s.absolute_readback +
    p.precharge.main_proof_time_components_s.handoff;
  return { reached_target: reached, charge_time_s: reached ? time : null,
    full_candidate_sequence_s: reached ? time + timing : null,
    minimum_window_margin_s: reached ? p.precharge.tps2490_candidate_window_s[0] - (time + timing) : null,
    peak_noncapacitive_current_a: peakLoad, peak_precharge_current_a: peakCharge,
    target_gap_v: targetGap, source_resistance_ohm: r, current_limit_a: ilim,
    fixed_sequence_overhead_s: timing };
}

function mechanicalStopProfile({ slopeDeg = 0, stopDistanceM = 3, torqueBuildDelayS = 0 }) {
  const m = input.frozen_configuration.vehicle_mass_kg;
  const v0 = input.frozen_configuration.vehicle_speed_kmh / 3.6;
  const distanceAfterBuildM = stopDistanceM - v0 * torqueBuildDelayS;
  if (distanceAfterBuildM <= 0) throw new Error('Torque build delay consumes the stopping distance.');
  const a = v0 * v0 / (2 * distanceAfterBuildM);
  const stopS = torqueBuildDelayS + v0 / a;
  const slope = Math.sin(slopeDeg * Math.PI / 180);
  return { m, v0, a, stopS, powerAt: (t) => {
    if (t < torqueBuildDelayS) return 0;
    const v = Math.max(0, v0 - a * (t - torqueBuildDelayS));
    return m * (a + 9.81 * slope) * v;
  }, slopeDeg, stopDistanceM, torqueBuildDelayS, distanceAfterBuildM };
}

function simulateRegen({ name, capF, cclA, scAvailable, scCellV = 2.0, dumpAvailable,
  profile, durationS = profile.stopS, mpptW = 0, pumpRunningW = 0,
  auxLoadW = p.regen.aux_power_worst_case_w ?? 0 }) {
  const dt0 = p.regen.dt_s, sampleDt = p.regen.sample_period_s;
  const busStart = 37.8, cBus = capF;
  const scCellC = p.supercap.cell_eol_sensitivity_f;
  const scCeq = scCellC / 8;
  const scMaxV = 8 * 2.2;
  let vBus = busStart, eBus = 0.5*cBus*vBus*vBus;
  let eSc = 0.5*scCeq*(8*scCellV)**2;
  const eScMax = 0.5*scCeq*scMaxV*scMaxV;
  const scEta = p.supercap.converter_efficiency_comparison;
  const scImax = p.supercap.charge_current_comparison_a;
  const busLimit = 39.009, dumpOnV = p.regen.dump_on_v_comparison, dumpOffV = p.regen.dump_off_v_comparison;
  let dumpOn = false, maxBus = vBus, first39 = null, firstDump = null, stop100 = false;
  let eRegen=0, eMppt=0, eAux=0, ePump=0, eBat=0, eScBus=0, eDump=0;
  let maxScPower=0, maxDumpPower=0, t=0, nextSample=0;
  const rows=[];
  const count = Math.ceil(durationS/dt0), dt = durationS/count;
  for (let k=0; k<=count; k++) {
    t = Math.min(k*dt, durationS);
    const preg = profile.powerAt(t);
    const pAux = auxLoadW;
    const pNet = preg + mpptW - pAux - pumpRunningW;
    if (!dumpOn && vBus >= dumpOnV && dumpAvailable) { dumpOn = true; if (firstDump===null) firstDump=t; }
    else if (dumpOn && vBus <= dumpOffV) dumpOn = false;
    const vSc = Math.sqrt(Math.max(0,2*eSc/scCeq));
    const pScLimit = (vSc*scImax)/scEta;
    const pScRoom = Math.max(0,(eScMax-eSc)/(scEta*dt));
    const pSc = scAvailable ? Math.min(Math.max(0,pNet),pScLimit,pScRoom) : 0;
    const pAfterSc = pNet - pSc;
    const pBat = Math.min(Math.max(0,pAfterSc),Math.max(0,cclA)*37.8);
    const pAfterBat = pAfterSc - pBat;
    const pDump = dumpOn ? vBus*vBus/p.regen.dump_r_ohm : 0;
    maxScPower=Math.max(maxScPower,pSc); maxDumpPower=Math.max(maxDumpPower,pDump);
    if (t>=nextSample || k===count) {
      rows.push([t, vBus, preg/2, preg/2, cclA, pBat, vSc/8, pSc, dumpOn?1:0, pDump,
        pAux, pumpRunningW, mpptW]);
      nextSample += sampleDt;
    }
    if (k===count) break;
    eRegen += preg*dt; eMppt += mpptW*dt; eAux += pAux*dt; ePump += pumpRunningW*dt;
    eBat += pBat*dt; eScBus += pSc*dt; eDump += pDump*dt;
    eSc = Math.min(eScMax,Math.max(0,eSc+pSc*scEta*dt));
    eBus = Math.max(0,eBus+(pNet-pSc-pBat-pDump)*dt);
    vBus = Math.sqrt(2*eBus/cBus); maxBus=Math.max(maxBus,vBus);
    if (first39===null && vBus>=busLimit) first39=t+dt;
    if (vBus>=p.regen.uncontrolled_bus_stop_v) { stop100=true; break; }
  }
  const eBusFinal=0.5*cBus*vBus*vBus;
  const residual=(eRegen+eMppt-eAux-ePump-eBat-eScBus-eDump)-(eBusFinal-0.5*cBus*busStart*busStart);
  return { name, modeled_duration_s:t, bus_cap_f:capF, initial_bus_v:busStart, maximum_bus_v:maxBus,
    comparison_39p009_v_crossed: first39!==null, first_cross_39p009_s:first39,
    central_dump_nominally_turned_on: firstDump!==null, first_dump_on_s:firstDump,
    central_dump_peak_w_model_only:maxDumpPower, dump_energy_j_model_only:eDump,
    total_regen_energy_j:eRegen, foc_a_regen_energy_j:eRegen/2, foc_b_regen_energy_j:eRegen/2,
    mppt_injection_energy_j:eMppt, auxiliary_load_energy_j:eAux, pump_load_energy_j:ePump,
    battery_energy_accepted_j_ideal_limit:eBat, battery_ccl_a_assumed:cclA,
    sc_energy_accepted_from_bus_j:eScBus, sc_energy_change_j:eSc-eSc, // overwritten below for clarity
    supercap_final_cell_v:Math.sqrt(2*eSc/scCeq)/8, supercap_cell_start_v:scCellV,
    supercap_available:scAvailable, supercap_charge_current_comparison_a:scImax,
    energy_balance_residual_j:residual, stopped_at_100v_model_boundary:stop100,
    comparison_status: first39===null ? 'BELOW_39p009_COMPARISON_LINE_IN_IDEAL_MODEL' : 'FAIL_CROSSES_39p009_COMPARISON_LINE',
    physical_qualification:'HOLD', waveform:rows };
}

function run() {
  const ph=p.precharge, packV=37.8, cHigh=ph.whole_bus_cap_initial_tolerance_high_f;
  const oldR=ph.legacy_resistor_r_high_ohm, oldWindows=ph.legacy_resistor_bms_window_s;
  const legacyConstant = {};
  for (const ia of [0,0.002,0.025,0.05,0.075,0.1]) {
    const key=`${(ia*1000).toFixed(1)}mA`;
    legacyConstant[key] = { at_min_window:rcConstantLoad(packV,oldR,cHigh,ia,oldWindows[0]),
      at_max_window:rcConstantLoad(packV,oldR,cHigh,ia,oldWindows[1]) };
  }
  const auxEnd=Math.max(...oldWindows);
  const currentDirect=simulateRcAux({packV,r:oldR,c:cHigh,backgroundA:ph.aux_sensitivity_background_a,
    regW:ph.regulator_aux_power_comparison_w,pumpW:ph.pump_aux_power_comparison_w,duration:auxEnd});
  const currentPumpGated=simulateRcAux({packV,r:oldR,c:cHigh-ph.pump_branch_cap_high_f,
    backgroundA:ph.aux_sensitivity_background_a,regW:ph.regulator_aux_power_comparison_w,pumpW:0,duration:auxEnd});
  const oldRcut = { cap_f:cHigh, r_high_ohm:oldR, timer_s:oldWindows,
    target_gap_v:ph.legacy_gap_comparison_v, no_load_time_s:legacyConstant['0.0mA'].at_min_window.time_to_0p5_v_s,
    '25mA_time_s':legacyConstant['25.0mA'].at_min_window.time_to_0p5_v_s,
    '50mA_time_s':legacyConstant['50.0mA'].at_min_window.time_to_0p5_v_s,
    end_gap_at_shortest_no_load_v:legacyConstant['0.0mA'].at_min_window.gap_v,
    end_gap_at_longest_no_load_v:legacyConstant['0.0mA'].at_max_window.gap_v,
    direct_reg_and_pump_end_gap_at_longest_v:currentDirect.end_gap_v,
    pump_gated_reg_aux_end_gap_at_longest_v:currentPumpGated.end_gap_v,
    direct_reg_and_pump_time_to_0p5_v_s:currentDirect.time_to_0p5_v_s,
    direct_aux_peak_current_a:currentDirect.peak_noncapacitive_load_a,
    pump_gated_precharge_cap_f:cHigh-ph.pump_branch_cap_high_f,
    status:'FAIL_AGAINST_0p5V_COMPARISON_WITHIN_EXISTING_BMS_WINDOW; 0.5V IS NOT A QUALIFIED SAFE CLOSURE LIMIT',
    physical_qualification:'HOLD' };

  const tps = {};
  for (const ma of [0.099,0.100]) {
    const key=`${(ma*1000).toFixed(0)}mA_total`;
    tps[key]=simulateTps({loadA:ma,c:cHigh,targetGap:ph.tps2490_candidate_gap_comparison_v});
  }
  const tpsRegOnly=simulateTps({loadA:0,c:cHigh,targetGap:ph.tps2490_candidate_gap_comparison_v,
    regW:ph.regulator_aux_power_comparison_w,bgA:ph.aux_sensitivity_background_a});
  const tpsAutoBoth=simulateTps({loadA:0,c:cHigh,targetGap:ph.tps2490_candidate_gap_comparison_v,
    regW:ph.regulator_aux_power_comparison_w,pumpW:ph.pump_aux_power_comparison_w,
    bgA:ph.aux_sensitivity_background_a});
  const cbr=p.regen.bus_cap_f_low_project ?? 0.05286;
  const flat=mechanicalStopProfile({slopeDeg:0,stopDistanceM:input.frozen_configuration.flat_stop_from_request_m,
    torqueBuildDelayS:input.frozen_configuration.flat_torque_build_delay_comparison_s});
  const gradePower=(gradePct)=>input.frozen_configuration.vehicle_mass_kg*9.81*
    Math.sin(Math.atan(gradePct/100))*(5/3.6);
  const cases=[];
  const profFlat={stopS:flat.stopS,powerAt:flat.powerAt};
  cases.push(simulateRegen({name:'project_flat_30kg_15kmh_3m_CCL0_SC_available_dump_available',
    capF:cbr,cclA:0,scAvailable:true,scCellV:2.0,dumpAvailable:true,profile:profFlat}));
  cases.push(simulateRegen({name:'project_flat_30kg_15kmh_3m_double_reject_SC_full_dump_available',
    capF:cbr,cclA:0,scAvailable:false,scCellV:2.2,dumpAvailable:true,profile:profFlat}));
  cases.push(simulateRegen({name:'project_flat_30kg_15kmh_3m_double_reject_dump_unavailable',
    capF:cbr,cclA:0,scAvailable:false,scCellV:2.2,dumpAvailable:false,profile:profFlat}));
  cases.push(simulateRegen({name:'project_flat_ideal_16A_CCL_and_SC_available',
    capF:cbr,cclA:16,scAvailable:true,scCellV:2.0,dumpAvailable:true,profile:profFlat}));
  const c20=gradePower(20), c30=gradePower(30);
  const grade20={stopS:1,powerAt:()=>c20};
  const grade30={stopS:1,powerAt:()=>c30};
  cases.push(simulateRegen({name:'20percent_grade_5kmh_1s_sensitivity_CCL0_SCfull_dump_available',
    capF:cbr,cclA:0,scAvailable:false,scCellV:2.2,dumpAvailable:true,profile:grade20,durationS:1}));
  cases.push(simulateRegen({name:'30percent_grade_5kmh_1s_sensitivity_CCL0_SCfull_dump_available',
    capF:cbr,cclA:0,scAvailable:false,scCellV:2.2,dumpAvailable:true,profile:grade30,durationS:1}));
  cases.push(simulateRegen({name:'30percent_grade_5kmh_1s_sensitivity_ideal_16A_CCL_SCfull_dump_available',
    capF:cbr,cclA:16,scAvailable:false,scCellV:2.2,dumpAvailable:true,profile:grade30,durationS:1}));
  cases.push(simulateRegen({name:'30percent_grade_5kmh_1s_plus_30W_MPPT_injection_sensitivity_double_reject',
    capF:cbr,cclA:0,scAvailable:false,scCellV:2.2,dumpAvailable:true,profile:grade30,durationS:1,
    mpptW:p.regen.mppt_input_sensitivity_w}));
  cases.push(simulateRegen({name:'30percent_grade_5kmh_1s_aux4p625W_best_case_sensitivity',
    capF:cbr,cclA:0,scAvailable:false,scCellV:2.2,dumpAvailable:true,profile:grade30,durationS:1,
    auxLoadW:p.regen.aux_power_comparison_w_total}));
  for (const c of cases) {
    const e0=0.5*(p.supercap.cell_eol_sensitivity_f/8)*(8*c.supercap_cell_start_v)**2;
    c.supercap_energy_change_j = (0.5*(p.supercap.cell_eol_sensitivity_f/8)*(8*c.supercap_final_cell_v)**2)-e0;
    delete c.sc_energy_change_j;
  }

  const waveformRows=[];
  for (const c of cases) for (const r of c.waveform) waveformRows.push([c.name,...r]);
  writeCsv('预充比较波形.csv',['t_s','legacy_bus_v','legacy_gap_v','precharge_current_a','aux_current_a','regulator_aux_on','pump_aux_on'],currentDirect.waveform);
  writeCsv('超容0V初充比较.csv',['t_s','supercap_cell_v','supercap_stack_v','supercap_charge_a','converter_input_w','converter_input_a_at_37p8v'],
    Array.from({length:401},(_,i)=>{const t=i*0.1, cell=Math.min(2.2,20*t/350), stack=8*cell, pin=stack*20/0.95; return [t,cell,stack,cell<2.2?20:0,pin,pin/37.8];}));
  writeCsv('双FOC制动与拒充波形.csv',['case','t_s','bus_v','foc_a_regen_w','foc_b_regen_w','battery_ccl_assumed_a','battery_absorb_w','supercap_cell_v','supercap_absorb_from_bus_w','dump_on','dump_w','unqualified_aux_load_w','pump_motor_load_w','mppt_injection_w'],waveformRows);
  writeCsv('四板许可门控.csv',['sequence','phase','main_proven','load_power_valid_foc_a','load_power_valid_foc_b','regulator_12v_protection','regulator_external_5v','pump_branch_power_enable','pump_motor_arm','sc_converter_charge_enable','notes'],[
    [1,'PRECHARGE',0,0,0,'BUS direct / startup envelope unknown',0,0,0,0,'Current pump branch is direct BUS; proposed upstream gate is not implemented. Current regulator 12V also starts from BUS when its internal threshold is crossed.'],
    [2,'MAIN_ROUTE_PROVEN',1,0,0,'must remain valid',0,0,0,0,'Candidate proof event only; never inferred from LOAD_POWER_VALID or timer expiry.'],
    [3,'AUXILIARY_RELEASE',1,1,1,'must remain valid','LPV after MAIN_PROVEN',1,0,0,'Pump power switch is a candidate interface; actual component and timing remain HOLD.'],
    [4,'LOCAL_READY_AND_NEW_ARM',1,1,1,'ready/valid','ready/valid',1,1,1,'FOC motion and supercapacitor charge only after local readiness; pump ARM is independent and explicit.'],
    [5,'FAULT_OR_POWER_LOSS',0,0,0,'hold through actual regen tail if physically available',0,0,0,0,'AUX drop order, final gate-off, current tail and per-island voltage are not simulated as qualified.']
  ]);

  const parentResultPath=path.join(HERE,'超容掉电与双峰值复核.json');
  const parentResult=fs.existsSync(parentResultPath)?JSON.parse(fs.readFileSync(parentResultPath,'utf8')):null;
  const sourceChecks=sourceLedger.filter(s=>s.baseline_sha256);
  const baselineAllMatch=sourceChecks.length>0 && sourceChecks.every(s=>s.exists && s.baseline_match);
  const scReq=p.supercap;
  const scCharge={equivalent_cap_nominal_f:scReq.cell_nominal_f/8,
    ideal_charge_current_a:scReq.charge_current_comparison_a,
    zero_to_2v_per_cell_s:scReq.cell_nominal_f*2/scReq.charge_current_comparison_a,
    zero_to_2p2v_per_cell_s:scReq.cell_nominal_f*2.2/scReq.charge_current_comparison_a,
    bus_input_power_at_2v_per_cell_w:8*2*scReq.charge_current_comparison_a/scReq.converter_efficiency_comparison,
    bus_input_power_at_2p2v_per_cell_w:8*2.2*scReq.charge_current_comparison_a/scReq.converter_efficiency_comparison,
    model_status:'PASS_MODEL_ONLY_FOR_IDEAL_CONSTANT_20A; actual 0V startup, efficiency, current and cell balancing HOLD',
    source_note:'20 A/95% is a model comparison only; it is not a guaranteed charger or discharge rating.'};
  const brakeEnergyWindow=0.5*p.supercap.cell_eol_sensitivity_f*8*((2.2**2)-(2.0**2));
  const result={
    schema:'agtc-four-board-joint-behavior-result-v1',
    run_utc:new Date().toISOString(),
    qualification:'HOLD',
    model_class:'Reduced-order behavioral/energy model; no switching waveform, no full circuit acceptance, no firmware or physical vehicle validation.',
    input_sha256:sha(fs.readFileSync(inputPath)),
    source_verification:{baseline_files_checked:sourceChecks.length,baseline_all_match:baselineAllMatch,
      current_sources:sourceLedger,interpretation:'Source hashes are reported as observed. Any missing or mismatched legacy binding is not silently treated as the current design.'},
    precharge:{legacy_resistor:oldRcut,
      tps2490_candidate:{status:'HOLD_CANDIDATE_NOT_IMPLEMENTED',candidate_inputs:{current_limit_min_a:ph.tps2490_candidate_current_min_a,
        r_series_ohm:ph.tps2490_candidate_series_resistance_ohm,target_gap_v:ph.tps2490_candidate_gap_comparison_v,
        cap_high_f:cHigh,hard_window_s:ph.tps2490_candidate_window_s},
        cases:tps,regulator_only_aux_sensitivity:tpsRegOnly,regulator_and_pump_auto_start_sensitivity:tpsAutoBoth,
        contract_result:'99/100mA combined comparison is inside the candidate 5.593s window in the model; formal topology and actual load envelope remain HOLD.',
        pump_gate_effect:'Pump gate moves 603.744uF out of precharge only if implemented; 0.229–0.389A capacitor step and 0.431J at 37.8V remain to be verified.'}},
    startup_and_gating:{supercap_zero_charge:scCharge,
      pump_stall:{status:'HOLD_NOT_SIMULATED',reason:'No qualified real-pump winding/back-EMF/start/stall model. 5.19A is a typical operating/chopping comparison, not a guaranteed stall limit.'},
      auxiliary_gating_status:'HOLD',waveform:'四板许可门控.csv'},
    regenerative_braking:{mechanical_primary:{mass_kg:flat.m,speed_kmh:flat.v0*3.6,stop_distance_m:flat.stopDistanceM,
      torque_build_delay_comparison_s:flat.torqueBuildDelayS,braking_distance_after_build_m:flat.distanceAfterBuildM,
      calculated_deceleration_m_per_s2:flat.a,calculated_stop_time_s:flat.stopS,
      translational_energy_j:0.5*flat.m*flat.v0*flat.v0,flat_initial_power_w:flat.powerAt(flat.torqueBuildDelayS),
      grade20pct_at5kmh_w:c20,grade30pct_at5kmh_w:c30,
      grade_point_duration:'Not frozen; 1 s files are display sensitivities only.',
      bus_cap_low_corner_f:cbr,high_corner_precharge_only_f:p.regen.bus_cap_f_high_screen,
      bus_start_v:37.8,comparison_line_v:39.009,
      energy_window_to_comparison_j:0.5*cbr*(39.009**2-37.8**2),
      central_dump_nominal_on_v:p.regen.dump_on_v_comparison,
      finding:'Central dump nominal comparator turns on above the 39.009V comparison line; it cannot be credited to keep this line without separate threshold/timing evidence.'},
      sc_energy_window_2p0_to2p2_v_eol_j:brakeEnergyWindow,
      cases:cases.map(({waveform,...rest})=>rest),
      waveform:'双FOC制动与拒充波形.csv',
      model_scope:'Each FOC injects one half of the same total bus power into the common BUS. SC is an ideal separate-port sink; battery CCL=0 is the default unqualified state. 16A is shown only as an ideal sensitivity. Dump is an ideal 0.82ohm hysteretic sink; no resistor thermal rating is inferred.'},
    external_boundaries:{mppt:'30W is an explicit bus-injection sensitivity input; no MPPT board/control model or gate is present.',
      vehicle_controller_aux:'No sourced current/startup profile; excluded.',
      pump_motor:'Excluded from energy closure; 63W running is not a guaranteed or persistent load.',
      battery:'Ideal voltage and current-limit screens only; no internal resistance, voltage sag, temperature, SOC, or current response.',
      mechanical:'No wheel torque/adhesion/rotational inertia/back-EMF/current tail or stopping validation.'},
    inherited_local_case:parentResult?{path:'超容掉电与双峰值复核.json',sha256:sha(fs.readFileSync(parentResultPath)),qualification:parentResult.qualification,
      isolation_cases:parentResult.isolation_cases,dual_foc_cases:parentResult.dual_foc_cases,
      scope:'Reused exactly as generated with ngspice_con.exe; not rerun here.'}:null,
    summary_status:{
      'FAIL':'Legacy resistor precharge misses even the old 0.5V comparison within the 2.346–2.738s window at the 79.784mF upper-capacitance corner; 0.5V is not a qualified safe-close threshold.',
      'PASS_MODEL_ONLY':'TPS2490 candidate arithmetic meets its 5.593s hard-window comparison at 99/100mA when the stated combined load and input assumptions hold.',
      'HOLD':'TPS2490 topology is not integrated into current BMS schematic; current per-branch dynamic absorption is not qualified; regulator/pump gating, water-pump stall, resistor pulse/thermal rating, 0V SC startup, battery and vehicle parking envelope remain open.'
    },
    waveform_files:['预充比较波形.csv','超容0V初充比较.csv','双FOC制动与拒充波形.csv','四板许可门控.csv']
  };
  fs.writeFileSync(path.join(HERE,'联合行为模型结果.json'),JSON.stringify(result,null,2)+'\n','utf8');
  console.log(JSON.stringify({result:path.join(HERE,'联合行为模型结果.json'),sourceCount:sourceLedger.length,
    legacyStatus:oldRcut.status,tpsStatus:result.precharge.tps2490_candidate.status,brakeCaseCount:cases.length},null,2));
}

run();
