// Combined dual charge refusal / rail revocation / BMS main-open reduced model.
// Node .../双拒充叠加断电联合仿真.mjs
import fs from 'node:fs'; import path from 'node:path'; import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
const here=path.dirname(fileURLToPath(import.meta.url));
const input=JSON.parse(fs.readFileSync(path.join(here,'双拒充叠加断电联合仿真输入.json'),'utf8'));
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const v0=input.vehicle.initial_speed_kmh/3.6, mass=input.vehicle.mass_kg, stopD=input.vehicle.stop_distance_m;
const dt=0.0005, horizon=5, delaySweep=input.braking.elapsed_delay_sweep;
const focBovV=input.bus.foc_paper_bov_v;
if(!Number.isFinite(focBovV)||focBovV<=input.bus.comparison_limit_v) throw new Error('Invalid FOC BOV input');
const sourceLedger=input.sources.map(s=>{const p=path.resolve(here,'../../..',s.path);return {path:s.path,role:s.role,exists:fs.existsSync(p),sha256:fs.existsSync(p)?sha(fs.readFileSync(p)):null};});
function run(seq,triggerMode,detectBuild,Vinit,load,eta,cutV,brakeCount,dumpMode){
 const open=seq.open_s, railLoss=Math.min(seq.rail12_s,seq.core_s);
 const trigger=triggerMode==='brake_request'?0:seq.can_s;
 const onset=trigger+detectBuild;
 const shedAt=load.shed_delay_s===undefined?null:Math.min(open,trigger)+load.shed_delay_s;
 const remain=Math.max(0,stopD-v0*onset), aReq=remain>0?v0*v0/(2*remain):Infinity;
 const aFixed=input.braking.fixed_100ms_deceleration_m_s2;
 const aPer=aFixed*brakeCount/2;
 let E=.5*input.bus.capacitance_f*Vinit*Vinit, Einitial=E, v=v0, x=0, t=0, dumpOn=false, focCut=false;
 let edump=0,eload=0,ereg=0,packSource=0,maxV=Vinit,minMovingV=Vinit,maxDumpPower=0,firstTrip=null,first39=null,firstBov=null,firstCut=null,stoppedAt=null, torqueEndedByCut=false;
 let trace=[]; const steps=Math.ceil(horizon/dt);
 const legacy=dumpMode==='legacy_40p32_rails_revoked';
 const onV=legacy?input.dump.legacy_on_v:input.dump.early_on_v;
 const offV=legacy?input.dump.legacy_off_v:input.dump.early_off_v;
 for(let k=0;k<steps;k++){
  t=k*dt; const V=Math.sqrt(2*E/input.bus.capacitance_f); maxV=Math.max(maxV,V);
  const afterOpen=t>=open;
  const dumpPowered=dumpMode==='independent_downstream_bus' ? V>input.dump.independent_supply_floor_v : t<railLoss;
  if(!dumpOn&&dumpPowered&&V>=onV){dumpOn=true;if(firstTrip===null)firstTrip=t;}
  if(dumpOn&&(!dumpPowered||V<=offV))dumpOn=false;
  if(firstCut===null&&V<=cutV) firstCut=t;
  if(V>=focBovV&&firstBov===null)firstBov=t;
  if(first39===null&&V>=input.bus.comparison_limit_v)first39=t;
  if(v>0)minMovingV=Math.min(minMovingV,V);
  if(brakeCount>0&&v>0&&(V>=focBovV||V<=cutV)){
   focCut=true;torqueEndedByCut=true;
  }
  const a=t>=onset&&!focCut&&v>0?aPer:0;
  const moveDt=a>0?Math.min(dt,v/a):dt;
  const vn=Math.max(0,v-a*moveDt);
  const pReg=.5*mass*(v*v-vn*vn)*eta/dt;
  const loadDemand=shedAt!==null&&t>=shedAt?load.after_shed_w:load.initial_w;
  const pLoad=afterOpen?Math.min(loadDemand,E/dt+pReg):loadDemand;
  const pDump=dumpOn?Math.min(V*V/input.dump.resistance_ohm,Math.max(0,E/dt+pReg-pLoad)):0;
  const pPack=afterOpen?0:Math.max(0,(Einitial-E)/dt-pReg+pLoad+pDump);
  maxDumpPower=Math.max(maxDumpPower,pDump);
  E=Math.max(0,E+(pReg-pLoad-pDump+pPack)*dt);
  packSource+=pPack*dt;
  edump+=pDump*dt; eload+=pLoad*dt; ereg+=pReg*dt;
  x+=(v+vn)*.5*moveDt;
  if(a>0&&v<=a*dt&&stoppedAt===null)stoppedAt=t+moveDt;
  v=stoppedAt===null?vn:0;
 }
 const finalV=Math.sqrt(2*E/input.bus.capacitance_f), balance=E-Einitial-ereg+eload+edump-packSource;
 return {sequence:seq.name,trigger_mode:triggerMode,trigger_s:trigger,commanded_torque_on_s:onset,detection_build_s:detectBuild,
  initial_bus_v:Vinit,pre_open_pack_source_j:+packSource.toFixed(4),aux_load_profile:load.name,
  initial_aux_load_w:load.initial_w,aux_shed_at_s:shedAt,regen_efficiency:eta,
  foc_bus_cutoff_v:cutV,active_foc_count:brakeCount,dump_mode:dumpMode,dump_on_v:onV,dump_release_v:offV,
  required_deceleration_for_this_elapsed_delay_m_s2:Number.isFinite(aReq)?+aReq.toFixed(4):null,
  stop_distance_m:stoppedAt===null?null:+x.toFixed(4),coasting_distance_at_5s_m:stoppedAt===null?+x.toFixed(4):null,
  stopped:stoppedAt!==null,stop_time_s:stoppedAt,braking_torque_ended_by_BUS_or_BOV:torqueEndedByCut,
  first_dump_threshold_s:firstTrip,first_39p009_crossing_s:first39,first_FOC_paper_BOV_s:firstBov,first_BUS_cutoff_s:firstCut,
  maximum_BUS_v:+Math.max(maxV,finalV).toFixed(4),min_BUS_while_moving_v:+minMovingV.toFixed(4),
  regen_energy_to_BUS_j:+ereg.toFixed(4),aux_energy_j:+eload.toFixed(4),dump_resistor_energy_j:+edump.toFixed(4),
  dump_resistor_peak_power_screen_w:+maxDumpPower.toFixed(1),BUS_energy_balance_error_j:+balance.toExponential(3),
  battery_absorption_w:0,supercap_absorption_w:0,stop_within_3m:stoppedAt!==null&&x<=stopD+1e-6,
  overvoltage_comparison_pass:Math.max(maxV,finalV)<input.bus.comparison_limit_v};
}
const cases=[];
for(const seq of input.sequences) for(const trigger of ['brake_request','can_fault']) for(const delay of delaySweep)
 for(const V of input.bus.initial_bus_sweep_v) for(const load of input.auxiliary_load_profiles) for(const eta of [0.8,1.0]) for(const cutoff of input.bus.local_cutoff_sensitivity_v)
  for(const count of [0,1,2]) for(const dump of ['candidate_early_threshold_old_rails','independent_downstream_bus','legacy_40p32_rails_revoked'])
   cases.push(run(seq,trigger,delay,V,load,eta,cutoff,count,dump));
const checks = {
 case_count: cases.length,
 all_sources_exist: sourceLedger.every(s=>s.exists),
 all_bus_energy_balances_within_0p02_j: cases.every(c=>Math.abs(c.BUS_energy_balance_error_j)<0.02),
 all_cases_zero_battery_and_supercap_regen_absorption: cases.every(c=>c.battery_absorption_w===0&&c.supercap_absorption_w===0),
 at_least_one_bov_cut_exercised: cases.some(c=>c.first_FOC_paper_BOV_s!==null&&c.braking_torque_ended_by_BUS_or_BOV&&!c.stopped),
 all_no_torque_cases_unstopped: cases.every(c=>c.active_foc_count!==0||!c.stopped),
};
if(Object.entries(checks).some(([key,value])=>key!=='case_count'&&!value)) throw new Error(JSON.stringify(checks));
const reference = cases.filter(c=>c.sequence==='simultaneous_all_revoked'&&c.trigger_mode==='brake_request'&&
 c.regen_efficiency===0.8&&c.active_foc_count===2&&c.initial_aux_load_w===70&&
 (c.dump_mode==='independent_downstream_bus'||c.detection_build_s===0.10&&c.aux_shed_at_s===null));
const batteryOnCases = cases.filter(c=>c.sequence==='main_opens_after_braking_begins');
const candidateCases = cases.filter(c=>c.active_foc_count===2&&c.dump_mode==='independent_downstream_bus');
const onCasesPass = candidateCases.filter(c=>c.stop_within_3m&&c.overvoltage_comparison_pass);
if(!reference.some(c=>c.stopped&&c.stop_within_3m)||!reference.some(c=>!c.stopped)) throw new Error('Missing conditional comparison');
if(!batteryOnCases.some(c=>c.first_dump_threshold_s!==null&&c.first_dump_threshold_s<0.175)) throw new Error('Missing braking before BMS opening');
const lowVoltage100ms=reference.filter(c=>c.initial_bus_v===22.5&&c.foc_bus_cutoff_v===18&&
 c.detection_build_s===0.10&&c.dump_mode==='independent_downstream_bus');
if(!lowVoltage100ms.some(c=>c.aux_shed_at_s!==null&&c.stop_within_3m)||
 !lowVoltage100ms.some(c=>c.aux_shed_at_s===null&&!c.stopped)) throw new Error('Missing pump shed sensitivity');
if(!lowVoltage100ms.some(c=>c.aux_load_profile==='pump_shed_65ms'&&c.stop_within_3m)||
 !lowVoltage100ms.some(c=>c.aux_load_profile==='pump_shed_75ms'&&!c.stopped)) throw new Error('Pump-shed limit comparison failed');
const energyWindow = (high,low)=>0.5*input.bus.capacitance_f*(high**2-low**2);
const result = {
 schema:'agtc-dual-rejection-power-loss-v3',
 qualification:'HOLD',
 input_file:'双拒充叠加断电联合仿真输入.json',
 source_ledger:sourceLedger,
 model:{
  dt_s:dt,horizon_s:horizon,delay_sweep:delaySweep,
  fixed_deceleration_m_s2:input.braking.fixed_100ms_deceleration_m_s2,
  pre_open:'Ideal pack can source to the initial bus voltage floor but cannot absorb power. The model also covers BMS opening after braking begins.',
  post_open:'Capacitor supports auxiliaries plus modeled regen and dump. FOC candidate torque latches off on paper BOV or selected cutoff. Independent dump has its own 18 V comparison supply floor.',
  dump:'Ideal hysteretic resistor. Gate/current timing, local parasitics and thermal qualification are outside this model.',
  torque:'Actual installed FOC torque is uncredited. Candidate force is a constant-deceleration assumption, with no rotational inertia, slip, actual torque-speed/current map, or body-diode tail. One healthy FOC has half the total modeled force and no yaw model.',
  sequence:'Input timings are sensitivity points; CAN loss is a separate trigger. Aux demand is fixed or steps from 70 W to 5 W, an idealized pump load-shed requirement. Positive pack source and all modeled loads are integrated.',
  results:'The full case_rows table uses column names listed in case_fields. No-stop rows have null stop distance. The 5 s coasting distance is illustrative.',
 },
 boundaries:{
  effective_bus_capacitance_f:input.bus.capacitance_f,
  bus_energy_22p5_to_15_j:+energyWindow(22.5,15).toFixed(4),
  bus_energy_22p5_to_18_j:+energyWindow(22.5,18).toFixed(4),
  ideal_latest_70w_to_5w_shed_for_100ms_zero_regen_s:+((energyWindow(22.5,18)-0.1*5)/(70-5)).toFixed(6),
  bus_energy_37p8_to_39p009_j:+energyWindow(39.009,37.8).toFixed(4),
  translation_kinetic_energy_j:+(0.5*mass*v0*v0).toFixed(4),
  fixed_braking_force_n:+(mass*input.braking.fixed_100ms_deceleration_m_s2).toFixed(4),
  fixed_shaft_torque_each_nm_ideal:+(mass*input.braking.fixed_100ms_deceleration_m_s2*0.03/8).toFixed(4),
  maximum_ideal_candidate_resistor_energy_j:Math.max(...candidateCases.map(c=>c.dump_resistor_energy_j)),
  resistor_thermal_rating:'Unknown. No pulse or continuous capability credited.',
 },
 checks,
 counts:{two_FOC_independent_dump:candidateCases.length,two_FOC_model_stop_and_voltage_pass:onCasesPass.length},
 representative_cases:reference,
 overall:'HOLD: actual two-FOC torque, independent power and gate timing, local voltage, load shutdown, battery isolation, and thermal capability are not qualified.',
 case_fields:Object.keys(cases[0]),
};
const head=JSON.stringify(result,null,2).slice(0,-2);
const rows=cases.map(c=>'    '+JSON.stringify(Object.values(c))).join(',\n');
const output=path.join(here,'双拒充叠加断电联合仿真结果.json');
fs.writeFileSync(output,head+',\n  "case_rows": [\n'+rows+'\n  ]\n}\n');
console.log(JSON.stringify({qualification:result.qualification,checks,counts:result.counts,result:output}));
