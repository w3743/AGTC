// Combined dual charge refusal / rail revocation / BMS main-open reduced model.
// Node .../双拒充叠加断电联合仿真.mjs
import fs from 'node:fs'; import path from 'node:path'; import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
const here=path.dirname(fileURLToPath(import.meta.url));
const input=JSON.parse(fs.readFileSync(path.join(here,'双拒充叠加断电联合仿真输入.json'),'utf8'));
const torqueEvidence=JSON.parse(fs.readFileSync(path.join(here,'FOC双机低阶扭矩筛查.json'),'utf8'));
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const v0=input.vehicle.initial_speed_kmh/3.6, mass=input.vehicle.mass_kg, stopD=input.vehicle.stop_distance_m;
const behaviorStep=0.0005, horizon=5, delaySweep=input.braking.elapsed_delay_sweep;
const focBovV=input.bus.foc_paper_bov_v;
if(!Number.isFinite(focBovV)||focBovV<=input.bus.comparison_limit_v) throw new Error('Invalid FOC BOV input');
const sourceLedger=input.sources.map(s=>{const p=path.resolve(here,'../../..',s.path);return {path:s.path,role:s.role,exists:fs.existsSync(p),sha256:fs.existsSync(p)?sha(fs.readFileSync(p)):null};});
function electricalBrake(speed,bus,commandNm,motor,angleErrorDeg=0,modulation=1){
 const omega=4*speed/0.03, omegaE=7*omega;
 const emf=omegaE*motor.kt_nm_per_a/(1.5*7);
 const resistance=motor.r_phase_ohm, angle=angleErrorDeg*Math.PI/180;
 // Commanded Id=0 with an angle error gives real d/q components. Intersect
 // both error-sign voltage intervals; never combine one sign's favorable
 // voltage with the other sign's torque. EMF and torque share the same flux.
 let lower=0, upper=motor.iq_peak_a;
 for(const signed of [-angle,angle]){
  const a=resistance*Math.sin(signed)+omegaE*motor.lq_h*Math.cos(signed);
  const b=omegaE*(motor.ld_h??motor.lq_h)*Math.sin(signed)-resistance*Math.cos(signed);
  const quadratic=a*a+b*b, center=-emf*b;
  const discriminant=center**2-quadratic*(emf**2-modulation**2*bus**2/3);
  if(discriminant<0)return {torque:0,copper:0,current:0,voltageLimited:true};
  const span=Math.sqrt(discriminant);
  lower=Math.max(lower,(center-span)/quadratic);
  upper=Math.min(upper,(center+span)/quadratic);
 }
 const current=Math.min(upper,commandNm/(motor.kt_nm_per_a*Math.cos(angle)));
 if(current<lower)return {torque:0,copper:0,current:0,voltageLimited:true};
 return {torque:motor.kt_nm_per_a*current*Math.cos(angle),copper:1.5*resistance*current**2,current,
  voltageLimited:current+1e-9<commandNm/(motor.kt_nm_per_a*Math.cos(angle))};
}
function run(seq,triggerMode,detectBuild,Vinit,load,eta,cutV,brakeCount,dumpMode,physics){
 const dt=physics?.step??behaviorStep, movingMass=mass*(physics?.massFactor??1);
 const busCap=physics?.busCapF??input.bus.capacitance_f;
 const open=seq.open_s, railLoss=Math.min(seq.rail12_s,seq.core_s);
 const trigger=triggerMode==='brake_request'?0:seq.can_s;
 const onset=trigger+detectBuild;
 const shedAt=load.shed_delay_s===undefined?null:Math.min(open,trigger)+load.shed_delay_s;
 const remain=Math.max(0,stopD-v0*onset), aReq=remain>0?v0*v0/(2*remain):Infinity;
 const aFixed=input.braking.fixed_100ms_deceleration_m_s2;
 const aPer=aFixed*brakeCount/2;
 let E=.5*busCap*Vinit*Vinit, Einitial=E, v=v0, x=0, t=0, dumpOn=false, focCut=false;
 let edump=0,eload=0,ereg=0,packSource=0,maxV=Vinit,minMovingV=Vinit,maxDumpPower=0,firstTrip=null,first39=null,firstBov=null,firstCut=null,stoppedAt=null, torqueEndedByCut=false;
 const trace=[], steps=Math.ceil(horizon/dt);
 let pendingDumpAt=null, pendingReleaseAt=null, dumpCurrent=0, maxBusDrawFromMotors=0;
 let maxLocalV=Vinit;
 let maximumPhaseCurrent=0,minimumDeliveredTorque=null,voltageLimitedTime=0,unbuildableTime=0;
 const legacy=dumpMode==='legacy_40p32_rails_revoked';
 const onV=legacy?input.dump.legacy_on_v:(physics?.dumpOnV??input.dump.early_on_v);
 const offV=legacy?input.dump.legacy_off_v:(physics?.dumpOffV??input.dump.early_off_v);
 const resistance=physics?.dumpResistanceOhm??input.dump.resistance_ohm;
 const inductance=physics?.dumpInductanceH??0;
 const localThreshold=physics?.focBovV??focBovV;
 for(let k=0;k<steps;k++){
  t=k*dt; const V=Math.sqrt(2*E/busCap); maxV=Math.max(maxV,V);
  const afterOpen=t>=open;
  const dumpPowered=dumpMode==='independent_downstream_bus' ? V>input.dump.independent_supply_floor_v : t<railLoss;
  if(!dumpOn&&dumpPowered&&V>=onV){
   if(pendingDumpAt===null)pendingDumpAt=t;
   if(firstTrip===null)firstTrip=t;
   if(t-pendingDumpAt+1e-12>=(physics?.dumpDelay??0))dumpOn=true;
  }else if(!dumpPowered||V<onV){pendingDumpAt=null;}
  if(dumpOn&&V<=offV){
   if(pendingReleaseAt===null)pendingReleaseAt=t;
   if(t-pendingReleaseAt+1e-12>=(physics?.dumpOffDelay??0))dumpOn=false;
  }else pendingReleaseAt=null;
  if(!dumpPowered)dumpOn=false;
  const localV=V+(physics?.focAboveCentralV??0);
  maxLocalV=Math.max(maxLocalV,localV);
  if(firstCut===null&&V<=cutV) firstCut=t;
  if(localV>=localThreshold&&firstBov===null)firstBov=t;
  if(first39===null&&V>=input.bus.comparison_limit_v)first39=t;
  if(v>0)minMovingV=Math.min(minMovingV,V);
  if(brakeCount>0&&v>0&&(localV>=localThreshold||V<=cutV)){
   focCut=true;torqueEndedByCut=true;
  }
  const brakeAllowed=t>=onset&&!focCut&&v>0&&brakeCount>0;
  let a=brakeAllowed?aPer:0, motorCopper=0;
  if(brakeAllowed&&physics){
   const availability=v*3.6<physics.hallBelowKmh?physics.hallFraction:1;
   const commandNm=movingMass*aPer*0.03/(4*brakeCount)*availability;
   const angle=v*3.6<(physics.sectorBelowKmh??0)?physics.sectorAngleDeg:physics.angleErrorDeg;
   const point=electricalBrake(v,V,commandNm,physics.motor,angle??0,physics.modulation??1);
   maximumPhaseCurrent=Math.max(maximumPhaseCurrent,point.current);
   if(commandNm>0){
    minimumDeliveredTorque=minimumDeliveredTorque===null?point.torque:Math.min(minimumDeliveredTorque,point.torque);
    if(point.voltageLimited)voltageLimitedTime+=dt;
    if(point.torque===0)unbuildableTime+=dt;
   }
   a=point.torque*brakeCount*4/(0.03*movingMass);
   motorCopper=point.copper*brakeCount;
  }
  const moveDt=a>0?Math.min(dt,v/a):dt;
  const vn=Math.max(0,v-a*moveDt);
  const mechanicalRemoved=.5*movingMass*(v*v-vn*vn)/dt;
  const terminalPower=mechanicalRemoved-motorCopper*moveDt/dt;
  // In the dq cases, copper is removed once at the motor. Apply the
  // subsequent conversion efficiency according to power-flow direction.
  const pReg=physics?(terminalPower>=0?terminalPower*eta:terminalPower/eta):mechanicalRemoved*eta;
  const pMotorDraw=Math.max(0,-pReg);
  maxBusDrawFromMotors=Math.max(maxBusDrawFromMotors,pMotorDraw);
  const netMotorPower=afterOpen?Math.max(-E/dt,pReg):pReg;
  const loadDemand=(shedAt!==null&&t>=shedAt?load.after_shed_w:load.initial_w)+(physics?.fixedAuxW??0);
  const pLoad=afterOpen?Math.min(loadDemand,E/dt+netMotorPower):loadDemand;
  let dumpAverageA=0;
  if(dumpOn){
   const asymptote=V/resistance;
   if(inductance>0){
    const tau=inductance/resistance;
    const fraction=-Math.expm1(-dt/tau);
    dumpAverageA=asymptote+(dumpCurrent-asymptote)*fraction*tau/dt;
    dumpCurrent+=(asymptote-dumpCurrent)*fraction;
   }else dumpAverageA=asymptote;
  }else dumpCurrent=0;
  const pDump=Math.min(V*dumpAverageA,Math.max(0,E/dt+netMotorPower-pLoad));
  const pPack=afterOpen?0:Math.max(0,(Einitial-E)/dt-netMotorPower+pLoad+pDump);
  maxDumpPower=Math.max(maxDumpPower,pDump);
  E=Math.max(0,E+(netMotorPower-pLoad-pDump+pPack)*dt);
  packSource+=pPack*dt;
  edump+=pDump*dt; eload+=pLoad*dt; ereg+=netMotorPower*dt;
  if(physics?.saveTrace&&t<=2&&k%Math.round(0.01/dt)===0){
   trace.push([t,V,v*3.6,x,a,netMotorPower,pDump,loadDemand,!focCut].map(v=>typeof v==='number'?+v.toFixed(5):v));
  }
  x+=(v+vn)*.5*moveDt;
  if(a>0&&v<=a*dt&&stoppedAt===null)stoppedAt=t+moveDt;
  v=stoppedAt===null?vn:0;
 }
 const finalV=Math.sqrt(2*E/busCap), balance=E-Einitial-ereg+eload+edump-packSource;
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
  overvoltage_comparison_pass:Math.max(maxV,finalV)<input.bus.comparison_limit_v,
  ...(physics?.focBovV?{maximum_local_FOC_bus_v:+Math.max(maxLocalV,finalV+(physics.focAboveCentralV??0)).toFixed(4),local_foc_trip_v:localThreshold}:{}),
  ...(physics?{physics:{name:physics.name,mass_factor:physics.massFactor,hall_below_kmh:physics.hallBelowKmh,
   hall_torque_fraction:physics.hallFraction,dump_delay_s:physics.dumpDelay,maximum_motor_draw_w:+maxBusDrawFromMotors.toFixed(3),
   fixed_control_and_source_power_w:physics.fixedAuxW??0,modulation:physics.modulation??1,
   high_speed_angle_error_deg:physics.angleErrorDeg??0,sector_below_kmh:physics.sectorBelowKmh??null,
   sector_angle_error_deg:physics.sectorAngleDeg??null,ideal_zero_speed_clamp:true,
   dump_resistance_ohm:physics.dumpResistanceOhm??input.dump.resistance_ohm,
   maximum_phase_peak_current_a:maximumPhaseCurrent,minimum_delivered_torque_per_motor_nm:minimumDeliveredTorque,
   requested_torque_voltage_limited_time_s:voltageLimitedTime,requested_torque_unbuildable_time_s:unbuildableTime},
   ...(physics.saveTrace?{trace}: {})}:{})};
}
const trimSweep=process.argv.includes('--candidate-tlv-trim-sweep');
if(trimSweep||process.argv.includes('--candidate-tlv')){
 const thresholdsPath=path.resolve(here,'../../FOC驱动模块/检查/FOC_BOV精密阈值筛查.json');
 const thresholdBytes=fs.readFileSync(thresholdsPath);
 const thresholds=JSON.parse(thresholdBytes);
 const central=thresholds.central_reuse_512p7k_5p11k;
 const foc=thresholds.cases.find(c=>c.rt_ohm===1069800);
 const kv=JSON.parse(fs.readFileSync(path.join(here,'FOC_KV口径与低压建扭.json'),'utf8'));
 const motor={r_phase_ohm:0.020,ld_h:50e-6,lq_h:50e-6,
  kt_nm_per_a:kv.constants.dc_bus_kv_with_sinusoidal_phase_peak_current,iq_peak_a:20};
 input.dump.early_on_v=central.rise_overdrive10_latest_v;
 input.dump.early_off_v=central.fall_overdrive10_earliest_v;
 input.dump.independent_supply_floor_v=18+1.3+0.065;
 const baseline={name:'pump_shed_20ms',initial_w:70,shed_delay_s:.020,after_shed_w:5};
 function simulateCandidate(sequence,bus,load,overrides){
  return run(sequence,'brake_request',.05,bus,load,.8,18,2,'independent_downstream_bus',{
   step:20e-6,motor,name:'nominal',massFactor:1,hallBelowKmh:0,hallFraction:1,
   sectorBelowKmh:1,sectorAngleDeg:35,angleErrorDeg:5,modulation:.95,
   dumpDelay:.0005,dumpOffDelay:.0001,dumpResistanceOhm:.861,dumpInductanceH:100e-6,
   fixedAuxW:3.2,focAboveCentralV:.1,focBovV:foc.up_low_v,...overrides,
  });
 }
 if(trimSweep){
  // Use the same comparator, 60 K temperature interval and divider corners
  // as the shared threshold study. Only the series trim changes.
  const precise=.001+25e-6*60, thickFilm=.01+100e-6*60;
  const bottomLow=5110*(1-precise), bottomHigh=5110*(1+precise);
  function thresholdCorners(trim,groundErrorV=.04){
   const topHigh=510000*(1+precise)+trim*(1+thickFilm);
   const topLow=510000*(1-precise)+trim*(1-thickFilm);
   return {on:.414*(1+topHigh/bottomLow)+25e-9*topHigh+groundErrorV,
    off:.377*(1+topLow/bottomHigh)-25e-9*topLow-groundErrorV};
  }
  const reference=thresholdCorners(2700);
  if(Math.abs(reference.on-central.rise_overdrive10_latest_v)>1e-10||
     Math.abs(reference.off-central.fall_overdrive10_earliest_v)>1e-10)throw Error('Threshold models disagree');
  const sourcePaths={
   regulator:'../../稳压模块/检查/比较器参考地验收/改前/主工程.enet',
   foc:'../../FOC驱动模块/检查/主工程图面验收/主工程.enet',
   bms:'../../BMS模块/检查/主工程图面验收/主工程.enet',
   pump:'../../水泵驱动模块/检查/主工程图面验收/主工程.enet',
  };
  const baselineSources=Object.fromEntries(Object.entries(sourcePaths).map(([board,relative])=>{
   const netlist=path.resolve(here,relative);
   const bytes=fs.readFileSync(netlist), digest=sha(bytes);
   const proof=JSON.parse(fs.readFileSync(path.join(path.dirname(netlist),'验收.json'),'utf8'));
   if((proof.status??proof.result)!=='PASS'||proof.exports_sha256['主工程.enet']!==digest)
    throw Error(`Official netlist no longer matches its readback: ${board}`);
   const byRef=Object.fromEntries(Object.values(JSON.parse(bytes.toString('utf8').replace(/^\uFEFF/, '')).components)
    .map(part=>[part.props.Designator,part]));
   return [board,{path:relative,sha256:digest,byRef}];
  }));
  function verifySource(board,ref,pins,mpn){
   const part=baselineSources[board].byRef[ref];
   if(!part||part.props['Manufacturer Part']!==mpn||
      Object.entries(pins).some(([pin,net])=>part.pinInfoMap[pin]?.net!==net))
    throw Error(`Unreviewed baseline: ${board}/${ref}`);
   return {board,ref,mpn,pins};
  }
  const sourceChecks=[
   verifySource('regulator','U_DUMP_DRV',{'1':'12V_DRV','5':'DUMP_OFF','6':'12V_DRV'},'UCC27511ADBVR'),
   verifySource('regulator','U_SC_OV',{'2':'SC_MINUS','3':'BUS_OV_DIV','5':'12V_DRV'},'TLV6700DDCR'),
   verifySource('regulator','R_BUS_OV_BOT',{'1':'BUS_OV_DIV','2':'SC_MINUS'},'RT0603BRD075K11L'),
   verifySource('regulator','R_SC_OV_BOT',{'1':'SC_OV_DIV','2':'SC_MINUS'},'HoAR0603-1/10W-10KR-0.1%-TCR25'),
   verifySource('regulator','C_T3_FILTER',{'1':'BUS_OV_DIV','2':'SC_MINUS'},'CC0805CG100V102GN'),
   verifySource('regulator','C_SC_OV_VDD',{'1':'12V_DRV','2':'SC_MINUS'},'CGA3E2X7R1H104KT0Y0S'),
   verifySource('foc','R_OVTOP',{'1':'BUS','2':'VBUS_OV_SENSE'},'RC0603FR-071ML'),
   verifySource('foc','U7',{'4':'VBUS_OV_SENSE','5':'DUMP_5V','6':'BUSOV_N'},'TLV6700DDCR'),
   verifySource('bms','R_ALERT_SER',{},'RMC06031K1%N'),
   verifySource('pump','R_IPROPI',{},'RMC06031.5K1%N'),
  ];
  const existingFocTopLow=1e6*(1-thickFilm);
  const existingFocEarliest=.396*(1+existingFocTopLow/(1e4*(1+precise)))-15e-9*existingFocTopLow;
  const trims=[];
  for(const trim of [0,1000,1500,2700]){
   const {on,off}=thresholdCorners(trim), cases=[];
   for(const busCap of [.05286,.042288])for(const delta of [.1,.3])for(const delay of [.0005,.002]){
    const runs=[];
    for(const sequence of input.sequences)for(const bus of [22.5,37.8]){
     runs.push(simulateCandidate(sequence,bus,baseline,{busCapF:busCap,focAboveCentralV:delta,
      dumpDelay:delay,dumpOnV:on,dumpOffV:off}));
    }
    if(runs.some(c=>Math.abs(c.BUS_energy_balance_error_j)>.02))throw Error('Trim sweep energy ledger failed');
    const screenEnd=foc.up_low_v-delta;
    const energy=.5*busCap*(screenEnd**2-on**2);
    cases.push({bus_cap_f:busCap,local_minus_central_v:delta,gate_delay_s:delay,
     ideal_window_450w_ms:Math.max(0,energy/450.03*1000),
     ideal_window_915w_ms:Math.max(0,energy/915*1000),
     passed:runs.filter(c=>c.stop_within_3m&&c.first_FOC_paper_BOV_s===null).length,
     total:runs.length,max_near_bridge_v:Math.max(...runs.map(c=>c.maximum_local_FOC_bus_v))});
   }
   trims.push({trim_ohm:trim,latest_on_overdrive_v:on,earliest_release_overdrive_v:off,
    release_margin_above_37p8_v:off-37.8,cases});
  }
  const riseLimit=Math.sqrt((foc.up_low_v-.3)**2-2*450.03*.0005/.05286);
  const noTrim=thresholdCorners(0), oneOhm=thresholdCorners(1);
  const maxTrimAtRise=(riseLimit-noTrim.on)/(oneOhm.on-noTrim.on);
  const minTrimAtRelease=(37.9-noTrim.off)/(oneOhm.off-noTrim.off);
  if(maxTrimAtRise>=minTrimAtRelease)
   throw Error('Expected rise/release tradeoff was not reproduced');
  const output=path.join(here,'双拒充叠加断电_复用电阻取舍.json');
  const sources=Object.fromEntries(Object.entries(baselineSources).map(([board,{path,sha256}])=>[board,{path,sha256}]));
  fs.writeFileSync(output,JSON.stringify({status:'CONDITIONAL_SCREEN_HOLD',sources,source_checks:sourceChecks,
   threshold_file_sha256:sha(thresholdBytes),
   present:{foc_earliest_static_v:existingFocEarliest,
    central_latest_static_v:thresholdCorners(0).on-.010*(1+510000*(1+precise)/bottomLow),
    central_dump_vdd:'12V_DRV'},
   candidate_foc_earliest_static_v:foc.up_low_v,trims,
   ground_reference_candidate:{scope:'Move U_SC_OV.2 and both divider returns, BUS filter and comparator bypass to Kelvin GND_SYS. Keep SC_PLUS sense unchanged.',
    assumed_residual_ground_error_v:.010,trim_ohm:1500,...thresholdCorners(1500,.010),
    sc_overvoltage_threshold_high_v:.404*(1+510000*(1+precise)/(10000*(1-precise)))
      +25e-9*510000*(1+precise)+80*.0005*1.01+.010},
   joint_stress:{bus_cap_f:.05286,central_to_near_bridge_v:.3,net_injection_w:450.03,
    gate_delay_s:.0005,comparison_release_target_v:37.9,minimum_trim_for_release_ohm:minTrimAtRelease,
    maximum_trim_for_rise_ohm:maxTrimAtRise,feasible:false},
   boundaries:['Candidate assumes the independent rail is already valid, both FOC retain torque, and tail feedback stays available.',
    'The 10 mV comparator overdrive, 0.5 ms activation and 100 mV release guard are engineering screens. Comparator timing limits and dump branch energy need physical verification.',
    'The ideal energy window runs to gate activation; R/L current build and release are included separately in the motion model. Neither model grants TVS braking credit.']
  },null,2)+'\n');
  console.log(JSON.stringify({output,cases:trims.reduce((n,t)=>n+t.cases.reduce((k,c)=>k+c.total,0),0),minTrimAtRelease,maxTrimAtRise}));
  process.exit(0);
 }
 const cases=[];
 for(const scenario of ['nominal','line_drop','delayed_dump','pump_retained','tail_feedback_lost']){
  for(const sequence of input.sequences)for(const bus of [22.5,37.8]){
   const load=scenario==='pump_retained'?{name:scenario,initial_w:70}:baseline;
   const tailLost=scenario==='tail_feedback_lost';
   cases.push(simulateCandidate(sequence,bus,load,{
    name:scenario,hallBelowKmh:tailLost?.5:0,hallFraction:tailLost?0:1,
    dumpDelay:scenario==='delayed_dump'?.002:.0005,focAboveCentralV:scenario==='line_drop'?.3:.1,
   }));
  }
 }
 if(cases.some(c=>Math.abs(c.BUS_energy_balance_error_j)>.02))throw Error('Candidate energy ledger failed');
 if(!cases.some(c=>c.physics.name==='nominal'&&c.stop_within_3m)||
    !cases.some(c=>c.physics.name==='line_drop'&&c.first_FOC_paper_BOV_s!==null)||
    cases.some(c=>c.physics.name==='tail_feedback_lost'&&c.stopped))throw Error('Missing expected discriminating case');
 const counts=Object.fromEntries([...new Set(cases.map(c=>c.physics.name))].map(name=>{
  const rows=cases.filter(c=>c.physics.name===name);
  return [name,{total:rows.length,stop_in_3m_without_foc_trip:rows.filter(c=>c.stop_within_3m&&c.first_FOC_paper_BOV_s===null).length}];
 }));
 const candidateCases=cases.map(({first_39p009_crossing_s,overvoltage_comparison_pass,first_FOC_paper_BOV_s,...rest})=>
  ({...rest,first_local_foc_bov_s:first_FOC_paper_BOV_s}));
 const result={status:'CONDITIONAL_CIRCUIT_AND_MOTION_MODEL',threshold_file_sha256:sha(thresholdBytes),
  central_on_v:input.dump.early_on_v,central_off_v:input.dump.early_off_v,foc_trip_v:foc.up_low_v,counts,cases:candidateCases,
  boundaries:['Two motor torque and Hall-angle assumptions are inherited; vehicle has ideal zero-speed termination.',
   'Dump R/L is .861 ohm/100 uH for comparison, bus-fed source floor is 19.365 V; the 0.5 ms turn-on and 0.1 ms release intervals are budgets. The model gives no credit to off-state freewheel current.',
   'Modeled dump energy is branch terminal energy, including stored inductive energy, not a measured resistor rating. Peak power is a branch electrical load.',
   'Both source absorptions are zero; 3.2 W is the comparison for FOC logic and hardware protection supply.']};
 const output=path.join(here,'双拒充叠加断电_共源TLV复算.json');
 fs.writeFileSync(output,JSON.stringify(result,null,2)+'\n');
 console.log(JSON.stringify({status:result.status,counts,output}));
 process.exit(0);
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
const physicsCases=[];
const pumpShed=input.auxiliary_load_profiles.find(c=>c.name==='pump_shed_50ms');
const motorCases=torqueEvidence.assumed_parameter_cases_not_measurements;
for(const [name,motor] of Object.entries(motorCases)){
 for(const sequence of input.sequences.filter(c=>['simultaneous_all_revoked','main_opens_after_braking_begins'].includes(c.name))){
  for(const bus of [22.5,37.8])for(const massFactor of [1,1.5]){
   for(const [hallBelowKmh,hallFraction] of [[0,1],[1,0.5],[0.5,0]]){
    physicsCases.push(run(sequence,'brake_request',0.05,bus,pumpShed,0.8,18,2,'independent_downstream_bus',
     {step:0.00005, motor, name, massFactor, hallBelowKmh, hallFraction, dumpDelay:0.0005,
      saveTrace:sequence.open_s===0&&massFactor===1&&name==='central_sensitivity'&&
       (hallBelowKmh===0||bus===37.8&&hallFraction===0)}));
   }
  }
 }
}
if(!physicsCases.every(c=>Math.abs(c.BUS_energy_balance_error_j)<0.02))throw new Error('Coupled motor and BUS energy balance failed');
// New-design preparation sweep. It deliberately retains a failing low-speed
// feedback case and an unshed pump; this is not final native/vehicle approval.
const kvBudget=JSON.parse(fs.readFileSync(path.join(here,'FOC_KV口径与低压建扭.json'),'utf8'));
const sourceBudget=JSON.parse(fs.readFileSync(path.join(here,'同域泄放互锁核算.json'),'utf8'));
const comparatorBudget=JSON.parse(fs.readFileSync(path.join(here,'独立泄放比较器门限角.json'),'utf8'));
const nextCases=[];
const selectedMotor={r_phase_ohm:0.020,ld_h:50e-6,lq_h:50e-6,
 kt_nm_per_a:kvBudget.constants.dc_bus_kv_with_sinusoidal_phase_peak_current,iq_peak_a:20};
const oldOn=input.dump.early_on_v,oldOff=input.dump.early_off_v;
input.dump.early_on_v=comparatorBudget.rising_range_v[1];
input.dump.early_off_v=comparatorBudget.release_range_v[0];
for(const sequence of input.sequences)for(const bus of [22.5,37.8])
 for(const angle of [2,5])for(const shed of [0.015,0.020,0.050])
  for(const hall of ['bounded_sector','tail_unavailable'])for(const conversion of [0.8,1]){
   const load={name:`pump70_to5_at_${shed*1000}ms`,initial_w:70,shed_delay_s:shed,after_shed_w:5};
   // Include both FOC control loads plus the independently budgeted source.
   const fixedAuxW=2*1.2+bus*sourceBudget.source_thermal.input_current_budget_a;
   nextCases.push(run(sequence,'brake_request',0.05,bus,load,conversion,18,2,'independent_downstream_bus',{
    step:0.00005,motor:selectedMotor,name:`DCbus_KV_angle${angle}_${hall}`,massFactor:1,
    hallBelowKmh:hall==='tail_unavailable'?0.5:0,hallFraction:hall==='tail_unavailable'?0:1,
    sectorBelowKmh:1,sectorAngleDeg:35,angleErrorDeg:angle,modulation:0.95,
    dumpDelay:0.0005,dumpResistanceOhm:0.82*1.05,fixedAuxW,
   }));
  }
input.dump.early_on_v=oldOn;input.dump.early_off_v=oldOff;
if(!nextCases.every(c=>Math.abs(c.BUS_energy_balance_error_j)<0.02))throw new Error('New preparation energy balance failed');
if(nextCases.some(c=>c.physics.name.endsWith('tail_unavailable')&&c.stopped))throw new Error('Missing tail must not be reported stopped');
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
 schema:'agtc-dual-rejection-power-loss-v4-preparation',
 qualification:'HOLD',
 input_file:'双拒充叠加断电联合仿真输入.json',
 source_ledger:sourceLedger,
 model:{
  dt_s:behaviorStep,horizon_s:horizon,delay_sweep:delaySweep,
  fixed_deceleration_m_s2:input.braking.fixed_100ms_deceleration_m_s2,
  pre_open:'Ideal pack can source to the initial bus voltage floor but cannot absorb power. The model also covers BMS opening after braking begins.',
  post_open:'Capacitor supports auxiliaries plus modeled regen and dump. FOC candidate torque latches off on paper BOV or selected cutoff. Independent dump has its own 18 V comparison supply floor.',
  dump:'Ideal hysteretic resistor. Gate/current timing, local parasitics and thermal qualification are outside this model.',
  torque:'Actual installed FOC torque is uncredited. Candidate force is a constant-deceleration assumption, with no rotational inertia, slip, actual torque-speed/current map, or body-diode tail. One healthy FOC has half the total modeled force and no yaw model.',
  sequence:'Input timings are sensitivity points; CAN loss is a separate trigger. Aux demand is fixed or steps from 70 W to 5 W, an idealized pump load-shed requirement. Positive pack source and all modeled loads are integrated.',
  coupled_motor:'These cases use the dq voltage limit and signed bus power with copper losses. Motor data, Hall derate, local rail availability and dump delay are sensitivities, not verified hardware bounds. No positive actual braking torque is credited. Trace FOC state only indicates that the assumed BUS cut has not latched.',
  new_preparation:'The new sweep links phase-peak Kt and flux under the DC-bus KV hypothesis, intersects both angle-error voltage intervals at 95% modulation, uses 35-degree sector uncertainty below 1 km/h, includes source and two FOC control consumption, and applies conversion loss after motor copper according to flow direction. Warm independent source and functioning local rails are imposed. Speed is ideally clamped at zero: actual stop detection, reversal prevention, switching/current-loop dynamics, body-diode currents, source startup and native readiness are not proven.',
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
 coupled_motor_case_count:physicsCases.length,
 coupled_motor_comparison_pass_count:physicsCases.filter(c=>c.stop_within_3m&&c.overvoltage_comparison_pass).length,
 coupled_motor_trace_fields:['time_s','common_bus_v','speed_kmh','distance_m','deceleration_m_s2','signed_motor_bus_power_w','dump_w','auxiliary_demand_w','candidate_FOC_bus_above_cut'],
 coupled_motor_cases:physicsCases,
 new_preparation_scope:'Conditional engineering envelope only; not the final all-board native verification or a measured 3 m stop.',
 new_preparation_dependencies:[...['FOC_KV口径与低压建扭.json','同域泄放互锁核算.json','独立泄放比较器门限角.json'].map(p=>({path:p,sha256:sha(fs.readFileSync(path.join(here,p)))})),
  {path:'双拒充叠加断电联合仿真.mjs',sha256:sha(fs.readFileSync(fileURLToPath(import.meta.url)))}],
 new_preparation_counts:{total:nextCases.length,stop_and_voltage_comparison_met:nextCases.filter(c=>c.stop_within_3m&&c.overvoltage_comparison_pass).length},
 new_preparation_cases:nextCases,
 representative_cases:reference,
 overall:'HOLD: actual two-FOC torque, independent power and gate timing, local voltage, load shutdown, battery isolation, and thermal capability are not qualified.',
 case_fields:Object.keys(cases[0]),
};
const head=JSON.stringify(result,null,2).slice(0,-2);
const rows=cases.map(c=>'    '+JSON.stringify(Object.values(c))).join(',\n');
const output=path.join(here,'双拒充叠加断电联合仿真结果.json');
fs.writeFileSync(output,head+',\n  "case_rows": [\n'+rows+'\n  ]\n}\n');
console.log(JSON.stringify({qualification:result.qualification,checks,counts:result.counts,result:output}));
