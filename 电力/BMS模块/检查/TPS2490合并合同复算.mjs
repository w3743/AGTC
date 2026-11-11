// Isolated review contract only; not an EDA file or an approved electrical design.
import {readFileSync,writeFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
const here = p=>fileURLToPath(new URL(p,import.meta.url));
const enet=readFileSync(here('./预充合闸候选基线/精简候选_P1.enet'));
const base=JSON.parse(enet), parts=Object.fromEntries(Object.values(base.components).map(x=>[x.props.Designator,x]));
const requireNet=(ref,pin,net)=>{const got=parts[ref]?.pinInfoMap?.[pin]?.net;if(got!==net)throw Error(`${ref}.${pin} expected ${net}, got ${got}`)};
for(const row of [['U_PRE_CMP','1','PRE_CMP_LO'],['U_PRE_ARM','6','PRE_RUN_OK'],['U_PRE_ARM','5','PRE_ARM_Q'],['U_GATE1','4','PRECHARGE_WINDOW'],['U_SEQ','8','TMR_IDLE'],['R_QPRE_BASE','1','PRECHARGE_WINDOW'],['Q_PRE1','2','BATT+_FUSED'],['Q_PRE2','2','QPRE_SW']])requireNet(...row);
const add=[
 {ref:'U_PRE_HS',mpn:'TPS2490DGSR',pins:{1:'PRE_PATH_EN',2:'PRE_TPS_VREF',3:'PRE_TPS_PROG',4:'PRE_TPS_TIMER',5:'B0',6:'NC',7:'PRE_LIN_OUT',8:'PRE_LINEAR_GATE',9:'PRE_TPS_SENSE',10:'BATT+_FUSED'}},
 {ref:'Q_PRE_HS',mpn:'CSD19536KTT',pins:{G:'PRE_LINEAR_GATE',D:'PRE_TPS_SENSE',S:'PRE_LIN_OUT'},pin_mapping:'TOLL/TO-263 exact CAD pin mapping to verify via official library'},
 {ref:'R_PRE_HS_SENSE',value:'56 mΩ total ±1.5%, Kelvin',pins:{1:'BATT+_FUSED',2:'PRE_TPS_SENSE'},mpn:'PENDING'},
 {ref:'R_PRE_HS_PROG',value:'47 kΩ',pins:{1:'PRE_TPS_VREF',2:'PRE_TPS_PROG'},mpn:'PENDING'},
 {ref:'C_PRE_HS_TIMER',value:'47–55 µF effective',pins:{1:'PRE_TPS_TIMER',2:'B0'},mpn:'PENDING'},
 {ref:'C_PRE_HS_IN',value:'100 nF min voltage/derating pending',pins:{1:'BATT+_FUSED',2:'B0'},mpn:'PENDING'},
 {ref:'D_PRE_HS_OUT',value:'OUT negative clamp; qualification pending',pins:{A:'B0',K:'PRE_LIN_OUT'},mpn:'PENDING'},
 {ref:'R_PRE_HS_EN_PD',value:'47 kΩ ±1%',pins:{1:'PRE_PATH_EN',2:'B0'},mpn:'PENDING'},
 {ref:'U_PRE_DIR_ARM',mpn:'SN74LVC1G08DCKR',pins:{1:'PRE_RUN_OK',2:'PRE_CMP_LO',3:'B0',4:'PRE_SAFE_ARM',5:'PRE_CTRL_3V3'}},
 {ref:'C_PRE_DIR_ARM',value:'100 nF',pins:{1:'PRE_CTRL_3V3',2:'B0'},mpn:'PENDING'},
 {ref:'U_PRE_CLOSE',mpn:'SN74LVC1G74DCTR',pins:{1:'FET_RELEASE',2:'PRE_CTRL_3V3',3:'PRE_CLOSE_N',4:'B0',5:'PRE_CLOSE_Q',6:'POR_OK_N',7:'PRE_CTRL_3V3',8:'PRE_CTRL_3V3'}},
 {ref:'C_PRE_CLOSE',value:'100 nF',pins:{1:'PRE_CTRL_3V3',2:'B0'},mpn:'PENDING'},
 {ref:'U_PRE_PATH',mpn:'SN74LVC1G08DCKR',pins:{1:'PRECHARGE_WINDOW',2:'PRE_CLOSE_N',3:'B0',4:'PRE_PATH_EN',5:'PRE_CTRL_3V3'}},
 {ref:'C_PRE_PATH',value:'100 nF',pins:{1:'PRE_CTRL_3V3',2:'B0'},mpn:'PENDING'},
 {ref:'U_TEST_EN',mpn:'SN74LVC1G11DCKR',pins:{1:'PRECHARGE_WINDOW',2:'B0',3:'FET_RELEASE',4:'PROBE_ENABLE',5:'PRE_CTRL_3V3',6:'PROBE_REQ'}},
 {ref:'C_TEST_EN',value:'100 nF',pins:{1:'PRE_CTRL_3V3',2:'B0'},mpn:'PENDING'},
 {ref:'Q_PROBE',mpn:'MMBT5551 (MDD C408396)',pins:{1:'PROBE_BASE',2:'PACK-',3:'PROBE_RETURN'}},
 {ref:'R_PROBE',mpn:'RN 3WS 820R F T/B A1 (C433538)',value:'820 Ω 3 W ±1%',pins:{1:'PACK+',2:'PROBE_RETURN'}},
 {ref:'R_PROBE_BASE',value:'270 Ω ±1%',pins:{1:'PROBE_ENABLE',2:'PROBE_BASE'},mpn:'PENDING'},
 {ref:'R_PROBE_PD',value:'100 kΩ ±1%',pins:{1:'PROBE_BASE',2:'PACK-'},mpn:'PENDING'},
 {ref:'R_PROBE_REQ_PD',value:'100 kΩ ±1%',pins:{1:'PROBE_REQ',2:'B0'},mpn:'PENDING'}];
const changes=[{pin:'U_PRE_ARM.6',from:'PRE_RUN_OK',to:'PRE_SAFE_ARM'},
 {pin:'Q_PRE1.2',from:'BATT+_FUSED',to:'PRE_LIN_OUT'},
 {pin:'Q_PRE2.2',from:'QPRE_SW',to:'PACK+'},
 {pin:'R_QPRE_BASE.1',from:'PRECHARGE_WINDOW',to:'PRE_PATH_EN'},
 {pin:'J_CORE_P2.11',from:'',to:'PROBE_REQ'},
 {ref:'R_TIMER_SET',action:'replace 9.31 kΩ with 9.53 kΩ RS-03K9531FT C294675; same connections'},
 {ref:'R_PRE',action:'remove from precharge; existing offboard asset retained separately; reuse undecided'}];
const merged=Object.fromEntries(Object.entries(parts).map(([ref,x])=>[ref,{
  mpn:x.props['Manufacturer Part']||null,
  pins:Object.fromEntries(Object.entries(x.pinInfoMap||{}).map(([pin,v])=>[pin,v.net||null]))
}]));
for(const c of changes) if(c.pin){const [ref,pin]=c.pin.split('.');if((merged[ref].pins[pin]||'')!==c.from)throw Error('pin changed: '+c.pin);merged[ref].pins[pin]=c.to;}
merged.R_TIMER_SET.mpn='RS-03K9531FT';merged.R_TIMER_SET.value='9.53 kΩ ±1%';
delete merged.R_PRE;
for(const item of add){if(merged[item.ref])throw Error('duplicate reference '+item.ref);merged[item.ref]={mpn:item.mpn||null,value:item.value||null,pins:item.pins};}
// Ideal state transitions only: asynchronous clear must require a new PRE_REQ,
// but the unchanged timer latch makes a second request sample idle=0.
function run(events){let arm=0,timer=0,local=1,dir=1;const out=[];
 for(const e of events){if(e.local!==undefined)local=e.local;if(e.dir!==undefined)dir=e.dir;
 if(!local||!dir)arm=0;
 if(e.req){const idle=1-timer;timer=1;arm=(local&&dir&&idle)?1:0;}
 out.push({arm,timer,window:arm&&timer?1:0});}return out;}
const seq=run([{req:1},{dir:0},{dir:1},{req:1}]);
const off=run([{dir:0},{req:1}]);
function pathState({window,closeSeen,release,probeReq}) {
 return {prePath:window&&!closeSeen?1:0,probe:window&&release&&probeReq?1:0};
}
const before=pathState({window:1,closeSeen:0,release:0,probeReq:0});
const probing=pathState({window:1,closeSeen:1,release:1,probeReq:1});
const failed=pathState({window:1,closeSeen:1,release:0,probeReq:1});
const test={start:seq[0].window===1,directionLoss:seq[1].window===0,
 noAutomaticRetry:seq[2].window===0,newRequestWhileTimerActive:seq[3].window===0,
 invalidDirectionAtRequest:off[1].window===0,
 prechargeBeforeMain:before.prePath===1&&before.probe===0,
 prechargeOffBeforeProbe:probing.prePath===0&&probing.probe===1,
 noReopenAfterMainClear:failed.prePath===0&&failed.probe===0};
if(!Object.values(test).every(Boolean))throw Error('ideal sequence failed');
const C=.079784+3.036e-6,V=37.8,Ilim=.045/(.056*1.015),Iload=.1183,R=.30,end=.076;
const crossover=Ilim*R,tc=C*(V-crossover)/(Ilim-Iload),rc=C*R;
const tr=rc*Math.log((crossover-Iload*R)/(end-Iload*R));
const charge=tc+tr,isolatedChargeC=.0015954846811044184;
const isolatedChargeDelay=isolatedChargeC/(end/R-Iload);
const probeRmax=820*(1+.05+.00035*50),probeRmin=820*(1-.05-.00035*50);
const probeTime=probeRmax*C*Math.log((22.1+.076-.2)/(22.1-.2464-.2));
const probeCurrentMin=(22.1-.2)/probeRmax,probeCurrentMax=(37.8+.076)/probeRmin;
const baseCurrentMin=(2.4-1)/(270*1.01)-1/(100000*.99);
const probeHold=1.1,verificationAllowance=.10;
const total=charge+isolatedChargeDelay+.075+.191+.010+probeHold+verificationAllowance;
const energy=37.8*(.055/(.056*.985))*7.057;
const continuityTest={probeRmaxOhm:probeRmax,probeRminOhm:probeRmin,probeMinA:probeCurrentMin,
 probeMaxA:probeCurrentMax,baseMinA:baseCurrentMin,openTrip_s:probeTime,probeHold_s:probeHold,
  powerMaxW:(37.8+.076)**2/probeRmin,assumptions:'no external source injects, passive open-loop capacitance upper bound, comparator remains powered/within error and BJT VCE <=0.2 V; transistor VCE guarantee only 25 C, 300 us data point'};
if(!(probeTime<probeHold&&probeCurrentMax<.05&&baseCurrentMin>.005))throw Error('probe conditional electrical check failed');
const out={status:'ISOLATED_CONTRACT_FOR_REVIEW_NOT_FOR_POWER_OR_RELEASE',
 baseline:{path:'检查/预充合闸候选基线/精简候选_P1.enet',sha256:createHash('sha256').update(enet).digest('hex'),boardCount:233},
 added:add,changes,mergedPinNets:merged,bomDelta:{boardAdded:add.length,boardRemoved:0,boardNet:add.length,offboardRemovedFromPrecharge:1,
 boardRelativeToPublishedNet:29+add.length,replacement:'R_TIMER_SET 9.31k to 9.53k',offboardAsset:'R_PRE remains existing off-board asset, not a disposal decision',purchaseCost:'unpriced items and assembly unknown'},
 tests:test,scenario:{directionRecovery:seq,notPermittedStart:off,before,probing,failed},
 continuityTest,model:{C_F:C,V_V:V,IlimMin_A:Ilim,assumedLoad_A:Iload,seriesOhm:R,deltaEnd_V:end,charge_s:charge,
 isolatedChargeUpperC:isolatedChargeC,isolatedChargeDelayBound_s:isolatedChargeDelay,
 afterSequence_s:total,shortestTimer_s:6.079,timerMargin_s:6.079-total,
 shortSourceEnergyUpperJ:energy,assumptions:'constant 118.3 mA, full current limit, 3.036 µF direct extra, isolated output charge conservatively added at minimum net current'},
 remaining:['PRE_CMP_LO is a negative-window threshold, not a zero-reverse-current guarantee; external source injection prohibited and reverse residual voltage/current acceptance is unresolved.',
 'The common 3.3 V signal and comparator power loss/slow ramps, 47 kΩ pull-down leakage and asynchronous clear/clock timing are unqualified; no automatic-retry guarantee in hardware yet.',
 'The isolated supply charge bound, startup/wiring, effective capacitor and 118.3 mA load are not qualified.',
 'EN turnoff does not guarantee physical current cessation: Q_PRE tail, TPS MOS SOA, reverse body diode, rail collapse and stored energy need full qualification.',
 'Probe only detects main-path opening under listed bound, not a milliohm resistance rating. Its production test must verify resistor value, transistor action, timer and actual OFF state; no claimed independent diagnosis of every component fault.',
 'Comparator, timing and test load need full error and actual input-noise validation; after MAIN_PROVEN the BQ protection handoff must be established before bridge/motion permission. EDA component mapping, output clamp, timer capacitor and unpriced MPNs await qualification.']};
writeFileSync(here('../原理图准备/TPS2490合并施工合同.json'),JSON.stringify(out,null,2)+'\n');
console.log(JSON.stringify(out,null,2));
