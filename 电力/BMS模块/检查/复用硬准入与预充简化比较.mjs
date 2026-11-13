// Analytical feasibility screen. All circuit and load limits remain assumptions.
import {writeFileSync,readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
const p={packV:37.8,capF:.079784,resistanceOhm:9.41965,windowChargeS:5.267,
 shortestWindowS:5.592878164523366,longestWindowS:6.54,
 shortVoltageV:37.8,shortRminOhm:7.284,comparisonCutoffS:7};
const time=(d,i)=>d>i*p.resistanceOhm?p.resistanceOhm*p.capF*Math.log((p.packV-i*p.resistanceOhm)/(d-i*p.resistanceOhm)):null;
const k=Math.exp(-p.windowChargeS/(p.resistanceOhm*p.capF));
const maxI=d=>(d-p.packV*k)/(p.resistanceOhm*(1-k));
const thresholds=[.076,.12,.26,.5].map(v=>({targetV:v,constantLoadMaxA:maxI(v),
 at10mA_s:time(v,.01),at25mA_s:time(v,.025),at50mA_s:time(v,.05),at118p3mA_s:time(v,.1183)}));
const iShort=p.shortVoltageV/p.shortRminOhm,pShort=p.shortVoltageV*iShort;
// Shunt value is a sensitivity bound for the stated temperature and tolerance,
// not an installed-unit or transient SOA guarantee.
const shuntRmin=.0005*(1-.01-.00005*50);
const closure=[.26,.2664,.5].map(deltaV=>({deltaV,shuntRminOhm:shuntRmin,
 passiveResistivePeakUpperA:deltaV/shuntRmin,differenceEnergyJ:.5*p.capF*deltaV**2}));
const src=fileURLToPath(new URL('./预充合闸候选基线/精简候选_P1.enet',import.meta.url));
// Reuse prior published worst-case comparator budget, changing only upper leg.
const side=[.9865,1.0135],tapHi=[],tapLo=[];
for(const v of [4.5219,5.25])for(const a of side)for(const b of side)for(const c of side)for(const d of side){
 const total=33200*(a+d)+1540*b+442*c;
 tapHi.push(v*(1540*b+442*c)/total);tapLo.push(v*442*c/total);
}
const eOut=.005+37.8/10**(91/20)+.000010+.000030+.002+2.625/1000+.000070+.003;
const posInner=(Math.min(...tapHi)-eOut)/.2002,posOuter=(Math.max(...tapHi)+eOut)/.1998;
const negInner=(Math.min(...tapLo)-eOut)/.2002,negOuter=(Math.max(...tapLo)+eOut)/.1998;
const dynamicAllowance=.020,prechargeTarget=posInner-.010;
const probeRmax=820*(1+.05+.00035*50), probeRmin=820*(1-.05-.00035*50);
const probeC=p.capF+3.036e-6;
// An accepted negative difference may be as large as the OUTER edge, not the guaranteed-pass inner edge.
// Extra-conservative post-closure initial difference, not the precharge direction threshold.
const worstAcceptedNegative=.262365;
const detectTime=probeRmax*probeC*Math.log((22.1+worstAcceptedNegative-2)/(22.1-posOuter-dynamicAllowance-2));
const chosenProbeS=4.1,periodExtra=.075+.070+.010+.100+.100;
const timerTol=[.006,25e-6*1.1,.005,.014,.004*1.1,.0024];
const tip=r=>(-.1284*(r/100)**2+46.9861*(r/100)-2651.8889)/100;
const timerLo=tip(11000*(1-.016))*timerTol.reduce((x,e)=>x*(1-e),1)-.05;
const timerHi=tip(11000*(1+.016))*timerTol.reduce((x,e)=>x*(1+e),1)-.05;
const ctime=load=>prechargeTarget>load*p.resistanceOhm?p.resistanceOhm*probeC*Math.log((37.8-load*p.resistanceOhm)/(prechargeTarget-load*p.resistanceOhm)):null;
const availableCharge=timerLo-chosenProbeS-periodExtra;
const q=Math.exp(-availableCharge/(p.resistanceOhm*probeC));
const maxLoad=(prechargeTarget-37.8*q)/(p.resistanceOhm*(1-q));
function rlcPeak(R,L,D){if(!L)return D/R;const w0=1/Math.sqrt(L*probeC),a=R/(2*L);
 if(a<w0){const w=Math.sqrt(w0*w0-a*a),t=Math.atan(w/a)/w;return D/(L*w)*Math.exp(-a*t)*Math.sin(w*t)}
 const b=Math.sqrt(a*a-w0*w0),t=Math.log((a+b)/(a-b))/(2*b);return D/(2*L*b)*(Math.exp((-a+b)*t)-Math.exp((-a-b)*t));}
const dWorst=posOuter+dynamicAllowance;
const fastDwell=7.284*.05286*Math.log((posInner-.020)/(negOuter+.020));
const par=(x,y)=>x*y/(x+y),ePrec=.001875,eTrim=.0135,topLo=par(4750*(1-ePrec),1e6*(1-eTrim)),topHi=par(4750*(1+ePrec),1e6*(1+eTrim)),supervisorFall=.792*(1+topLo/(1000*(1+ePrec))),supervisorRise=.808*1.0203*(1+topHi/(1000*(1-ePrec)))+1e-7*topHi;
const divTol=.0135, pu=4700,pd=100000, puLo=pu*(1-divTol),puHi=pu*(1+divTol),pdLo=pd*(1-divTol),pdHi=pd*(1+divTol),iLeak=5e-6;
const divHigh=2.7*pdLo/(puHi+pdLo)-iLeak*puHi*pdLo/(puHi+pdLo);
const divLow=.3,divMax=3.6,divZero=iLeak*puHi*pdHi/(puHi+pdHi);
const rcR=33000*(1+.016),rcC=1.265e-6,rcFinal=2.9-rcR*10e-6,rcTime=-rcR*rcC*Math.log(1-2/rcFinal);
const reuseRpre={status:'PREFERRED_CONDITIONAL_SCREEN_NOT_QUALIFIED',resistors:{upperSplitOhm:1540,positiveDirectionOhm:442,timerOhm:11000,dwellOhm:33000},
 supervisor:{mpn:'TPS3760A012DYYR',dividerOhm:{topPrecision:4750,parallel:1e6,bottom:1000},precisionToleranceWithTemperature:.001875,parallelToleranceWithTemperature:.0135,senseLeakageMaxA:1e-7,fallMinV:supervisorFall,riseMaxV:supervisorRise,resetPullup:'PRE_CTRL_3V3',bufferInput:{pullupOhm:pu,pulldownOhm:pd,toleranceIncludingTemp:divTol,totalLeakageAssumptionA:iLeak,minimumReleaseHighAt2p7V:divHigh,maximumResetLowV:divLow,maximumInputAt3p6V:divMax,maxResetSinkA:3.6/puLo,threshold:'VIH >= 1.7 V at 2.3–2.7 V and >= 2 V at 3–3.6 V; intermediate threshold bounded by monotonicity assumption.',lossOf3v3:'TPS3809 K33 falling threshold 2.87–2.99 V; reset propagation 10 us typical, no maximum, logic deassertion before 2.7 V not guaranteed for arbitrary fall slew.',senseDuringVddZero:'SENSE independent, absolute maximum 70 V; no explicit unpowered leakage/backfeed bound.'},slowEdgeBuffer:'SN74LVC1G17DBVR',limitations:'Static only; undervoltage propagation at small overdrive and fast supply collapse require actual waveform verification.'},dwellAssumptions:{driverVolts:2.9,resistorMaxOhm:rcR,capMaxF:rcC,totalLeakageMaxA:10e-6,highInputV:2,conditionalDwellSeconds:rcTime},errorAssumptionOutputV:eOut,guaranteedPassPositiveV:posInner,widestStaticPositiveV:posOuter,
 directionTripMinPositiveV:negInner,directionTripMaxPositiveV:negOuter,
 directionTripMinAfterExampleLatencyV:negInner-.020,closureWorstWithExampleLatencyV:dWorst,
 fastestChargeDwellWindowS:fastDwell,assumedDwellMaxS:.070,mainRequestServiceMaxS:.100,
 fastestWindowResidualS:fastDwell-.070-.100,
 capF:probeC,temperatureAdjustedProbeRmaxOhm:probeRmax,worstAcceptedNegativeV:worstAcceptedNegative,openDetectionSeconds:detectTime,probeHoldSeconds:chosenProbeS,
 probeCurrentMaxA:(37.8+worstAcceptedNegative)/probeRmin,probePowerMaxW:(37.8+worstAcceptedNegative)**2/probeRmin,
 probeBaseMinA:(2.4-1)/(270*1.01)-1/(100000*.99),
 timerShortestSeconds:timerLo,timerLongestSeconds:timerHi,allocatedOtherSeconds:periodExtra,chargeAllowanceSeconds:availableCharge,
 chargeAt10mASeconds:ctime(.010),chargeAt25mASeconds:ctime(.025),chargeAt42mACapsPlusFocSeconds:ctime(.045736),maxConstantLoadA:maxLoad,
 capacitorLeakageReference:{perCapA:.007,count:6,totalA:.042,condition:'KEMET 100 V rated, +20 C, 5 minutes; not a 38 V, 0–60 C startup guarantee'},
 shortPowerW:pShort,shortEnergyAtLongestJ:pShort*timerHi,
 closureDifferenceEnergyJ:.5*probeC*dWorst**2,
 rlcExamples:[0,1e-7,1e-6].map(L=>({loopROhm:.002,loopLH:L,peakA:rlcPeak(.002,L,dWorst)})),
 perFocConditionalStaticA:.001049+.000413+2*.000065+.000176+.000100,
 limitations:['Comparator errors and hysteresis reuse prior 442-ohm case assumptions; widened unbalanced ladder and monitor supplies need independent requalification.',
 'FOC current additions are conditional maxima at differing datasheet test points; electrolytic leakage, converter startup and all other branches still need a full bound.',
 'RLC numbers are ideal examples, not actual R/L lower bounds. Difference energy is total loop energy, not a per-MOS SOA guarantee.',
 'External sources must be unable to inject during open-main test. Probe transistor VCE <= 2 V at full temperature for 4.1 s remains an engineering budget; 0.2 V is specified only at the 25 C/300 us test point.',
 'Existing 300 W resistor 60 C free-air 253 W comparison applies only specified installation. Precharge MOS thermal and OFF tail are not qualified.']};
const parsed=JSON.parse(readFileSync(src));
const byRef=Object.fromEntries(Object.values(parsed.components).map(c=>[c.props.Designator,c]));
const check=(ref,pin,expected)=>{if((byRef[ref]?.pinInfoMap?.[pin]?.net||'')!==expected)throw Error(`baseline mismatch ${ref}.${pin}`)};
for(const x of [['U_PRE_ARM','6','PRE_RUN_OK'],['R_QPRE_BASE','1','PRECHARGE_WINDOW'],['J_CORE_P2','11',''],['U_PRE_CMP','1','PRE_CMP_LO'],['U_PRE_AND','1','PRE_CMP_LO'],['R_PRE_REF_HI','2','PRE_REF_MID'],['R_PRE_REF_LO','1','PRE_REF_MID'],['R_PRE_REF_LO','2','PRE_REF_LO'],['R_PRE_REF_BOT','1','PRE_REF_LO'],['R_DFETOFF_GATE','2','DFETOFF_G'],['R_DFETOFF_GS','1','DFETOFF_G'],['R_DFETOFF_PU','1','REG1'],['U1','30','DFETOFF']])check(...x);
const newItems=[
 {ref:'R_PRE_5V_SENSE_TOP',mpn:'RT0603BRD074K75L',supplier:'C705782',value:'4.75 kΩ 0.1% 25 ppm/°C',pins:{1:'VOUT_5V',2:'PRE_5V_SENSE'}},
 {ref:'R_PRE_5V_SENSE_BOT',mpn:'TC0325B1001T5F',supplier:'C2989112',value:'1.00 kΩ 0.1% 25 ppm/°C',pins:{1:'PRE_5V_SENSE',2:'B0'}},
 {ref:'R_PRE_5V_SENSE_TRIM',mpn:'0603WAF1004T5E',supplier:'C22935',value:'1 MΩ ±1%',pins:{1:'VOUT_5V',2:'PRE_5V_SENSE'}},
 {ref:'R_PRE_5V_PU',value:'4.7 kΩ ±1%',pins:{1:'PRE_CTRL_3V3',2:'PRE_5V_RESET_RAW'}},
 {ref:'U_PRE_5V_BUF',mpn:'SN74LVC1G17DBVR',supplier:'C7836',pins:{1:null,2:'PRE_5V_RESET_RAW',3:'B0',4:'PRE_SENSE5_OK',5:'PRE_CTRL_3V3'}},
 {ref:'C_PRE_5V_BUF',value:'100 nF',pins:{1:'PRE_CTRL_3V3',2:'B0'}},
 {ref:'C_PRE_5V_POR',value:'100 nF',pins:{1:'PRE_CTRL_3V3',2:'B0'}},
 {ref:'U_PRE_DIR_ARM',mpn:'SN74LVC1G08DCKR',supplier:'C7832',pins:{1:'PRE_RUN_OK',2:'PRE_LO_PHASE',3:'B0',4:'PRE_SAFE_ARM',5:'PRE_CTRL_3V3'}},
 {ref:'C_PRE_DIR_ARM',value:'100 nF',pins:{1:'PRE_CTRL_3V3',2:'B0'}},
 {ref:'U_PRE_LO_PHASE',mpn:'SN74LVC1G32DBVR',supplier:'C10096',reuse:'Frozen U_MAIN_HOLD_OR exact DBVR device and footprint',pins:{1:'PRE_CMP_LO',2:'FET_RELEASE',3:'B0',4:'PRE_LO_PHASE',5:'PRE_CTRL_3V3'}},
 {ref:'C_PRE_LO_PHASE',value:'100 nF',pins:{1:'PRE_CTRL_3V3',2:'B0'}},
 {ref:'U_PRE_CLOSE',mpn:'SN74LVC1G74DCUR',supplier:'C70285',pins:{1:'FET_RELEASE',2:'PRE_CTRL_3V3',3:'PRE_CLOSE_N',4:'B0',5:'PRE_CLOSE_Q',6:'POR_OK_N',7:'PRE_CTRL_3V3',8:'PRE_CTRL_3V3'}},
 {ref:'C_PRE_CLOSE',value:'100 nF',pins:{1:'PRE_CTRL_3V3',2:'B0'}},
 {ref:'U_PRE_PATH',mpn:'SN74LVC1G08DCKR',supplier:'C7832',pins:{1:'PRECHARGE_WINDOW',2:'PRE_CLOSE_N',3:'B0',4:'PRE_PATH_EN',5:'PRE_CTRL_3V3'}},
 {ref:'C_PRE_PATH',value:'100 nF',pins:{1:'PRE_CTRL_3V3',2:'B0'}},
 {ref:'U_TEST_EN',mpn:'SN74LVC1G11DCKR',supplier:'C529281',pins:{1:'PRECHARGE_WINDOW',2:'B0',3:'FET_RELEASE',4:'PROBE_ENABLE',5:'PRE_CTRL_3V3',6:'PROBE_REQ'}},
 {ref:'C_TEST_EN',value:'100 nF',pins:{1:'PRE_CTRL_3V3',2:'B0'}},
 {ref:'Q_PROBE',mpn:'MMBT5551 MDD C408396',pins:{1:'PROBE_BASE',2:'PACK-',3:'PROBE_RETURN'}},
 {ref:'R_PROBE',mpn:'RN 3WS 820R F T/B A1 C433538',value:'820 Ω 3 W ±1%',pins:{1:'PACK+',2:'PROBE_RETURN'}},
 {ref:'R_PROBE_BASE',value:'270 Ω ±1%',pins:{1:'PROBE_ENABLE',2:'PROBE_BASE'}},
 {ref:'R_PROBE_PD',value:'100 kΩ ±1%',pins:{1:'PROBE_BASE',2:'PACK-'}},
 {ref:'R_PROBE_REQ_PD',value:'100 kΩ ±1%',pins:{1:'PROBE_REQ',2:'B0'}},
 {ref:'U_LOAD_POWER_VALID_B',mpn:'VO617A-2X007T',reuse:'Same exact purchase MPN and CAD carrier as A; supplier SKU remains unverified.',pins:{1:'LOAD_VALID_B_LED_A',2:'LOAD_VALID_LED_K',3:'LOAD_POWER_VALID_B',4:'LOAD_EN_BIAS_EXT_B'}},
 {ref:'R_LOAD_POWER_VALID_B_LED',mpn:'0603WAF5100T5E',supplier:'C23193',value:'510 Ω ±1%',pins:{1:'VOUT_5V',2:'LOAD_VALID_B_LED_A'}},
 {ref:'J_LOAD_POWER_VALID_B',mpn:'2EDGVC-5.08-02P-14-100A(H)',supplier:'C581328',pins:{1:'LOAD_EN_BIAS_EXT_B',2:'LOAD_POWER_VALID_B'}},
 {ref:'R_LOAD_POWER_VALID_REG_LED',mpn:'0603WAF5100T5E',supplier:'C23193',value:'510 Ω ±1%',pins:{1:'VOUT_5V',2:'LOAD_VALID_REG_LED_A'}},
 {ref:'J_LPV_REG',mpn:'2541WV-2X01P',supplier:'C41430884',pins:{1:'LOAD_VALID_REG_LED_A',2:'LOAD_VALID_LED_K'}}];
// Reuse exact domestic/shared procurement identities; CAD carriers are audited separately.
for (const x of newItems) {
 if (x.ref.startsWith('C_')) Object.assign(x,{mpn:'0603B104K160NT',manufacturer:'Fenghua',supplier:'C2846678',value:'100 nF X7R ±10% 16 V 0603'});
 if (['R_PROBE_PD','R_PROBE_REQ_PD'].includes(x.ref)) Object.assign(x,{mpn:'RMC0603100K1%N',manufacturer:'Tyohm',supplier:'C269699'});
 if (x.ref==='R_PRE_5V_PU') Object.assign(x,{mpn:'RS-03K4701FT',manufacturer:'Fenghua',supplier:'C116693'});
 if (x.ref==='R_PROBE_BASE') Object.assign(x,{mpn:'1206W4F2700T5E',manufacturer:'UNI-ROYAL',supplier:'C4466',value:'270 Ω ±1% 1206',reuse:'Existing BMS exact 1206 identity; no new 0603 selection needed.'});
}
const proposed=Object.fromEntries(Object.entries(byRef).map(([ref,c])=>[ref,{mpn:c.props['Manufacturer Part']||null,
 pins:Object.fromEntries(Object.entries(c.pinInfoMap||{}).map(([pin,v])=>[pin,v.net||null]))}]));
proposed.U_PRE_5V_POR={mpn:'TPS3760A012DYYR',supplier:'C5218894',pins:{1:'PRE_CTRL_3V3',2:null,3:'PRE_5V_SENSE',4:null,5:null,6:'PRE_5V_RESET_RAW',7:null,8:'B0',9:null,10:null,11:null,12:null,13:'B0',14:null}};
proposed.R_PRE_5V_PD.pins['1']='PRE_5V_RESET_RAW';
proposed.R_DFETOFF_GATE.pins['2']='DFETOFF';proposed.R_DFETOFF_GS.pins['1']='DFETOFF';proposed.R_DFETOFF_PU.pins['1']='B0';delete proposed.Q_DFETOFF;
proposed.U_PRE_ARM.pins['6']='PRE_SAFE_ARM';proposed.R_QPRE_BASE.pins['1']='PRE_PATH_EN';
proposed.J_CORE_P2.pins['11']='PROBE_REQ';Object.assign(proposed.R_TIMER_SET,{mpn:'0603WAF1102T5E',supplier:'C25950',value:'11.0 kΩ'});
proposed.U_PRE_AND.pins['1']='PRE_LO_PHASE';
proposed.R_PRE_REF_HI.pins['2']='PRE_REF_LO';Object.assign(proposed.R_PRE_REF_HI,{mpn:'0603WAD1541T5E',supplier:'C2766999',value:'1.54 kΩ ±0.5%'});
proposed.R_PRE_REF_LO.pins['1']='PRE_REF_LO';proposed.R_PRE_REF_LO.pins['2']='PRE_REF_MID';
proposed.R_PRE_REF_BOT.pins['1']='PRE_REF_MID';
Object.assign(proposed.R_PRE_DWELL,{mpn:'0603WAF3302T5E',supplier:'C4216',value:'33 kΩ'});
Object.assign(proposed.R_LOAD_POWER_VALID_LED,{mpn:'0603WAF5100T5E',supplier:'C23193',value:'510 Ω ±1%'});
for(const x of newItems){if(proposed[x.ref])throw Error('duplicate '+x.ref);proposed[x.ref]=x;}
// Manufacturer package pin tables independently reviewed.
const pinAudit={
 U_PRE_5V_BUF:{1:null,2:'PRE_5V_RESET_RAW',3:'B0',4:'PRE_SENSE5_OK',5:'PRE_CTRL_3V3'},
 U_PRE_5V_POR:{1:'PRE_CTRL_3V3',2:null,3:'PRE_5V_SENSE',4:null,5:null,6:'PRE_5V_RESET_RAW',7:null,8:'B0',9:null,10:null,11:null,12:null,13:'B0',14:null},
 U_PRE_DIR_ARM:{1:'PRE_RUN_OK',2:'PRE_LO_PHASE',3:'B0',4:'PRE_SAFE_ARM',5:'PRE_CTRL_3V3'},
 U_PRE_LO_PHASE:{1:'PRE_CMP_LO',2:'FET_RELEASE',3:'B0',4:'PRE_LO_PHASE',5:'PRE_CTRL_3V3'},
 U_PRE_PATH:{1:'PRECHARGE_WINDOW',2:'PRE_CLOSE_N',3:'B0',4:'PRE_PATH_EN',5:'PRE_CTRL_3V3'},
 U_PRE_CLOSE:{1:'FET_RELEASE',2:'PRE_CTRL_3V3',3:'PRE_CLOSE_N',4:'B0',5:'PRE_CLOSE_Q',6:'POR_OK_N',7:'PRE_CTRL_3V3',8:'PRE_CTRL_3V3'},
 U_TEST_EN:{1:'PRECHARGE_WINDOW',2:'B0',3:'FET_RELEASE',4:'PROBE_ENABLE',5:'PRE_CTRL_3V3',6:'PROBE_REQ'}};
for(const [ref,pins] of Object.entries(pinAudit)) for(const [pin,net] of Object.entries(pins))
 if(proposed[ref]?.pins[pin]!==net) throw Error('pin audit mismatch '+ref+'.'+pin);
const gate=(window,seen,release,req)=>({precharge:window&&!seen?1:0,probe:window&&release&&req?1:0});
const s0=gate(1,0,0,0),s1=gate(1,1,1,1),s2=gate(1,1,0,1),s3=gate(0,1,1,1);
const phase=(lo,release)=>!!(lo||release);
const logicChecks={phasePositiveBeforeClose:phase(1,0),phaseReverseRejectedBeforeClose:!phase(0,0),
 phaseNearZeroAfterClose:phase(0,1),phaseRejectsAgainOnClear:!phase(0,0),
 highWindowStillRejectsAfterClose:!(0&&phase(0,1)),prechargeOnly:s0.precharge===1&&s0.probe===0,probeAfterOff:s1.precharge===0&&s1.probe===1,
 noRestartAfterMainClear:s2.precharge===0&&s2.probe===0,timeoutForcesProbeOff:s3.precharge===0&&s3.probe===0,
 detectionFitsProbe:detectTime<chosenProbeS,tenMilliampFits:ctime(.01)+chosenProbeS+periodExtra<timerLo,
 supervisorStaticThresholdsFit:supervisorFall>4.5&&supervisorRise<4.75,supervisorBufferConditionalCorners:divHigh>2&&divLow<.7&&divMax<5.5,positiveDirectionWithDelay:negInner-dynamicAllowance>0, conditionalRcWithin70ms:rcTime<.070,fastWindowAllowsDwellAndService:fastDwell>.070+.100,
 capsAndFocSensitivityFits:ctime(.045736)+chosenProbeS+periodExtra<timerLo,
 transistorCurrentWithin25CSaturationPoint:reuseRpre.probeCurrentMaxA<.05&&reuseRpre.probeBaseMinA>.005,
 heavy118mALoadFails:ctime(.1183)===null};
if(!Object.values(logicChecks).every(Boolean))throw Error('conditional verification failed');
const contract={status:'PREFERRED_ISOLATED_CONDITIONAL_NOT_RELEASED',baselineSha256:createHash('sha256').update(readFileSync(src)).digest('hex'),
 mergedPinNets:proposed,added:newItems,changedPins:['U_PRE_ARM.6 PRE_RUN_OK→PRE_SAFE_ARM','R_QPRE_BASE.1 PRECHARGE_WINDOW→PRE_PATH_EN',
 'J_CORE_P2.11 NC→PROBE_REQ','R_DFETOFF_GATE.2 DFETOFF_G→DFETOFF','R_DFETOFF_GS.1 DFETOFF_G→DFETOFF','R_DFETOFF_PU.1 REG1→B0','remove Q_DFETOFF','R_PRE_5V_PD.1 PRE_SENSE5_OK→PRE_5V_RESET_RAW','U_PRE_AND.1 PRE_CMP_LO→PRE_LO_PHASE',
 'R_PRE_REF_HI.2 PRE_REF_MID→PRE_REF_LO','R_PRE_REF_LO.1 PRE_REF_MID→PRE_REF_LO',
 'R_PRE_REF_LO.2 PRE_REF_LO→PRE_REF_MID','R_PRE_REF_BOT.1 PRE_REF_LO→PRE_REF_MID'],
 replaced:['U_PRE_5V_POR TLV809EA46DBZR→TPS3760A012DYYR','R_TIMER_SET 9.31k→11.0k','R_PRE_REF_HI 442Ω→1.54k','R_PRE_DWELL 100k→33k','R_LOAD_POWER_VALID_LED RC0603FR-07510RL→0603WAF5100T5E'],
 lpvHarness:{focA:'J_LOAD_POWER_VALID.1/.2 → FOC A same-named J_LOAD_POWER_VALID.1/.2',focB:'J_LOAD_POWER_VALID_B.1/.2 → FOC B J_LOAD_POWER_VALID.1/.2',regulator:'J_LPV_REG.1/.2 → regulator J_LPV_IN.1/.2, existing LTV-817S-TA1-C LED',population:'Fit R_LOAD_VALID_BIAS 1 kΩ on both FOC boards. No A/B bias, signal or ground cross-link. Existing BMS low-side driver controls all three independently limited LEDs.',minimumLedA:(supervisorFall-1.65-.10)/(510*1.016),maximumThreeLedA:3*5.25/(510*.984)},
 bomDelta:{boardAdded:newItems.length,boardRemoved:1,boardRelativeToPublishedNet:29+newItems.length-1,
 retained:'Q_PRE1/2 and existing offboard 8.2Ω/300W R_PRE',notAdded:'TPS2490, CSD19536KTT, shunt, timer capacitor and associated parts',
 cost:'USD 7.99 historical two-chip comparison avoided only if this alternative qualifies; landed BOM and stock unknown'},
 configurationContract:{DFETOFF_PIN_CONFIG:'0x92FB=0xC2 active-low BOTHOFF',wiring:'U_SEQ.5 FET_RELEASE → existing 1k → U1.30 DFETOFF; 47k and existing 100k to B0; remove Q_DFETOFF',prerequisite:'Before any MAIN_REQ or FET_EN, hold host and power stages inhibited, confirm hardware FETs actually off, configure FET_INIT_OFF and 0xC2, read back configuration and FET status, and only then perform fresh precharge.',productionOtp:'Program and verify 0x92FB=0xC2, FET_INIT_OFF=1, PDSG_EN=0, and PF_EN/OTPF enabled via existing I2C; no traction connection until done. This is a human production/commissioning requirement, not a claim already programmed.',resetBehavior:'TI FAQ 1.8: short RST_SHUT preserves registers, so 0xC2 remains configured and asserted low DFETOFF disables FETs; full RESET or wake from SHUTDOWN reloads OTP and requires the specified OTP defaults. Short RST_SHUT preserves prior FET state independently and is not itself a turn-off command.',resetSource:'https://www.ti.com/lit/an/sluaaq5/sluaaq5.pdf',supplyFallLimitation:'FET_RELEASE output and total DFETOFF injection must be bounded when PRE_CTRL_3V3 collapses; 47k/100k do not override an actively high output.'},logicChecks,manufacturerPinAudit:{pinAudit,sources:['https://www.ti.com/lit/ds/symlink/sn74lvc1g17.pdf','https://www.ti.com/lit/ds/symlink/tps3760.pdf','https://www.ti.com/lit/ds/symlink/sn74lvc1g08.pdf','https://www.ti.com/lit/ds/symlink/sn74lvc1g32.pdf','https://www.ti.com/lit/ds/symlink/sn74lvc1g74.pdf','https://www.ti.com/lit/ds/symlink/sn74lvc1g11.pdf']},reuseRpre,
 limitations:['No physical circuit has been modified, powered, or validated.','This test detects an open main under the stated passive-load and no-source assumptions; it is not a calibrated milliohm measurement.',
 'PRE_REQ/MAIN_REQ, phase-OR before main gate conduction, 70 ms dwell, 100 ms service, VCE <= 2 V across 4.1 s and full temperature, comparator saturation recovery, clock tolerance, actual R/L/SCD and MOS SOA require qualification.',
 'Probe production test must verify its own resistor and switching. Normal charge is inhibited until BQ protection handoff and FOC local safety/ARM are established.',
 'Once MAIN_PROVEN is set, existing MAIN_OR bypasses the precharge-window comparator; this must not be represented as continuous monitoring.',
 'SC, MPPT, motor regeneration and other sources must not inject during test; measured injection and effective capacitance above the model require recomputation.']};
writeFileSync(fileURLToPath(new URL('../原理图准备/复用预充合并施工合同.json',import.meta.url)),JSON.stringify(contract,null,2)+'\n');
const result={status:'ALTERNATIVE_CONDITIONAL_NOT_APPROVED',input:p,
 sourceFrozenSha256:createHash('sha256').update(readFileSync(src)).digest('hex'), thresholds,closure,reuseRpre,
 shortResistor:{currentA:iShort,powerW:pShort,energyAt6p54sJ:pShort*p.longestWindowS,
 energyAt7sJ:pShort*p.comparisonCutoffS,extraPhysicalTailNotIncluded:true},
 architecture:{baseline:'isolated 29 net added parts, original published plus comparator/qualification, not a new change',
 removeFromComparison:['precision ADC path','independent 1–4 A diagnostic pulse and related driver/timeout'],
 mainProven:'Replace claim with conditional AUX_RELEASE logic contract only; no physical main-path proof. All power stages remain inhibited pending local validation and actual load behavior.',
 uncertain:['analog thresholds and dynamic error','worst precharge load over voltage/time/temperature','safe closure delta, loop R/L, SCD and MOS SOA','undervoltage/partial-supply shutdown and timing','receive-side independent auxiliary vs motion permission'],
 closureBoundAssumptions:'Shunt in the actual isolated capacitor return, no alternate return, 25 to 75 C temperature sensitivity, ±1% and ±50 ppm/C, positive R/L, passive loop and zero initial current. Peak bound is conservative, not an actual peak. The capacitor-difference energy is total incremental loop energy, not per MOS; no source/load/transient injection included.'},
 tests:{openPathCannotBeProvenByZeroCurrentObservation:true,oldTwoChannelsWorstErrorFailsHalfVolt:1>.5,
 static29PartPlus20mVDelayExceeds260mV:.2464+.020>.260,
 lowLoadComparisonFits:time(.5,.025)<p.windowChargeS,
 existing118mALoadCannotReachHalfVolt:.1183*p.resistanceOhm>.5}};
writeFileSync(fileURLToPath(new URL('./复用硬准入与预充简化比较.json',import.meta.url)),JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify(result,null,2));
