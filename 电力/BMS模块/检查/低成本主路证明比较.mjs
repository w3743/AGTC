// Static comparisons only. No eFuse, sensor, switch or timer is qualified.
// Run: node 电力/BMS模块/检查/低成本主路证明比较.mjs
import {writeFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
const p={busMin:22.1,busMax:37.8,rPreMin:7.284,rPreMax:9.41965,rAddMin:2.2*.95,rAddMax:2.2*1.05,switchMax:.2,
  cc2Error:.1,otherCurrent:.005,prechargeLeak:.020,healthyR:.006,faultR:.030,senseError:.010,threshold:.038,window:.260};
const iMin=p.busMin/(p.rPreMax+p.rAddMax+p.switchMax), iMax=p.busMax/(p.rPreMin+p.rAddMin);
// Acceptance requires indicated CC2 >=1.85 A; 4.03 A is the comparison upper bound.
const observedMin=1.85, trueFaultMin=observedMin-p.cc2Error-p.otherCurrent-p.prechargeLeak;
const healthyMax=(iMax+p.otherCurrent)*p.healthyR+p.senseError, faultMin=trueFaultMin*p.faultR-p.senseError;
const practicalThresholdA=1.74;
const practicalCurrentMin=practicalThresholdA-p.cc2Error-p.otherCurrent-p.prechargeLeak;
const practicalFaultMin=practicalCurrentMin*p.faultR-p.senseError;
const practicalVoltageThreshold=(healthyMax+practicalFaultMin)/2;
const oneA={currentMin:.85,currentMax:1.10,readError:.005,threshold:.016};
const oneGood=oneA.currentMax*p.healthyR+oneA.readError;
const oneBad=oneA.currentMin*p.faultR-oneA.readError;
const small={currentMin:.1096,currentMax:.2175,readError:.010};
const smallGood=small.currentMax*p.healthyR+small.readError, smallBad=small.currentMin*p.faultR-small.readError;
function modelDecision({adcHealthy,platformValid,absoluteVolts}) {
  if(adcHealthy !== true || platformValid !== true) return 'REJECT';
  return absoluteVolts <= p.threshold ? 'CONDITIONAL_PASS' : 'REJECT';
}
// Startup screening uses BQ and the absolute path at different times. It cannot
// detect a Kelvin short introduced after that screening.
function startupScreen({bqDiff, largeResponse, recovered}) {
  return bqDiff >= 4 && largeResponse && recovered;
}
const startupNegativeControls = {
  alwaysShortedKelvinRejected: !startupScreen({bqDiff:4,largeResponse:false,recovered:true}),
  adcStuckRejected: !startupScreen({bqDiff:4,largeResponse:false,recovered:false}),
  shortAfterScreenCanStillPassModel: startupScreen({bqDiff:4,largeResponse:true,recovered:true}) &&
    modelDecision({adcHealthy:true,platformValid:true,absoluteVolts:0})==='CONDITIONAL_PASS',
};
const tests={resistorNormalCurrentWindow:iMin>=1.85&&iMax<=4.04,
 absolute38mVSeparatesEndpoints:healthyMax<p.threshold&&faultMin>p.threshold,
 oneA5mVSeparatesEndpoints:oneGood<oneA.threshold&&oneBad>oneA.threshold,
 smallPulseCannotSeparate:smallGood>=smallBad,
 stuckSensorWithUnqualifiedDiagnosticRejected:modelDecision({adcHealthy:false,platformValid:true,absoluteVolts:0})==='REJECT',
 saturatedSensorRejected:modelDecision({adcHealthy:false,platformValid:true,absoluteVolts:0})==='REJECT',
 openPlatformRejected:modelDecision({adcHealthy:true,platformValid:false,absoluteVolts:0})==='REJECT',
 openPathCannotMeetCurrent:(p.window/p.rPreMin)<trueFaultMin,
 old1575mAContractDoesNotSupport38mV:.030*1.575-p.senseError < p.threshold,
 lowVoltageCornerNotGuaranteedToMeetIndicatedThreshold:iMin-p.cc2Error<observedMin,
 lowVoltageCornerPassesAlternativeUnderBudget:iMin-p.cc2Error>=practicalThresholdA,
 alternativeSeparatesEndpoints:healthyMax<practicalVoltageThreshold&&practicalVoltageThreshold<practicalFaultMin,
 shortedAddedResistorBreaksFiveAmpLimit:p.busMax/p.rPreMin>5,
 shortedOriginalResistorBreaksFiveAmpLimit:p.busMax/p.rAddMin>5};
if(!Object.values(tests).every(Boolean)||!Object.values(startupNegativeControls).every(Boolean))throw Error('comparison self-test failed');
const result={status:'COMPARISON_ONLY_HOLD',inputs:p,normalPassive:{currentMinA:iMin,currentMaxA:iMax,powerUpperW:p.busMax*iMax,
 energy40msJ:p.busMax*iMax*.04,assumptions:'Both resistors intact; switches and wires <=0.2 ohm; voltage at actual input; ignores turn-on and capacitance.'},
 absoluteKelvin:{requiredIndicatedA:observedMin,trueFaultCurrentMinA:trueFaultMin,healthyReadMaxV:healthyMax,faultReadMinV:faultMin,thresholdV:p.threshold,
   slowCornerWorstIndicatedA:iMin-p.cc2Error,
   oldContractTrueFaultMinA:1.575,oldContractFaultReadMinV:.030*1.575-p.senseError,oldThresholdV:.035,
   warning:'38 mV is invalid for the old 1.575 A minimum; choose the matching complete contract.'},
 alternativeReadThreshold:{requiredIndicatedA:practicalThresholdA,slowCornerWorstIndicatedA:iMin-p.cc2Error,
   indicatedCurrentMarginA:iMin-p.cc2Error-practicalThresholdA,trueFaultCurrentMinA:practicalCurrentMin,
   healthyReadMaxV:healthyMax,faultReadMinV:practicalFaultMin,voltageThresholdV:practicalVoltageThreshold,
   marginEachV:practicalVoltageThreshold-healthyMax,
   warning:'Only under complete assumed current and voltage error budgets; actual quantization, sampling, drift, loads and dynamics are not qualified.'},
 resistorShortIdealScreenA:{addedShort:p.busMax/p.rPreMin,originalShort:p.busMax/p.rAddMin},
 lowOneAmp:{...oneA,healthyReadMaxV:oneGood,faultReadMinV:oneBad,warning:'0.85–1.10 A and ±5 mV are unqualified system assumptions; TPS2660 table at 1 V across eFuse is not a guarantee at actual ~12–30 V drop.'},
 oldSmallPulse:{...small,healthyReadMaxV:smallGood,faultReadMinV:smallBad},tests,startupNegativeControls,
 limits:['Self-test stuck-sensor branch assumes an independent diagnostic reports the fault; no such circuit is specified or verified.',
 'Absolute Kelvin accuracy, source taps, precharge disconnection, 1 ms physical stop and independent time limit remain unqualified.',
 'Healthy <=6 mOhm and faulty >=30 mOhm apply to the actual operating point. Intermediate and non-ohmic faults require separate criteria.',
 'Passive current limit only exists with both resistors intact. Shorted resistor, shorts before series elements, switch short, battery BUS voltage and fault energy require independent protection.',
 'TPS2660 fast trip threshold is not a maximum transient current guarantee; actual current and thermal behavior at large voltage drops must be established.']};
writeFileSync(fileURLToPath(new URL('./低成本主路证明比较.json',import.meta.url)),JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify(result,null,2));
