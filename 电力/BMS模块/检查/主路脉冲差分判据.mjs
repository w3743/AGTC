// Paper comparison only. No circuit, physical measurement, or protection qualification.
// Run: node 电力/BMS模块/检查/主路脉冲差分判据.mjs
import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const p = {
  healthyMaxOhm: 0.006, faultMinOhm: 0.030, prechargeMinOhm: 7.284,
  shuntTrueMinA: 4.3, shuntTrueMaxA: 5.0, nonPathCurrentMaxA: 0.005,
  baselineAbsMaxV: 0.005, differentialReadErrorMaxV: 0.010,
  sourceDisturbanceMaxV: 0.002, synchronizationErrorMaxV: 0.002,
  thresholdV: 0.080, pulseDurationMs: 40, inaGain: 0.2,
  cutoutV: 0.260, supplyMaxV: 37.8, tailMaxMs: 1,
};
const parallel = (a, b) => a * b / (a + b);
// The healthy upper bound conservatively ignores the parallel precharge path.
const iLow = p.shuntTrueMinA - p.nonPathCurrentMaxA;
const iHigh = p.shuntTrueMaxA + p.nonPathCurrentMaxA;
const error = p.baselineAbsMaxV + p.differentialReadErrorMaxV +
  p.sourceDisturbanceMaxV + p.synchronizationErrorMaxV;
const healthyUpper = iHigh * p.healthyMaxOhm + error;
const faultLower = iLow * parallel(p.faultMinOhm, p.prechargeMinOhm) - error;
const fixedOtherError = error - p.differentialReadErrorMaxV;
const readErrorLimit = Math.min(p.thresholdV - iHigh * p.healthyMaxOhm - fixedOtherError,
  iLow * parallel(p.faultMinOhm, p.prechargeMinOhm) - p.thresholdV - fixedOtherError);
const openPathCurrentMaxA = p.cutoutV / p.prechargeMinOhm;
const sourcePulseEnergyJ = p.supplyMaxV * p.shuntTrueMaxA * p.pulseDurationMs / 1000;
const sourceTailEnergyJ = p.supplyMaxV * p.shuntTrueMaxA * p.tailMaxMs / 1000;
const mainWindowEnergyJ = p.cutoutV * iHigh * p.pulseDurationMs / 1000;
const mainTailEnergyJ = p.supplyMaxV * iHigh * p.tailMaxMs / 1000;
const prechargePowerMaxW = p.cutoutV ** 2 / p.prechargeMinOhm;
// These failure tests check the required decision logic, not the existence of hardware diagnostics.
function decide({diagnosticsValid, platformValid, deltaV}) {
  if (!diagnosticsValid || !platformValid) return 'REJECT';
  return deltaV <= p.thresholdV ? 'MODEL_ACCEPT' : 'REJECT';
}
const selfTests = {
  healthyBoundary: decide({diagnosticsValid:true, platformValid:true, deltaV:healthyUpper}) === 'MODEL_ACCEPT',
  fault30mOhmBoundary: decide({diagnosticsValid:true, platformValid:true, deltaV:faultLower}) === 'REJECT',
  stuckAdcEvenIfDeltaZero: decide({diagnosticsValid:false, platformValid:true, deltaV:0}) === 'REJECT',
  saturatedReadoutEvenIfDeltaZero: decide({diagnosticsValid:false, platformValid:true, deltaV:0}) === 'REJECT',
  openMainNoCurrentPlatform: decide({diagnosticsValid:true, platformValid:false, deltaV:0}) === 'REJECT',
  openCannotReachPlatform: openPathCurrentMaxA < iLow,
};
const result = {
  status: 'CONDITIONAL_COMPARISON_HOLD_NOT_IMPLEMENTED', input: p,
  outputs: {
    mainOrPrechargeCurrentRangeA: [iLow, iHigh],
    healthyLargestDeltaReadV: healthyUpper, faultSmallestDeltaReadV: faultLower,
    thresholdV: p.thresholdV, healthyMarginV: p.thresholdV - healthyUpper,
    faultMarginV: faultLower - p.thresholdV,
    differentialReadErrorAbsoluteLimitV: readErrorLimit,
    analogOutputDeltaErrorBudgetV: p.differentialReadErrorMaxV * p.inaGain,
    openPathCurrentMaxA, openPathVoltageForMinimumPlatformV: iLow * p.prechargeMinOhm,
    sourcePulseEnergyJ, sourceTailEnergyJ, mainWindowEnergyJ, mainTailEnergyJ,
    conservativeMainWindowPlusFullVoltageTailJ: mainWindowEnergyJ + mainTailEnergyJ,
    prechargePowerMaxW, prechargePulseEnergyMaxJ: prechargePowerMaxW * p.pulseDurationMs / 1000,
  },
  selfTests,
  assumptions: [
    'One genuine 4.3–5.0 A positive battery-shunt current pulse, 40 ms; 5 mA bounds currents outside the two modeled positive paths.',
    'Pulse and baseline sample the same unsaturated Kelvin pair; each sample and shunt current are fresh, stable and synchronized; signed battery-minus-PACK voltage is used.',
    'Baseline true difference <= 5 mV; 10 mV budget applies to total *change* of readout error between the two samples, not to absolute error. Both are unqualified assumptions.',
    'A healthy path is <=6 mOhm and a fault path is >=30 mOhm during the pulse; these are comparison bounds, not vehicle ratings.',
    'Both paths share the same Kelvin terminals; precharge resistance >=7.284 Ohm. A bypass, other power source, sensor fault, offset change, or current outside budget invalidates the result.',
    'Before asserting the pulse, hard 5 A current limit, physical timeout, independent 0.260 V cutout and <=1 ms current cessation are prerequisites and remain unqualified.',
    'An 80 mV threshold with these budgets separates only the modeled examples; it does not establish rated current, thermal SOA, gate health or safe permission.',
    '6–30 mOhm is an unspecified comparison interval: a reading can be ambiguous, and acceptance there does not qualify rated operation.',
    'A fully open main path cannot supply the required current platform while the 0.260 V hard window is valid; timeout or window loss rejects, rather than assuming a 4.3 A sample.',
    'Source energy is a supply-side bound. Main-path bound applies only while true voltage stays <=0.260 V and assumes <=1 ms full-voltage tail; stored energy and overvoltage require separate analysis.',
  ],
};
if (!(healthyUpper < p.thresholdV && p.thresholdV < faultLower && Object.values(selfTests).every(Boolean))) throw new Error('Comparison or decision self-test failed');
const out = fileURLToPath(new URL('./主路脉冲差分判据.json', import.meta.url));
writeFileSync(out, JSON.stringify(result, null, 2) + '\n', 'utf8');
console.log(JSON.stringify(result, null, 2));
