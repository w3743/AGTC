// The orchestration prepends a plan derived from official geometry and the pin contract.
const page = await eda.dmt_Schematic.getCurrentSchematicPageInfo();
if (page?.uuid !== '90aa6c46493624cf') throw new Error('Wrong FOC page');

if (plan.removeWires.length && !await eda.sch_PrimitiveWire.delete(plan.removeWires)) {
  throw new Error('Could not remove old pin branches');
}
if (plan.removePorts.length && !await eda.sch_PrimitiveComponent.delete(plan.removePorts)) {
  throw new Error('Could not remove obsolete ports');
}
for (const [id, net, line, port] of plan.keepWires) {
  stage = {operation: 'retained wire', net, line};
  if (!await eda.sch_PrimitiveWire.modify(id, {net, line})) {
    throw new Error('Could not preserve local branch ' + net);
  }
  stage = {operation: 'retained port', net, port};
  if (port && !await eda.sch_PrimitiveComponent.createNetPort('BI', net, ...port, false)) {
    throw new Error('Could not name local branch ' + net);
  }
}

for (const [ref, id, x, y, expected, passive] of plan.moves) {
  stage = {operation: 'move', ref};
  const placement = passive ? {x, y, rotation: 0} : {x, y};
  if (!await eda.sch_PrimitiveComponent.modify(id, placement)) throw new Error('Could not move ' + ref);
  const pins = await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(id);
  for (const p of pins) {
    const number = String(p.getState_PinNumber());
    const net = expected[number];
    if (!net) {
      if (!p.getState_NoConnected()) throw new Error('NC marker lost ' + ref + '.' + number);
      continue;
    }
    const px = p.getState_X();
    const py = p.getState_Y();
    const angle = p.getState_Rotation();
    const length = 150;
    const directions = {0: [length, 0], 90: [0, length], 180: [-length, 0], 270: [0, -length]};
    if (!directions[angle]) throw new Error('Unsupported pin direction ' + ref + '.' + number);
    const [dx, dy] = directions[angle];
    const stop = [px + dx, py + dy];
    stage = {operation: 'new wire', ref, number, net, px, py};
    const wire = await eda.sch_PrimitiveWire.create([[px, py, ...stop]], net);
    if (!wire) throw new Error('Could not draw labelled stub ' + ref + '.' + number);
    stage.operation = 'new port';
    if (!await eda.sch_PrimitiveComponent.createNetPort('BI', net, ...stop, angle, false)) {
      throw new Error('Could not name relocated pin ' + ref + '.' + number);
    }
  }
}

const labels = [
  [100, 44, 'U1  三相门驱与降压针网', 17, true],
  [100, 17, '左侧为泵、门极与采样，右侧为控制、故障和降压。', 12],
  [100, -303, '门极输出经4.7Ω到相桥。SH/PH为源参考，SO送采样。', 12],
  [100, -328, 'EN为独立使能，nFAULT并入BREAK_N。降压与旁路见上方。', 12],
  [1125, 44, 'U3  三相低端电流窗口', 17, true],
  [1125, 17, 'NEG_U/V/W来自各相CSA，VNEG_REF为比较阈值。', 12],
  [1125, -230, '三个开集电极输出共接BREAK_N。4OUT保留NC。', 12],
  [1125, -255, '过压U7另经双管转接汇入同一故障网。', 12],
  [1900, 44, 'J_MCU_L  G431 电源、PWM 与总线', 17, true],
  [1900, 17, '按针号连接。PB12/TIM1_BKIN为低有效。', 12],
  [1900, -230, '1/2供5V，3/4为GND_SYS，5为BREAK_N。', 12],
  [1900, -255, 'PA8/9/10为高侧PWM，PB13/14/15为低侧。', 12],
  [2690, 44, 'J_MCU_R  G431 使能、SPI 与采样', 17, true],
  [2690, 17, '按针号连接。PB11经6脚控制门驱使能。', 12],
  [2690, -230, '1/2为MCU_3V3，3/4为GND_SYS，8为VREF_CORE。', 12],
  [2690, -255, '装配前核核心板版本和Pin 1方向。', 12],
  [110, 2280, 'CSA → ADC及U3。BUS → U7 → BREAK_N；至J_MCU_L.5。', 14, true],
  [1780, 2280, '两组G431排针移至下方，按同名网接温度、Hall与CAN。', 14, true],
  [2470, 930, '门驱U1见下方针网。充电泵及Buck按同名网接入。', 14, true],
];
if (plan.support) labels.push(
  [1125, -300, 'U3  本地DUMP_5V去耦', 12, true],
  [100, -394, 'U1 外围  /  充电泵、去耦与降压', 17, true],
  [100, -420, '飞跨、VCP、VGLS、DVDD与VREF去耦按本排网名接U1。', 12],
  [100, -527, 'BUS输入去耦。VM/VDRAIN与Buck VIN共接BUS。', 12],
  [100, -638, 'Buck：BUCK_BST至SW，续流二极管至GND_SYS，电感至5V_BUCK。', 12],
  [100, -748, 'Buck输出去耦、纹波注入及反馈分压。', 12],
);
for (const [x, y, label, size, bold] of labels) {
  if (!await eda.sch_PrimitiveText.create(x, y, label, 0, null, null, size, !!bold)) {
    throw new Error('Could not place panel label');
  }
}
if (!await eda.sch_Document.save()) throw new Error('Saving the copied sheet failed');
return {relocated: plan.moves.map(move => move[0]), pruned: plan.keepWires.length,
        removedWireGroups: plan.removeWires.length, removedPorts: plan.removePorts.length};
