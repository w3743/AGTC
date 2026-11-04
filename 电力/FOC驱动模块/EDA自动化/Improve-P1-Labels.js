const page = await eda.dmt_Schematic.getCurrentSchematicPageInfo();
if (page?.uuid !== '90aa6c46493624cf') throw new Error('Wrong FOC page');

// Move rendered labels only. Electrical properties and primitive positions are untouched.
const labels = [
  ['22c484f646f478e6', 1000, 2088], ['b1d35ee15ed67037', 1000, 2076],
  ['898ab5d23d6dce0d', 1505, 2080], ['b41763da4883b120', 1505, 2068],
  ['115ef0c342d41358', 1028, 1843], ['eef30641b66aaf31', 1028, 1832],
  ['40a5c8eff7de11af', 1137, 1844], ['4b7723186357f8d4', 1137, 1832],
  ['00c3eaf0c1e4b6e2', 1820, 1980], ['7d29ebcc62e269bd', 2000, 1980],
  ['69eba0e97e9671cd', 1717, 804], ['14ae83288acac7cf', 1717, 793],
  ['a601a3df70e94fe4', 1777, 898], ['6b0b85260dd5be3c', 1777, 887],
  ['ffe17c9e2e2776ac', 1777, 1068], ['8327974b01869d87', 1777, 1057],
  ['813a1d90b84f9776', 2700, 1700], ['5fa3255b15275f2f', 2702.5, 1673],
  ['222a06b74ec99de5', 2700, 1850], ['4769c30fe447b094', 2702.5, 1823],
  ['dbefe4ad6edfd73b', 2700, 2000], ['1e36a35e5bf83d3a', 2702.5, 1973],
  ['98dcb3a032ade277', 1197.5, 1907], ['03aa5b750ab36893', 1387.5, 1883],
];
for (const [id, x, y] of labels) {
  if (!await eda.sch_PrimitiveAttribute.modify(id, {x, y})) throw new Error('Cannot move ' + id);
}
// The pin index supplies these names beside the connectors. Retain the net-port names
// in the native document while suppressing text that overlaps adjacent signals.
const hidden = [
  '4469b9988e3ab6f1', '537659916cc7c8d0', 'bacd3953a95ddd61',
  '2a15dbc4c7cd54bb', 'd1c1c4ebfab7ab00', 'bf5b4415b9598927',
  '6a74cb0bb8b7a4ef', '0b2aee3915aae7e9', 'b3c0e8c86915eb0d',
  '5fc608f6197d0b53', 'b57ed59a0eb81dfb',
];
for (const id of hidden) {
  if (!await eda.sch_PrimitiveAttribute.modify(id, {valueVisible: false})) throw new Error('Cannot hide ' + id);
}

const notes = [
  [120, 2272, '保护信号走向', 15, true],
  [120, 2245, '三相分流 → U1 CSA → ADC 与 U3 低端窗口 → BREAK_N。', 12],
  [120, 2220, 'BUS → U7 比较器 → BUSOV_N → 双管转接 → BREAK_N。', 12],
  [120, 2195, 'BREAK_N → J_MCU_L.5（低有效）。门驱使能另走 DRV_ENABLE。', 12],
  [120, 2170, 'J_PWR_SENSE.6/.8 仅作故障线观察或开漏拉低。', 12],
  [1730, 2295, 'G431 排针接线索引（按图示针号，方向以实板核对为准）', 13, true],
  [1730, 2273, 'J_MCU_L  1/2 5V_BUCK   3/4 GND_SYS   5 BREAK_N', 10],
  [1730, 2258, '6/7/8 PWM_UL/VL/WL   10/11/12 PWM_UH/VH/WH', 10],
  [1730, 2243, '13/14 FDCAN1_RX/TX   15 HALL_A   18 HALL_B   20 CAN_STB', 10],
  [1730, 2228, '9/16/17/19/21–24 NC', 10],
  [1730, 2207, 'J_MCU_R  1/2 MCU_3V3   3/4 GND_SYS   5 HALL_C   6 DRV_ENABLE', 10],
  [1730, 2192, '8 VREF_CORE   9 SAFE_ADC   10 TEMP_MOTOR_ADC   12 BUS_ADC', 10],
  [1730, 2177, '11/13/14/16 DRV_SPI_MOSI/SCK/MISO/CS_N', 10],
  [1730, 2162, '15 SO_V_ADC   17 SO_U_ADC   18 TEMP_MOS_ADC   20 SO_W_ADC', 10],
  [1730, 2147, '7/19/21–24 NC', 10],
  [2430, 1000, '外部接口与上电顺序', 15, true],
  [2430, 973, 'J_LOAD_POWER_VALID.1 偏置，.2 许可。仅 A 板安装偏置电阻。', 12],
  [2430, 948, 'BMS 主回路成立后发许可，FOC 先建立 5 V，再初始化 PWM 与门驱。', 12],
  [2430, 923, 'J_CAN 1=GND_SYS  2=CANH_CONN  3=CANL_CONN', 12],
  [2430, 898, 'J_HALL 1=5V_BUCK  2/3/4=HALL_A/B/C_RAW  5=GND_SYS', 12],
  [2430, 873, 'J_PWR_SENSE 1/2/3=5V_BUCK/MCU_3V3/GND_SYS', 12],
  [2430, 848, '4/5=SAFE_ADC/BUS_ADC，6/8=BREAK_N，7=DRV_ENABLE', 12],
  [1050, 830, 'CBYP 每相两颗近桥去耦', 13, true],
  [1050, 803, 'CBUS1–3 为板外 10 mF/100 V。', 12],
  [1050, 780, '两个 FOC 分别经各自 F_MAIN 接公共母线。', 12],
  [3265, 650, 'U6 独立 BOV/过流检测供电（DUMP_5V）', 13, true],
];
for (const [x, y, content, size, bold] of notes) {
  if (!await eda.sch_PrimitiveText.create(x, y, content, 0, null, null, size, !!bold)) {
    throw new Error('Cannot place ' + content);
  }
}
if (!await eda.sch_Document.save()) throw new Error('Saving the copied sheet failed');
return {movedLabels: labels.length, hiddenOverprints: hidden.length, addedNotes: notes.length};
