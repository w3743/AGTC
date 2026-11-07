if(globalThis.pumpNotesCreated)throw Error('Notes already created');
const texts=[
[150,210,'水泵驱动 / 21 V 有刷直流水泵',22,true],
[150,255,'9S BUS 22.5-37.8 V | 单向 PWM + 电制动 | CAN 控制与状态回报 | 设计输入：运行最大约 3 A',11,false],
[150,320,'01  辅助电源  BUS → 5 V / 0.3 A',15,true],
[2300,320,'02  CAN 收发器与接口保护',15,true],
[150,930,'03  并桥驱动 / VREF / 电流镜 / 默认态',15,true],
[2300,920,'04  电流 / 母线电压 / 板温采样',15,true],
[2300,1380,'05  WeAct Normal QFN48 STM32G431CBU6 核心板底座',15,true],
[150,1790,'06  母线入口与本地储能',15,true],
[2300,1820,'核心板：P1 ↔ J_MCU_A，P2 ↔ J_MCU_B；同号连接。',11,false],
[2300,1850,'PA8/TIM1_CH1 AF6 → IN2；PB5 → IN1；PB7 → nSLEEP。',11,false],
[2300,1880,'PC10 → CAN_STB；PC11 ← nFAULT；PB8/PB9 = FDCAN1 RX/TX AF9。',11,false],
[2300,1910,'PA0/ADC1_IN1 = 电流；PA1/ADC12_IN2 = 母线；PA2/ADC1_IN3 = 板温。',11,false],
[2300,1940,'PB8 兼 BOOT0：烧录 nSWBOOT0=0、nBOOT0=1，读回并验证复位启动。',11,false],
[150,2070,'控制与装配约束',15,true],
[150,2110,'RUN：nSLEEP=1，IN1=1，IN2=反相PWM，20 kHz；D=u×min(0.95,21/VBUS)。',11,false],
[150,2140,'BRAKE：nSLEEP=1，IN1=IN2=1（高侧短接）；COAST：IN1=IN2=0；复位默认休眠。',11,false],
[150,2170,'并桥：MODE1=MODE2=DVDD；VREF=1.65 V；IPROPI 1.5k 唯一电阻；名义运行斩波约5.19 A。',11,false],
[150,2200,'RSVD 32/36/38/39 悬空；EP45接地铜与散热过孔。IPROPI运行斩波不等于制动限流。',11,false],
[150,2230,'J_BUS 之前在线束串接 F_BUS 0997005.WXN（5 A/58 VDC）及58 V保险座；J_MOTOR 两端均为桥输出。',11,false],
[150,2260,'R_TERM 默认DNP，仅总线物理末端装120R；CAN失联/过温/故障本地停机，故障恢复后重新使能。',11,false],
[150,2290,'3V3来自核心板；USB和母线并用须处理5V回灌。C_VM_BULK高40mm；插座方向与高度按实物核对。',11,false],
[2300,2070,'设计状态与边界',15,true],
[2300,2110,'49个板内器件；核心板、保险/保险座、线端配件列在整套BOM。无阀门驱动。',11,false],
[2300,2140,'SMCJ40A标称脉冲钳位64.5V；低于70V绝对最大，高于60V推荐工作上限。',11,false],
[2300,2170,'本图为原理图设计；启动、电制动、供电稳定性与实际热性能在样机阶段测试。',11,false],
[2300,2200,'电机停转不等于水路瞬时断水；本设计不冻结停转毫秒数。',11,false],
[2300,2260,'水泵驱动模块 | 2026-10-18 | 当前原生工程原位维护',11,false]];
for(const [x,y,t,size,bold] of texts)await eda.sch_PrimitiveText.create(x,2500-y,t,0,null,null,size,bold,false,false,1);
globalThis.pumpNotesCreated=true;
const cleanup={alreadyRemoved:true};
await eda.sch_Document.save();return {textCount:texts.length,cleanup};
