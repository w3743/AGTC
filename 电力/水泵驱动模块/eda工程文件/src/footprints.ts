// 嘉立创EDA/EasyEDA官方库 (JLCEDA/EasyEDA Official Library)
// https://lceda.cn/  https://easyeda.com
// Frozen pad geometry. Dimensions in millimetres; angles in degrees.
// Native assets retain their land patterns. Supplier URLs identify exact catalog bindings.
import type { PumpReference } from "./parts"

export type Land = readonly [
  pin: number, x: number, y: number, width: number, height: number,
  rotation: number, shape: "rect" | "ellipse", drillDiameter: number,
]

type LandPattern = { title: string; source: string; pads: readonly Land[] }

const landPatterns = {
  C_5V_IN: {
    title: "CAP-SMD_BD10.0-L10.3-W10.3-FD",
    source: "电力/元件封装库/原生片段/FOOTPRINT/4e28cbe83b6a4ecef673afd8cf105ed06bdd998df3d771dd9c69dd8fe2335caf.jsonl",
    pads: [
      [1, -4.499864, 0.0, 4.499991, 1.6500094, 0, "rect", 0],
      [2, 4.499864, 0.0, 4.499991, 1.6500094, 0, "rect", 0],
    ],
  },
  C_5V_IN_HF: {
    title: "C0805",
    source: "电力/元件封装库/原生片段/FOOTPRINT/fc34dfb0864c5b276330ff26b60d083f2a8c18a65a07490c7932982a9bf03744.jsonl",
    pads: [
      [1, -0.999998, 0.0, 1.4100048, 1.35001, 0, "rect", 0],
      [2, 0.999998, 0.0, 1.4100048, 1.35001, 0, "rect", 0],
    ],
  },
  C_5V_OUT: {
    title: "CAP-TH_BD8.0-P3.50-D0.6-FD",
    source: "电力/元件封装库/原生片段/FOOTPRINT/70d25a58a5e9c9bce8a3ec80dc5510fdc4f7395f113cb0a05534e838793363fe.jsonl",
    pads: [
      [1, -1.75006, 0.0, 1.5999968, 1.5999968, 270, "ellipse", 0.999998],
      [2, 1.75006, 0.0, 1.5999968, 1.5999968, 270, "ellipse", 0.999998],
    ],
  },
  C_5V_OUT_HF: {
    title: "C0603",
    source: "电力/元件封装库/原生片段/FOOTPRINT/244dbd312e3677ad93ceca56386cc0a2a9723c2b820af6931e4cc4146e1274dd.jsonl",
    pads: [
      [1, -0.700024, 0.0, 0.7999984, 0.8999982, 0, "rect", 0],
      [2, 0.700024, 0.0, 0.7999984, 0.8999982, 0, "rect", 0],
    ],
  },
  C_CAN_ISO_BULK: {
    title: "C1210",
    source: "电力/元件封装库/原生片段/FOOTPRINT/22cf8b0b9b347a0f6e4f4549a2621bf579deda82239415e774e17fdecebc5ac3.jsonl",
    pads: [
      [1, -1.517396, 0.0, 1.634998, 2.6999946, 0, "rect", 0],
      [2, 1.517396, 0.0, 1.634998, 2.6999946, 0, "rect", 0],
    ],
  },
  C_VM_BULK: {
    title: "CAP-TH_BD12.5-P5.00-D0.6-FD",
    source: "电力/元件封装库/原生片段/FOOTPRINT/11e0eca92e577701c12f8928b34732393339677a250a400c91c92efd05a6334b.jsonl",
    pads: [
      [1, -2.499995, 0.0, 2.0999958, 2.0999958, 0, "ellipse", 1.1999976],
      [2, 2.499995, 0.0, 2.0999958, 2.0999958, 0, "ellipse", 1.1999976],
    ],
  },
  D_5V: {
    title: "SMC_L6.9-W5.9-LS7.9-RD",
    source: "电力/元件封装库/原生片段/FOOTPRINT/570194ab49c80c43abf7efc61e4e7ad909c6c770ac4b5c1f39cfae701a7f93db.jsonl",
    pads: [
      [1, -3.686302, 0.0, 2.6999946, 3.2999934, 0, "rect", 0],
      [2, 3.686302, 0.0, 2.6999946, 3.2999934, 0, "rect", 0],
    ],
  },
  D_CAN: {
    title: "SOT-23-3_L2.9-W1.3-P1.90-LS2.4-BR",
    source: "电力/元件封装库/原生片段/FOOTPRINT/3f672dc5bd4b455a47b0f542cbff8c7f4eeb463a07070ae842248fbb412e066f.jsonl",
    pads: [
      [1, 0.999998, -0.94996, 0.999998, 0.6500114, 0, "rect", 0],
      [2, 0.999998, 0.94996, 0.999998, 0.6500114, 0, "rect", 0],
      [3, -0.999998, 0.0, 0.999998, 0.6500114, 0, "rect", 0],
    ],
  },
  D_PUMP_FW: {
    title: "SMC_L7.1-W6.2-LS8.1-R-RD",
    source: "https://easyeda.com/api/products/C19229/components",
    pads: [
      [1, 3.435096, 0.0, 3.0299914, 3.8200076, 180.0, "rect", 0.0],
      [2, -3.435096, 0.0, 3.0299914, 3.8200076, 180.0, "rect", 0.0],
    ],
  },
  J_BUS: {
    title: "CONN-TH_P5.08_2EDGVC-5.08-02P-14-100A-H",
    source: "电力/元件封装库/原生片段/FOOTPRINT/d26903829db3a20fde754d48fc8a3cf064f3d59c77375d9e4d91ef5d7c586014.jsonl",
    pads: [
      [1, -2.54, 0.0, 2.3999952, 2.3999952, 0, "rect", 1.700022],
      [2, 2.54, 0.0, 2.3999952, 2.3999952, 0, "ellipse", 1.700022],
    ],
  },
  J_CAN: {
    title: "CONN-TH_3P-P2.50_A2501WV2-3P",
    source: "电力/元件封装库/原生片段/FOOTPRINT/1e44edbca6d34d7028f07061e7bf0c26018cc6eca63636275375ff411d2d7d31.jsonl",
    pads: [
      [1, -2.499995, 0.0, 1.5500096, 1.5500096, 180, "ellipse", 0.9000236],
      [2, 0.000127, 0.0, 1.5500096, 1.5500096, 180, "ellipse", 0.9000236],
      [3, 2.499995, 0.0, 1.5500096, 1.5500096, 180, "ellipse", 0.9000236],
    ],
  },
  J_MCU_A: {
    title: "HDR-TH_24P-P2.54-V-F-R2-C12-S2.54-2",
    source: "电力/元件封装库/原生片段/FOOTPRINT/d6ab413131906500cbe4a9c10328576c7fe5b68c0ed1a8aad658489402c3746f.jsonl",
    pads: [
      [1, -13.97, -1.27, 1.5999968, 1.5999968, 0, "rect", 1.0499852],
      [2, -13.97, 1.27, 1.5999968, 1.5999968, 0, "ellipse", 1.0499852],
      [3, -11.43, -1.27, 1.5999968, 1.5999968, 0, "ellipse", 1.0499852],
      [4, -11.43, 1.27, 1.5999968, 1.5999968, 0, "ellipse", 1.0499852],
      [5, -8.89, -1.27, 1.5999968, 1.5999968, 0, "ellipse", 1.0499852],
      [6, -8.89, 1.27, 1.5999968, 1.5999968, 0, "ellipse", 1.0499852],
      [7, -6.35, -1.27, 1.5999968, 1.5999968, 0, "ellipse", 1.0499852],
      [8, -6.35, 1.27, 1.5999968, 1.5999968, 0, "ellipse", 1.0499852],
      [9, -3.81, -1.27, 1.5999968, 1.5999968, 0, "ellipse", 1.0499852],
      [10, -3.81, 1.27, 1.5999968, 1.5999968, 0, "ellipse", 1.0499852],
      [11, -1.27, -1.27, 1.5999968, 1.5999968, 0, "ellipse", 1.0499852],
      [12, -1.27, 1.27, 1.5999968, 1.5999968, 0, "ellipse", 1.0499852],
      [13, 1.27, -1.27, 1.5999968, 1.5999968, 0, "ellipse", 1.0499852],
      [14, 1.27, 1.27, 1.5999968, 1.5999968, 0, "ellipse", 1.0499852],
      [15, 3.81, -1.27, 1.5999968, 1.5999968, 0, "ellipse", 1.0499852],
      [16, 3.81, 1.27, 1.5999968, 1.5999968, 0, "ellipse", 1.0499852],
      [17, 6.35, -1.27, 1.5999968, 1.5999968, 0, "ellipse", 1.0499852],
      [18, 6.35, 1.27, 1.5999968, 1.5999968, 0, "ellipse", 1.0499852],
      [19, 8.89, -1.27, 1.5999968, 1.5999968, 0, "ellipse", 1.0499852],
      [20, 8.89, 1.27, 1.5999968, 1.5999968, 0, "ellipse", 1.0499852],
      [21, 11.43, -1.27, 1.5999968, 1.5999968, 0, "ellipse", 1.0499852],
      [22, 11.43, 1.27, 1.5999968, 1.5999968, 0, "ellipse", 1.0499852],
      [23, 13.97, -1.27, 1.5999968, 1.5999968, 0, "ellipse", 1.0499852],
      [24, 13.97, 1.27, 1.5999968, 1.5999968, 0, "ellipse", 1.0499852],
    ],
  },
  L_5V: {
    title: "IND-SMD_L8.0-W8.0_SWPA8040S",
    source: "电力/元件封装库/原生片段/FOOTPRINT/4cd9ae96250b9221df5131f51ca51abe98e1450ce31047e098c64b18167b47dc.jsonl",
    pads: [
      [1, 3.377946, 0.0, 2.9239972, 6.4200024, 0, "rect", 0],
      [2, -3.377946, 0.0, 2.9239972, 6.4200024, 0, "rect", 0],
    ],
  },
  NTC_BOARD: {
    title: "RES-TH_L2.5-W2.5-P1.70-D0.3",
    source: "电力/元件封装库/原生片段/FOOTPRINT/71735347278a8b35cf0f80c268a8c9b61a044462f97edf429e395292fe82bf4e.jsonl",
    pads: [
      [1, -0.850011, 0.0, 0.999998, 0.999998, 0, "ellipse", 0.5500116],
      [2, 0.850011, 0.0, 0.999998, 0.999998, 0, "ellipse", 0.5500116],
    ],
  },
  Q_BRAKE: {
    title: "TO-252-2_L6.6-W6.1-P4.60-LS10.0-BR-CW",
    source: "https://easyeda.com/api/products/C29781153/components",
    pads: [
      [1, 3.292094, -2.29997, 2.7999944, 1.2999974, 0.0, "rect", 0.0],
      [2, -3.292094, 0.0, 6.5000124, 5.999988, 0.0, "rect", 0.0],
      [3, 3.292094, 2.29997, 2.7999944, 1.2999974, 0.0, "rect", 0.0],
    ],
  },
  Q_BRAKE_CTL: {
    title: "SOT-23-3_L2.9-W1.6-P1.90-LS2.8-BR",
    source: "https://easyeda.com/api/products/C408396/components",
    pads: [
      [1, 1.235075, -0.94996, 1.0700004, 0.5999988, 0.0, "rect", 0.0],
      [2, 1.235075, 0.94996, 1.0700004, 0.5999988, 0.0, "rect", 0.0],
      [3, -1.235075, 0.0, 1.0700004, 0.5999988, 0.0, "rect", 0.0],
    ],
  },
  Q_PWM: {
    title: "TO-252-2_L6.6-W6.1-P4.57-LS9.9-TL-CW",
    source: "https://easyeda.com/api/products/C108638/components",
    pads: [
      [1, -3.291967, 2.272792, 2.7999944, 1.2999974, 0.0, "rect", 0.0],
      [2, 3.291967, 0.0, 6.5000124, 5.999988, 0.0, "rect", 0.0],
      [3, -3.291967, -2.272792, 2.7999944, 1.2999974, 0.0, "rect", 0.0],
    ],
  },
  R_ADC_BUS: {
    title: "R0603",
    source: "电力/元件封装库/原生片段/FOOTPRINT/b00aae913e137469b6aa14ab6259fa804e7e799a565845af0448192806239beb.jsonl",
    pads: [
      [1, -0.753364, 0.0, 0.8064754, 0.8640064, 0, "rect", 0],
      [2, 0.753364, 0.0, 0.8064754, 0.8640064, 0, "rect", 0],
    ],
  },
  R_BRAKE_GATE_LOW: {
    title: "R1206",
    source: "电力/元件封装库/原生片段/FOOTPRINT/226319c662fccb215d2c046621b348c29e0029d74e7a25d69e782d8ee61c3aab.jsonl",
    pads: [
      [1, -1.478788, 0.0, 1.207516, 1.7010126, 0, "rect", 0],
      [2, 1.478788, 0.0, 1.207516, 1.7010126, 0, "rect", 0],
    ],
  },
  R_PUMP_SHUNT: {
    title: "R2512",
    source: "https://easyeda.com/api/products/C2924538/components",
    pads: [
      [1, -3.066288, 0.0, 1.2825222, 3.4560002, 0.0, "rect", 0.0],
      [2, 3.066288, 0.0, 1.2825222, 3.4560002, 0.0, "rect", 0.0],
    ],
  },
  R_TERM: {
    title: "RES-TH_BD1.9-L3.5-P7.50-D0.5",
    source: "电力/元件封装库/原生片段/FOOTPRINT/779635f488345cfdec4d612beff6118b3613ca55dc3a16b2131db758d733f290.jsonl",
    pads: [
      [1, -3.750056, 0.0, 1.524, 1.524, 0, "ellipse", 0.9000236],
      [2, 3.750056, 0.0, 1.524, 1.524, 0, "ellipse", 0.9000236],
    ],
  },
  U_5V: {
    title: "TO-252-5_L6.6-W5.8-P1.27-TL",
    source: "电力/元件封装库/原生片段/FOOTPRINT/bce6defd730cd2edb0e03446d6d809a2d009039332c7ac2c2557ae038e177ddd.jsonl",
    pads: [
      [1, -3.5433, 2.54, 0.8255, 2.1999956, 90, "rect", 0],
      [2, -3.5433, 1.27, 0.8255, 2.1999956, 90, "rect", 0],
      [3, -3.5433, 0.0, 0.8255, 2.1999956, 90, "rect", 0],
      [4, -3.5433, -1.27, 0.8255, 2.1999956, 90, "rect", 0],
      [5, -3.5433, -2.54, 0.8255, 2.1999956, 90, "rect", 0],
      [6, 3.5433, 0.0, 6.2103, 6.2103, 90, "rect", 0],
    ],
  },
  U_CAN: {
    title: "LGA-16_L5.2-W4.7-P0.65-TL_CA-IS2062A-FRANK2",
    source: "电力/元件封装库/原生片段/FOOTPRINT/44ad6a9a911419e3c760148eccf5ba8a164bd6f78c829727307b8da075c3ab38.jsonl",
    pads: [
      [1, -2.192528, 2.275078, 0.9849866, 0.3640074, 0, "rect", 0],
      [2, -2.192528, 1.625092, 0.9849866, 0.3640074, 0, "rect", 0],
      [3, -2.192528, 0.975106, 0.9849866, 0.3640074, 0, "rect", 0],
      [4, -2.192528, 0.32512, 0.9849866, 0.3640074, 0, "rect", 0],
      [5, -2.192528, -0.32512, 0.9849866, 0.3640074, 0, "rect", 0],
      [6, -2.192528, -0.975106, 0.9849866, 0.3640074, 0, "rect", 0],
      [7, -2.192528, -1.625092, 0.9849866, 0.3640074, 0, "rect", 0],
      [8, -2.192528, -2.275078, 0.9849866, 0.3640074, 0, "rect", 0],
      [9, 2.192528, -2.275078, 0.9849866, 0.3640074, 0, "rect", 0],
      [10, 2.192528, -1.625092, 0.9849866, 0.3640074, 0, "rect", 0],
      [11, 2.192528, -0.975106, 0.9849866, 0.3640074, 0, "rect", 0],
      [12, 2.192528, -0.32512, 0.9849866, 0.3640074, 0, "rect", 0],
      [13, 2.192528, 0.32512, 0.9849866, 0.3640074, 0, "rect", 0],
      [14, 2.192528, 0.975106, 0.9849866, 0.3640074, 0, "rect", 0],
      [15, 2.192528, 1.625092, 0.9849866, 0.3640074, 0, "rect", 0],
      [16, 2.192528, 2.275078, 0.9849866, 0.3640074, 0, "rect", 0],
    ],
  },
  U_PWM: {
    title: "SOT-23-5_L2.9-W1.6-P0.95-LS2.8-BL",
    source: "https://easyeda.com/api/products/C53436462/components",
    pads: [
      [1, -0.94996, -1.100074, 0.5999988, 0.999998, 0.0, "rect", 0.0],
      [2, 0.0, -1.100074, 0.5999988, 0.999998, 0.0, "rect", 0.0],
      [3, 0.94996, -1.100074, 0.5999988, 0.999998, 0.0, "rect", 0.0],
      [4, 0.94996, 1.100074, 0.5999988, 0.999998, 0.0, "rect", 0.0],
      [5, -0.94996, 1.100074, 0.5999988, 0.999998, 0.0, "rect", 0.0],
    ],
  },
} satisfies Record<string, LandPattern>

const patternByReference = {
  C_5V_IN: "C_5V_IN",
  C_5V_IN_HF: "C_5V_IN_HF",
  C_5V_OUT: "C_5V_OUT",
  C_5V_OUT_HF: "C_5V_OUT_HF",
  C_ADC_BUS: "C_5V_OUT_HF",
  C_ADC_I: "C_5V_OUT_HF",
  C_ADC_NTC: "C_5V_OUT_HF",
  C_CAN_ISOIN_1U: "C_5V_OUT_HF",
  C_CAN_ISO_BULK: "C_CAN_ISO_BULK",
  C_CAN_ISO_HF: "C_5V_OUT_HF",
  C_CAN_VCC: "C_5V_OUT_HF",
  C_CAN_VDDP_BULK: "C_CAN_ISO_BULK",
  C_CAN_VIO: "C_5V_OUT_HF",
  C_DRV_VCC: "C_5V_OUT_HF",
  C_DRV_VM1: "C_5V_IN_HF",
  C_FB_FF: "C_5V_OUT_HF",
  C_VM_BULK: "C_VM_BULK",
  D_5V: "D_5V",
  D_BUS: "D_5V",
  D_CAN: "D_CAN",
  D_PUMP_FW: "D_PUMP_FW",
  J_BUS: "J_BUS",
  J_CAN: "J_CAN",
  J_MCU_A: "J_MCU_A",
  J_MCU_B: "J_MCU_A",
  J_MOTOR: "J_BUS",
  L_5V: "L_5V",
  NTC_BOARD: "NTC_BOARD",
  Q_BRAKE: "Q_BRAKE",
  Q_BRAKE_CTL: "Q_BRAKE_CTL",
  Q_PWM: "Q_PWM",
  R_ADC_BUS: "R_ADC_BUS",
  R_ADC_I: "R_ADC_BUS",
  R_BRAKE_BASE: "R_ADC_BUS",
  R_BRAKE_GATE_HIGH: "R_ADC_BUS",
  R_BRAKE_GATE_LOW: "R_BRAKE_GATE_LOW",
  R_BUS_BOTTOM: "R_ADC_BUS",
  R_BUS_TOP1: "R_ADC_BUS",
  R_BUS_TOP2: "R_ADC_BUS",
  R_CAN_TXD_PU: "R_ADC_BUS",
  R_FB_BOTTOM: "R_ADC_BUS",
  R_FB_TOP: "R_ADC_BUS",
  R_IN1_PD: "R_ADC_BUS",
  R_IN2_PD: "R_ADC_BUS",
  R_NTC_PULL: "R_ADC_BUS",
  R_PUMP_SHUNT: "R_PUMP_SHUNT",
  R_PWM_GATE_SER: "R_BRAKE_GATE_LOW",
  R_SLEEP: "R_ADC_BUS",
  R_TERM: "R_TERM",
  U_5V: "U_5V",
  U_CAN: "U_CAN",
  U_PWM: "U_PWM",
} satisfies Record<PumpReference, keyof typeof landPatterns>

export function landPatternFor(ref: PumpReference): LandPattern {
  return landPatterns[patternByReference[ref]]
}
