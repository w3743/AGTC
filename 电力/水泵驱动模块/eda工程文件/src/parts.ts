// Editable TSX input. The Python comparison builds its reference independently.
// Layout and local wiring live in the five function modules.

export type PartDefinition = {
  mpn: string
  lcsc: string | null
  footprint: string
  label: string
  symbol: "capacitor" | "resistor" | "inductor" | "chip" | "tvs" | "schottky" | "nmos" | "pmos" | "npn"
  pins: Record<number, string | null>
  roles?: Record<number, string>
  polarized?: boolean
  dnp?: boolean
  note?: string
}

export const parts = {
  // PowerAndProtection
  J_BUS: {
    mpn: "2EDGVC-5.08-02P-14-100A(H)", lcsc: "C581328", footprint: "CONN-TH_P5.08_2EDGVC-5.08-02P-14-100A-H",
    symbol: "chip", label: "2EDGVC-5.08-02P-14-100A(H)",
    pins: { 1: "BUS_M", 2: "GND_SYS" },
  },
  C_VM_BULK: {
    mpn: "UHW2A471MHD", lcsc: "C1585392", footprint: "CAP-TH_BD12.5-P5.00-D0.6-FD",
    symbol: "capacitor", label: "470uF",
    polarized: true,
    pins: { 1: "BUS_M", 2: "GND_SYS" },
  },
  D_BUS: {
    mpn: "SMCJ40A", lcsc: "C840777", footprint: "SMC_L6.9-W5.9-LS7.9-RD",
    symbol: "tvs", label: "SMCJ40A",
    pins: { 1: "BUS_M", 2: "GND_SYS" },
  },

  // Buck5V
  C_5V_IN: {
    mpn: "EMK1KM330GB0D00T", lcsc: null, footprint: "CAP-SMD_BD10.0-L10.3-W10.3-FD",
    symbol: "capacitor", label: "33uF",
    polarized: true,
    pins: { 1: "BUS_M", 2: "GND_SYS" },
  },
  C_5V_IN_HF: {
    mpn: "0805B104K101NT", lcsc: "C89258", footprint: "C0805",
    symbol: "capacitor", label: "100nF",
    pins: { 1: "BUS_M", 2: "GND_SYS" },
  },
  U_5V: {
    mpn: "XL7015E1", lcsc: "C73013", footprint: "TO-252-5_L6.6-W5.8-P1.27-TL",
    symbol: "chip", label: "XL7015E1",
    pins: { 1: "BUS_M", 2: "XL_SW", 3: "GND_SYS", 4: "XL_FB", 5: "GND_SYS", 6: "GND_SYS" },
    roles: { 1: "VIN", 2: "SW", 3: "GND", 4: "FB", 5: "EN", 6: "GND_TAB" },
  },
  D_5V: {
    mpn: "SS310-SMC", lcsc: "C2848691", footprint: "SMC_L6.9-W5.9-LS7.9-RD",
    symbol: "schottky", label: "SS310-SMC",
    pins: { 1: "XL_SW", 2: "GND_SYS" },
  },
  L_5V: {
    mpn: "SWPA8065S101MT", lcsc: "C334254", footprint: "IND-SMD_L8.0-W8.0_SWPA8040S",
    symbol: "inductor", label: "100uH",
    pins: { 1: "XL_SW", 2: "+5V" },
  },
  C_5V_OUT: {
    mpn: "UHE1C331MPD", lcsc: "C445436", footprint: "CAP-TH_BD8.0-P3.50-D0.6-FD",
    symbol: "capacitor", label: "330uF",
    polarized: true,
    pins: { 1: "+5V", 2: "GND_SYS" },
  },
  C_5V_OUT_HF: {
    mpn: "0603B104K500NT", lcsc: "C30926", footprint: "C0603",
    symbol: "capacitor", label: "100nF",
    pins: { 1: "+5V", 2: "GND_SYS" },
  },
  R_FB_TOP: {
    mpn: "0603WAF1002T5E", lcsc: "C25804", footprint: "R0603",
    symbol: "resistor", label: "10k",
    pins: { 1: "+5V", 2: "XL_FB" },
  },
  R_FB_BOTTOM: {
    mpn: "RMC06033.3K1%N", lcsc: "C269710", footprint: "R0603",
    symbol: "resistor", label: "3.3k",
    pins: { 1: "XL_FB", 2: "GND_SYS" },
  },
  C_FB_FF: {
    mpn: "GCJ188R71H333KA12D", lcsc: "C2173059", footprint: "C0603",
    symbol: "capacitor", label: "33nF",
    pins: { 1: "+5V", 2: "XL_FB" },
  },

  // DiscreteHBridge
  C_DRV_VCC: {
    mpn: "0603B104K500NT", lcsc: "C30926", footprint: "C0603",
    symbol: "capacitor", label: "100nF",
    pins: { 1: "+5V", 2: "GND_SYS" },
  },
  U_PWM: {
    mpn: "NSG1416", lcsc: "C53436462", footprint: "SOT-23-5",
    symbol: "chip", label: "NSG1416",
    pins: { 1: null, 2: "+5V", 3: "PUMP_PWM", 4: "GND_SYS", 5: "PWM_GATE_DRV" },
    roles: { 1: "NC", 2: "VDD", 3: "IN", 4: "GND", 5: "OUT" },
  },
  R_PWM_GATE_SER: {
    mpn: "RS-06K10R0FT", lcsc: "C140052", footprint: "R1206",
    symbol: "resistor", label: "10",
    pins: { 1: "PWM_GATE_DRV", 2: "PWM_GATE" },
  },
  Q_PWM: {
    mpn: "NCE0117K", lcsc: "C108638", footprint: "TO-252-2L",
    symbol: "nmos", label: "NCE0117K",
    pins: { 1: "PWM_GATE", 2: "MOTOR_B", 3: "PUMP_SHUNT" },
  },
  R_SLEEP: {
    mpn: "0603WAF1002T5E", lcsc: "C25804", footprint: "R0603",
    symbol: "resistor", label: "10k",
    pins: { 1: "PWM_GATE", 2: "PUMP_SHUNT" },
  },
  R_IN2_PD: {
    mpn: "0603WAF1002T5E", lcsc: "C25804", footprint: "R0603",
    symbol: "resistor", label: "10k",
    pins: { 1: "PUMP_PWM", 2: "GND_SYS" },
  },
  D_PUMP_FW: {
    mpn: "SS510C", lcsc: "C19229", footprint: "SMC (DO-214AB)",
    symbol: "schottky", label: "SS510C",
    pins: { 1: "BUS_M", 2: "MOTOR_B" },
  },
  J_MOTOR: {
    mpn: "2EDGVC-5.08-02P-14-100A(H)", lcsc: "C581328", footprint: "CONN-TH_P5.08_2EDGVC-5.08-02P-14-100A-H",
    symbol: "chip", label: "2EDGVC-5.08-02P-14-100A(H)",
    pins: { 1: "BUS_M", 2: "MOTOR_B" },
  },
  R_IN1_PD: {
    mpn: "0603WAF1002T5E", lcsc: "C25804", footprint: "R0603",
    symbol: "resistor", label: "10k",
    pins: { 1: "PUMP_BRAKE", 2: "GND_SYS" },
  },
  R_BRAKE_BASE: {
    mpn: "0603WAF1002T5E", lcsc: "C25804", footprint: "R0603",
    symbol: "resistor", label: "10k",
    pins: { 1: "PUMP_BRAKE", 2: "BRAKE_BASE" },
  },
  Q_BRAKE_CTL: {
    mpn: "MMBT5551", lcsc: "C408396", footprint: "SOT-23",
    symbol: "npn", label: "MMBT5551",
    pins: { 1: "BRAKE_BASE", 2: "GND_SYS", 3: "BRAKE_SINK" },
  },
  Q_BRAKE: {
    mpn: "E100P045CL1", lcsc: "C29781153", footprint: "TO-252",
    symbol: "pmos", label: "E100P045CL1",
    pins: { 1: "BRAKE_GATE", 2: "MOTOR_B", 3: "BUS_M" },
  },
  R_BRAKE_GATE_LOW: {
    mpn: "RS-06K2702FT", lcsc: "C294775", footprint: "R1206",
    symbol: "resistor", label: "27k",
    pins: { 1: "BRAKE_GATE", 2: "BRAKE_SINK" },
  },
  R_BRAKE_GATE_HIGH: {
    mpn: "0603WAF1002T5E", lcsc: "C25804", footprint: "R0603",
    symbol: "resistor", label: "10k",
    pins: { 1: "BUS_M", 2: "BRAKE_GATE" },
  },
  C_DRV_VM1: {
    mpn: "C2012X7R2E103K125AA", lcsc: "C2167598", footprint: "C0805",
    symbol: "capacitor", label: "10nF",
    pins: { 1: "BUS_M", 2: "GND_SYS" },
  },
  R_PUMP_SHUNT: {
    mpn: "HoJLR2512-2W-20mR-1%-75ppm", lcsc: "C2924538", footprint: "R2512",
    symbol: "resistor", label: "0.020",
    note: "20mΩ · 2W",
    pins: { 1: "PUMP_SHUNT", 2: "GND_SYS" },
  },
  R_ADC_I: {
    mpn: "0603WAF1001T5E", lcsc: "C21190", footprint: "R0603",
    symbol: "resistor", label: "1k",
    pins: { 1: "PUMP_SHUNT", 2: "PUMP_I_SENSE" },
  },
  C_ADC_I: {
    mpn: "0603B102K500NT", lcsc: "C44369", footprint: "C0603",
    symbol: "capacitor", label: "1nF",
    pins: { 1: "PUMP_I_SENSE", 2: "GND_SYS" },
  },

  // McuCoreInterface
  J_MCU_A: {
    mpn: "JXT-PM2.54-2X12P-A9", lcsc: "C54803369", footprint: "HDR-TH_24P-P2.54-V-F-R2-C12-S2.54-2",
    symbol: "chip", label: "JXT-PM2.54-2X12P-A9",
    pins: {
      1: "+5V", 2: "+5V", 3: "GND_SYS", 4: "GND_SYS", 5: null, 6: null,
      7: null, 8: null, 9: null, 10: "PUMP_PWM", 11: null, 12: null,
      13: null, 14: null, 15: null, 16: null, 17: null, 18: null,
      19: null, 20: "PUMP_BRAKE", 21: null, 22: null, 23: "CAN_RX", 24: "CAN_TX",
    },
    roles: { 10: "PA8", 20: "PB5" },
  },
  J_MCU_B: {
    mpn: "JXT-PM2.54-2X12P-A9", lcsc: "C54803369", footprint: "HDR-TH_24P-P2.54-V-F-R2-C12-S2.54-2",
    symbol: "chip", label: "JXT-PM2.54-2X12P-A9",
    pins: {
      1: "+3V3", 2: "+3V3", 3: "GND_SYS", 4: "GND_SYS", 5: null, 6: null,
      7: null, 8: null, 9: null, 10: null, 11: "PUMP_I_SENSE", 12: null,
      13: null, 14: null, 15: null, 16: null, 17: "ADC_BUS", 18: "ADC_NTC",
      19: null, 20: null, 21: null, 22: null, 23: null, 24: null,
    },
    roles: { 11: "PA7", 17: "PA1", 18: "PA2" },
  },
  R_BUS_TOP1: {
    mpn: "TC0350B1103T5F", lcsc: "C50670544", footprint: "R0603",
    symbol: "resistor", label: "110k",
    pins: { 1: "BUS_M", 2: "BUS_DIV_MID" },
  },
  R_BUS_TOP2: {
    mpn: "TC0350B1103T5F", lcsc: "C50670544", footprint: "R0603",
    symbol: "resistor", label: "110k",
    pins: { 1: "BUS_DIV_MID", 2: "BUS_SENSE" },
  },
  R_BUS_BOTTOM: {
    mpn: "HoAR0603-1/10W-10KR-0.1%-TCR25", lcsc: "C5123581", footprint: "R0603",
    symbol: "resistor", label: "10k",
    pins: { 1: "BUS_SENSE", 2: "GND_SYS" },
  },
  R_ADC_BUS: {
    mpn: "0603WAF1001T5E", lcsc: "C21190", footprint: "R0603",
    symbol: "resistor", label: "1k",
    pins: { 1: "BUS_SENSE", 2: "ADC_BUS" },
  },
  C_ADC_BUS: {
    mpn: "0603B104K500NT", lcsc: "C30926", footprint: "C0603",
    symbol: "capacitor", label: "100nF",
    pins: { 1: "ADC_BUS", 2: "GND_SYS" },
  },
  R_NTC_PULL: {
    mpn: "HoAR0603-1/10W-10KR-0.1%-TCR25", lcsc: "C5123581", footprint: "R0603",
    symbol: "resistor", label: "10k",
    pins: { 1: "+3V3", 2: "ADC_NTC" },
  },
  NTC_BOARD: {
    mpn: "MF52A103F3950(A1)", lcsc: "C13879", footprint: "RES-TH_L2.5-W2.5-P1.70-D0.3",
    symbol: "resistor", label: "10k",
    note: "10kΩ · B3950",
    pins: { 1: "ADC_NTC", 2: "GND_SYS" },
  },
  C_ADC_NTC: {
    mpn: "C1608X7R1H103K080AA", lcsc: "C2180830", footprint: "C0603",
    symbol: "capacitor", label: "10nF",
    pins: { 1: "ADC_NTC", 2: "GND_SYS" },
  },

  // IsolatedCan
  R_CAN_TXD_PU: {
    mpn: "0603WAF1002T5E", lcsc: "C25804", footprint: "R0603",
    symbol: "resistor", label: "10k",
    pins: { 1: "+3V3", 2: "CAN_TX" },
  },
  U_CAN: {
    mpn: "CA-IS2062A", lcsc: "C22438481", footprint: "LGA-16_L5.2-W4.7-P0.65-TL_CA-IS2062A-FRANK2",
    symbol: "chip", label: "CA-IS2062A",
    pins: {
      1: "CAN_RX", 2: null, 3: null, 4: "CAN_TX", 5: "GND_SYS", 6: "+5V",
      7: "+3V3", 8: "GND_SYS", 9: "GND_CAN", 10: "CAN_ISO_5V", 11: "CAN_ISO_5V", 12: "GND_CAN",
      13: null, 14: "CANL", 15: "CANH", 16: null,
    },
    roles: { 1: "RXD", 2: "NC", 3: "NC", 4: "TXD", 5: "GNDP1", 6: "VDDP", 7: "VDDL", 8: "GND1", 9: "GND2", 10: "VISOIN", 11: "VISOOUT", 12: "GNDP2", 13: "NC", 14: "CANL", 15: "CANH", 16: "NC" },
  },
  D_CAN: {
    mpn: "PESD2CANFD24V-TR", lcsc: "C552486", footprint: "SOT-23-3_L2.9-W1.3-P1.90-LS2.4-BR",
    symbol: "chip", label: "PESD2CANFD24V-TR",
    pins: { 1: "CANH", 2: "CANL", 3: "GND_CAN" },
    roles: { 1: "CANH", 2: "CANL", 3: "GND" },
  },
  J_CAN: {
    mpn: "B3B-XH-A(LF)(SN)", lcsc: "C144394", footprint: "CONN-TH_3P-P2.50_A2501WV2-3P",
    symbol: "chip", label: "B3B-XH-A(LF)(SN)",
    pins: { 1: "GND_CAN", 2: "CANH", 3: "CANL" },
  },
  R_TERM: {
    mpn: "MFR25SFRE52-120R", lcsc: "C1744031", footprint: "RES-TH_BD1.9-L3.5-P7.50-D0.5",
    symbol: "resistor", label: "120",
    note: "120Ω · DNP",
    dnp: true,
    pins: { 1: "CANH", 2: "CANL" },
  },
  C_CAN_VCC: {
    mpn: "0603B104K500NT", lcsc: "C30926", footprint: "C0603",
    symbol: "capacitor", label: "100nF",
    pins: { 1: "+5V", 2: "GND_SYS" },
  },
  C_CAN_VDDP_BULK: {
    mpn: "1210X476K160NT", lcsc: "C2961286", footprint: "C1210",
    symbol: "capacitor", label: "47uF",
    pins: { 1: "+5V", 2: "GND_SYS" },
  },
  C_CAN_VIO: {
    mpn: "0603B105K250NT", lcsc: "C59302", footprint: "C0603",
    symbol: "capacitor", label: "1uF",
    pins: { 1: "+3V3", 2: "GND_SYS" },
  },
  C_CAN_ISO_BULK: {
    mpn: "1210X476K160NT", lcsc: "C2961286", footprint: "C1210",
    symbol: "capacitor", label: "47uF",
    pins: { 1: "CAN_ISO_5V", 2: "GND_CAN" },
  },
  C_CAN_ISO_HF: {
    mpn: "0603B104K500NT", lcsc: "C30926", footprint: "C0603",
    symbol: "capacitor", label: "100nF",
    pins: { 1: "CAN_ISO_5V", 2: "GND_CAN" },
  },
  C_CAN_ISOIN_1U: {
    mpn: "0603B105K250NT", lcsc: "C59302", footprint: "C0603",
    symbol: "capacitor", label: "1uF",
    pins: { 1: "CAN_ISO_5V", 2: "GND_CAN" },
  },

} satisfies Record<string, PartDefinition>

export type PumpReference = keyof typeof parts
