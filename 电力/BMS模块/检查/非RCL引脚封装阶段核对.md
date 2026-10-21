# BMS 非 R/C/L 器件引脚/焊盘功能核对阶段结论

- 当前设计输入 `sourceInputSha256`：`5A3DEEB6379C99A47EA0ACCC786745AD96F73BFBEA41587EEC8C4CB0B61DDDE9`
- 当前施工清单 SHA256：`EE976B09A02EF2B8FDEF6ABC708E37385302E01B025579DFB458B2B13069F46F`
- 当前官方 elibz SHA256：`0FBDE058A3DE520EF5C05C50F07087E8F2F320FC62077F1D264A8BA05DF70F8F`
- 连接 JSON SHA256：
  - `AFE元件连接.json`：`6F82B628F826FB19638FF3638B02A797B3C59F660294D89C9EB6B6A7EDF5B073`
  - `功率元件连接.json`：`FA5834A66E24BBCCA56CECDC708FC142EEB5CB38DAE904E01AF461E6134BBF36`
  - `接口元件连接.json`：`1C111014F99224BD4FE3A7A2576351F5F0ADF5FBE61E1899231B87494646F46C`
  - `风扇元件连接.json`：`CB6880C5051884F3FD2B48EDFC466697EF5041693C002B45CFA5432AF5C7ED92`

## 汇总

本轮从当前连接 JSON 重读并与施工清单逐项对齐：36 个位号、19 种 MPN。结果：

- 结构一致：实体方向待核：5。
- PASS：引脚功能与端号对应：11。
- 部分依据：厂家未标数字脚序、库数字映射与正式源一致：1。
- REVIEW：封装极性方向待核：2。
- PASS：仅编号对应：2。
- PASS：EP焊盘与BATN公共地对应：8。
- PASS：端号对应；焊盘形状待人工审阅：6。
- 缺库源：无法核端号/焊盘：1。
- 当前 MPN 和 pin map 与施工清单 layout 重读一致：36/36。
- 当前 elibz 可读 Device 源：35/36；D_CAN_TVS 虽有施工清单 binding UUID，但没有可读 Device/Symbol/Footprint 原始源。

## 每个位号结论

| 位号 | 当前 MPN | 当前正式 pin→net | 阶段 | 当前 device / symbol / footprint UUID | 结论 |
|---|---|---|---|---|---|
| J_CORE_P1 | 254PM-2x12P-V | `8=NC, 6=NC, 17=NC, 3=B0, 13=NC, 7=NC, 16=NC, 5=NC, 9=NC, 15=FAN_RUN, 23=CAN_RX, 12=NC, 14=NC, 10=NC, 4=B0, 22=MAIN_REQ, 20=BQ_ALERT_N, 19=BMS_LOCAL_OK, 2=VOUT_5V, 24=CAN_TX, 21=PRE_REQ, 1=VOUT_5V, 11=NC, 18=NC` | 结构一致：实体方向待核 | `267440d4c22b4aa5bc805ec75934cfc2 / 68b04dc0184145e9afcf619c599cb9d5 / 0b8d789b21f44c4e803ba1ca8c2753f1` | 2×12 符号与封装 1–24 编号集合齐全，正式源 pin 1–24 与库编号对应；本轮未独立核验排针/排母的实体视角与配对线序。 |
| J_CORE_P2 | 254PM-2x12P-V | `8=NC, 6=BQ_I2C_SDA, 17=NC, 3=B0, 13=BAL_ENABLE_CMD, 7=NC, 16=NC, 5=BQ_I2C_SCL, 9=NC, 15=NC, 23=NC, 12=NC, 14=NC, 10=NC, 4=B0, 22=NC, 20=NC, 19=NC, 2=CORE_3V3, 24=NC, 21=NC, 1=CORE_3V3, 11=NC, 18=NC` | 结构一致：实体方向待核 | `267440d4c22b4aa5bc805ec75934cfc2 / 68b04dc0184145e9afcf619c599cb9d5 / 0b8d789b21f44c4e803ba1ca8c2753f1` | 2×12 符号与封装 1–24 编号集合齐全，正式源 pin 1–24 与库编号对应；本轮未独立核验排针/排母的实体视角与配对线序。 |
| J_FAN | 2EDGVC-5.08-02P-14-100A(H) | `1=FAN_5V, 2=B0` | 结构一致：实体方向待核 | `e2136f2a86054070a01a98a1b42be9ea / c040430ea03345fdb7a9a35f8ac3adeb / cd5a23d6442042e79479e5ec711ddd1c` | 符号端号、库焊盘端号与正式源端号集合一致（1–2）；未独立核验端子实体视角、朝向和接线侧编号。 |
| J_LOAD_POWER_VALID | 2EDGVC-5.08-02P-14-100A(H) | `1=LOAD_EN_BIAS_EXT, 2=LOAD_POWER_VALID` | 结构一致：实体方向待核 | `e2136f2a86054070a01a98a1b42be9ea / c040430ea03345fdb7a9a35f8ac3adeb / cd5a23d6442042e79479e5ec711ddd1c` | 符号端号、库焊盘端号与正式源端号集合一致（1–2）；未独立核验端子实体视角、朝向和接线侧编号。 |
| J_CAN | 2EDGVC-5.08-03P-14-100A(H) | `1=CANH, 2=CANL, 3=PACK-` | 结构一致：实体方向待核 | `0b02de662cf742e48250e9a484a03710 / bb8a71d7584f4b27bd6e91be07c5e32f / a915f53f5815402699030ef425256137` | 连接器符号端号、库焊盘端号与正式源端号集合一致（1–3）；本轮未以厂家机械图独立确认器件视角、端子朝向及线缆侧编号。 |
| U_BAL_OPTO_Q1 | AT3H74-CuH-S | `1=BAL_OPTO_LED_0, 2=BAL_OPTO_LED_1, 16=BAL_EN_CLAMP_GATE_1, 15=B0, 3=BAL_OPTO_LED_1, 4=BAL_OPTO_LED_2, 14=BAL_EN_CLAMP_GATE_2, 13=TAP_B1, 5=BAL_OPTO_LED_2, 6=BAL_OPTO_LED_3, 12=BAL_EN_CLAMP_GATE_3, 11=TAP_B2, 7=BAL_OPTO_LED_3, 8=BAL_OPTO_LED_4, 10=BAL_EN_CLAMP_GATE_4, 9=TAP_B3` | PASS：引脚功能与端号对应 | `5c65181be86d494c942ee8f6bcdc1462 / e533973ce9fb44598f9d00c0d10325ce / 73f4598ae01341a999a1eda3d7f94fc0` | 16 个符号 pin 与封装 pad 编号 1–16 一一对应；当前正式源输入 LED 对和每路 C/E 端号映射与厂家脚图相符。 |
| U_BAL_OPTO_Q2 | AT3H74-CuH-S | `1=BAL_OPTO_LED_4, 2=BAL_OPTO_LED_5, 16=BAL_EN_CLAMP_GATE_5, 15=TAP_B4, 3=BAL_OPTO_LED_5, 4=BAL_OPTO_LED_6, 14=BAL_EN_CLAMP_GATE_6, 13=TAP_B5, 5=BAL_OPTO_LED_6, 6=BAL_OPTO_LED_7, 12=BAL_EN_CLAMP_GATE_7, 11=TAP_B6, 7=BAL_OPTO_LED_7, 8=BAL_OPTO_LED_8, 10=BAL_EN_CLAMP_GATE_8, 9=TAP_B7` | PASS：引脚功能与端号对应 | `5c65181be86d494c942ee8f6bcdc1462 / e533973ce9fb44598f9d00c0d10325ce / 73f4598ae01341a999a1eda3d7f94fc0` | 16 个符号 pin 与封装 pad 编号 1–16 一一对应；当前正式源输入 LED 对和每路 C/E 端号映射与厂家脚图相符。 |
| D_BAT | BAT46W-7-F | `1=BAT_DIODE, 2=CELL_B9` | REVIEW：封装极性方向待核 | `a4a5071e96464e3cb64a8e44e5c0c90e / 1f07de743ab94c58be31411b983e2b2c / eabc9173c4664f8a90fe92f757e4228f` | 正式源 pin1=BAT_DIODE、pin2=CELL_B9；封装 pad 1/2 与符号 pin 1/2 齐全，但库符号只写数字，当前原始库无法证明 pad1 与器件阴极色带方向一致。 |
| D_BREG | BAT46W-7-F | `1=BREG_DIODE, 2=CELL_B9` | REVIEW：封装极性方向待核 | `a4a5071e96464e3cb64a8e44e5c0c90e / 1f07de743ab94c58be31411b983e2b2c / eabc9173c4664f8a90fe92f757e4228f` | 正式源 pin1=BREG_DIODE、pin2=CELL_B9；封装 pad 1/2 与符号 pin 1/2 齐全，但库符号只写数字，当前原始库无法证明 pad1 与器件阴极色带方向一致。 |
| U1 | BQ7695202PFBR | `1=VC15_FILT, 2=VC8_FILT, 3=VC8_FILT, 4=VC8_FILT, 5=VC8_FILT, 6=VC8_FILT, 7=VC8_FILT, 8=VC8_FILT, 9=VC7_FILT, 10=VC6_FILT, 11=VC5_FILT, 12=VC4_FILT, 13=VC3_FILT, 14=VC2_FILT, 15=VC1_FILT, 16=VC0_FILT, 17=B0, 18=SRP, 19=NC, 20=SRN, 21=BQ_TS1, 22=NTC_BOARD_NODE, 23=BQ_TS3, 24=BQ_REG18, 25=BQ_ALERT_IC_N, 26=BQ_I2C_SCL, 27=BQ_I2C_SDA, 28=NC, 29=BQ_CFETOFF, 30=DFETOFF, 31=BQ_DCHG, 32=NC, 33=RST_SHUT, 34=NC, 35=REG1, 36=REGIN, 37=BREG, 38=NC, 39=NC, 40=NC, 41=BQ_LD, 42=BQ_PACK, 43=DSG, 44=NC, 45=CHG, 46=CP1, 47=BAT, 48=VC16_FILT` | PASS：仅编号对应 | `11a7ff39fa6a461a89d259faea7aed4a / 9d334130c9d0406ca221f499d6bd292d / bfef61e59b344d0eb8a970b05c25bc25` | 48 个符号端号与封装 pad 端号逐一对应，正式源 pin 1–48 的网络/空引脚键完整落在对应编号。保留意图 NC，不代表 AFE 配置、采样链或保护逻辑已验收。 |
| U_CAN | CA-IS2062A | `1=CAN_RX, 2=NC, 3=NC, 4=CAN_TX, 5=B0, 6=VOUT_5V, 7=CORE_3V3, 8=B0, 9=PACK-, 10=CAN_ISO_5V, 11=CAN_ISO_5V, 12=PACK-, 13=NC, 14=CANL, 15=CANH, 16=NC` | PASS：引脚功能与端号对应 | `f0c935430d56463499dcfd5a80ea6af5 / 0b6c26fbbbf2439fb20cd11158cc3030 / 1bc0f77a3d724950bf13445f1e2cdc53` | 16 个 pin 号与封装焊盘编号对应；符号 pin 名与厂家功能号相符，正式源 CAN_RX/TX、两侧电源/地、CANL/H 映射到相同端号，NC 端保持空接。 |
| U_ETA_BAL1 | ETA3000D2I | `1=TAP_B1, 2=TAP_B2, 3=BAL_ETA_SW_1, 4=B0, 5=BAL_ETA_BIAS_1, 6=NC, 7=BAL_ETA_ISET_1, 8=BAL_ETA_EN_1, 9=B0` | PASS：EP焊盘与BATN公共地对应 | `15d201998529493cb3fc5dec2536f526 / 1122ffc6eef1424fbcda2055c99b4c25 / e9425520e51749b88aba9a4607a4e5af` | 厂家 pin4=BATN 且说明为负端/公共 GND；厂家要求 exposed pad 接 GND。官方库把中心裸露焊盘编号为 CAD pin9；当前 8 颗 pin4 与 pin9 同网，均对… |
| U_ETA_BAL2 | ETA3000D2I | `1=TAP_B2, 2=TAP_B3, 3=BAL_ETA_SW_2, 4=TAP_B1, 5=BAL_ETA_BIAS_2, 6=NC, 7=BAL_ETA_ISET_2, 8=BAL_ETA_EN_2, 9=TAP_B1` | PASS：EP焊盘与BATN公共地对应 | `15d201998529493cb3fc5dec2536f526 / 1122ffc6eef1424fbcda2055c99b4c25 / e9425520e51749b88aba9a4607a4e5af` | 厂家 pin4=BATN 且说明为负端/公共 GND；厂家要求 exposed pad 接 GND。官方库把中心裸露焊盘编号为 CAD pin9；当前 8 颗 pin4 与 pin9 同网，均对… |
| U_ETA_BAL3 | ETA3000D2I | `1=TAP_B3, 2=TAP_B4, 3=BAL_ETA_SW_3, 4=TAP_B2, 5=BAL_ETA_BIAS_3, 6=NC, 7=BAL_ETA_ISET_3, 8=BAL_ETA_EN_3, 9=TAP_B2` | PASS：EP焊盘与BATN公共地对应 | `15d201998529493cb3fc5dec2536f526 / 1122ffc6eef1424fbcda2055c99b4c25 / e9425520e51749b88aba9a4607a4e5af` | 厂家 pin4=BATN 且说明为负端/公共 GND；厂家要求 exposed pad 接 GND。官方库把中心裸露焊盘编号为 CAD pin9；当前 8 颗 pin4 与 pin9 同网，均对… |
| U_ETA_BAL4 | ETA3000D2I | `1=TAP_B4, 2=TAP_B5, 3=BAL_ETA_SW_4, 4=TAP_B3, 5=BAL_ETA_BIAS_4, 6=NC, 7=BAL_ETA_ISET_4, 8=BAL_ETA_EN_4, 9=TAP_B3` | PASS：EP焊盘与BATN公共地对应 | `15d201998529493cb3fc5dec2536f526 / 1122ffc6eef1424fbcda2055c99b4c25 / e9425520e51749b88aba9a4607a4e5af` | 厂家 pin4=BATN 且说明为负端/公共 GND；厂家要求 exposed pad 接 GND。官方库把中心裸露焊盘编号为 CAD pin9；当前 8 颗 pin4 与 pin9 同网，均对… |
| U_ETA_BAL5 | ETA3000D2I | `1=TAP_B5, 2=TAP_B6, 3=BAL_ETA_SW_5, 4=TAP_B4, 5=BAL_ETA_BIAS_5, 6=NC, 7=BAL_ETA_ISET_5, 8=BAL_ETA_EN_5, 9=TAP_B4` | PASS：EP焊盘与BATN公共地对应 | `15d201998529493cb3fc5dec2536f526 / 1122ffc6eef1424fbcda2055c99b4c25 / e9425520e51749b88aba9a4607a4e5af` | 厂家 pin4=BATN 且说明为负端/公共 GND；厂家要求 exposed pad 接 GND。官方库把中心裸露焊盘编号为 CAD pin9；当前 8 颗 pin4 与 pin9 同网，均对… |
| U_ETA_BAL6 | ETA3000D2I | `1=TAP_B6, 2=TAP_B7, 3=BAL_ETA_SW_6, 4=TAP_B5, 5=BAL_ETA_BIAS_6, 6=NC, 7=BAL_ETA_ISET_6, 8=BAL_ETA_EN_6, 9=TAP_B5` | PASS：EP焊盘与BATN公共地对应 | `15d201998529493cb3fc5dec2536f526 / 1122ffc6eef1424fbcda2055c99b4c25 / e9425520e51749b88aba9a4607a4e5af` | 厂家 pin4=BATN 且说明为负端/公共 GND；厂家要求 exposed pad 接 GND。官方库把中心裸露焊盘编号为 CAD pin9；当前 8 颗 pin4 与 pin9 同网，均对… |
| U_ETA_BAL7 | ETA3000D2I | `1=TAP_B7, 2=TAP_B8, 3=BAL_ETA_SW_7, 4=TAP_B6, 5=BAL_ETA_BIAS_7, 6=NC, 7=BAL_ETA_ISET_7, 8=BAL_ETA_EN_7, 9=TAP_B6` | PASS：EP焊盘与BATN公共地对应 | `15d201998529493cb3fc5dec2536f526 / 1122ffc6eef1424fbcda2055c99b4c25 / e9425520e51749b88aba9a4607a4e5af` | 厂家 pin4=BATN 且说明为负端/公共 GND；厂家要求 exposed pad 接 GND。官方库把中心裸露焊盘编号为 CAD pin9；当前 8 颗 pin4 与 pin9 同网，均对… |
| U_ETA_BAL8 | ETA3000D2I | `1=TAP_B8, 2=TAP_B9, 3=BAL_ETA_SW_8, 4=TAP_B7, 5=BAL_ETA_BIAS_8, 6=NC, 7=BAL_ETA_ISET_8, 8=BAL_ETA_EN_8, 9=TAP_B7` | PASS：EP焊盘与BATN公共地对应 | `15d201998529493cb3fc5dec2536f526 / 1122ffc6eef1424fbcda2055c99b4c25 / e9425520e51749b88aba9a4607a4e5af` | 厂家 pin4=BATN 且说明为负端/公共 GND；厂家要求 exposed pad 接 GND。官方库把中心裸露焊盘编号为 CAD pin9；当前 8 颗 pin4 与 pin9 同网，均对… |
| Q_BREG | FCX495TA | `1=BREG, 2=BREG_COLLECTOR, 3=REGIN` | 部分依据：厂家图未标数字脚序 | `ccb55617316e45928d4c73767f613adf / 61246eeac9fc4b8180bdbe1d98f254fb / 56b98ce1f5664118ae6dbbfaa5718ffb` | 原厂 SOT-89 顶视图只标 B/C/E 与位置、C/tab，未标数字端号。当前库 symbol 1=B、2=C、3=E，footprint pads 1/2/3 齐全；正式源 1=BREG、2=BREG_COLLECTOR、3=REGIN 与库数字映射一致，但厂家数字脚序未被直接证明。 |
| Q_CHG1 | HYG022N10NS1TA | `1=CHG_G1, 2=BATT+_FUSED, 3=BATT+_FUSED, 4=BATT+_FUSED, 5=BATT+_FUSED, 6=BATT+_FUSED, 7=BATT+_FUSED, 8=BATT+_FUSED, 9=FET_MID` | PASS：端号对应；焊盘形状待人工审阅 | `72f44b11027f4f00ad6210e6490dc8e7 / 992811325d4240078c3d0ec9919b7c97 / ae68bf96a0ea4cd286f165c1e2f44505` | TOLL：pin1=G、pins2–8=S、散热 tab=D；正式源 CAD pin9 映射 D。e30 是 pin8 纵向焊盘，e31 是同编号 pin8 的横向延伸，不是新增引脚；e31 不… |
| Q_CHG2 | HYG022N10NS1TA | `1=CHG_G2, 2=BATT+_FUSED, 3=BATT+_FUSED, 4=BATT+_FUSED, 5=BATT+_FUSED, 6=BATT+_FUSED, 7=BATT+_FUSED, 8=BATT+_FUSED, 9=FET_MID` | PASS：端号对应；焊盘形状待人工审阅 | `72f44b11027f4f00ad6210e6490dc8e7 / 992811325d4240078c3d0ec9919b7c97 / ae68bf96a0ea4cd286f165c1e2f44505` | TOLL：pin1=G、pins2–8=S、散热 tab=D；正式源 CAD pin9 映射 D。e30 是 pin8 纵向焊盘，e31 是同编号 pin8 的横向延伸，不是新增引脚；e31 不… |
| Q_CHG3 | HYG022N10NS1TA | `1=CHG_G3, 2=BATT+_FUSED, 3=BATT+_FUSED, 4=BATT+_FUSED, 5=BATT+_FUSED, 6=BATT+_FUSED, 7=BATT+_FUSED, 8=BATT+_FUSED, 9=FET_MID` | PASS：端号对应；焊盘形状待人工审阅 | `72f44b11027f4f00ad6210e6490dc8e7 / 992811325d4240078c3d0ec9919b7c97 / ae68bf96a0ea4cd286f165c1e2f44505` | TOLL：pin1=G、pins2–8=S、散热 tab=D；正式源 CAD pin9 映射 D。e30 是 pin8 纵向焊盘，e31 是同编号 pin8 的横向延伸，不是新增引脚；e31 不… |
| Q_DSG1 | HYG022N10NS1TA | `1=DSG_G1, 2=PACK+, 3=PACK+, 4=PACK+, 5=PACK+, 6=PACK+, 7=PACK+, 8=PACK+, 9=FET_MID` | PASS：端号对应；焊盘形状待人工审阅 | `72f44b11027f4f00ad6210e6490dc8e7 / 992811325d4240078c3d0ec9919b7c97 / ae68bf96a0ea4cd286f165c1e2f44505` | TOLL：pin1=G、pins2–8=S、散热 tab=D；正式源 CAD pin9 映射 D。e30 是 pin8 纵向焊盘，e31 是同编号 pin8 的横向延伸，不是新增引脚；e31 不… |
| Q_DSG2 | HYG022N10NS1TA | `1=DSG_G2, 2=PACK+, 3=PACK+, 4=PACK+, 5=PACK+, 6=PACK+, 7=PACK+, 8=PACK+, 9=FET_MID` | PASS：端号对应；焊盘形状待人工审阅 | `72f44b11027f4f00ad6210e6490dc8e7 / 992811325d4240078c3d0ec9919b7c97 / ae68bf96a0ea4cd286f165c1e2f44505` | TOLL：pin1=G、pins2–8=S、散热 tab=D；正式源 CAD pin9 映射 D。e30 是 pin8 纵向焊盘，e31 是同编号 pin8 的横向延伸，不是新增引脚；e31 不… |
| Q_DSG3 | HYG022N10NS1TA | `1=DSG_G3, 2=PACK+, 3=PACK+, 4=PACK+, 5=PACK+, 6=PACK+, 7=PACK+, 8=PACK+, 9=FET_MID` | PASS：端号对应；焊盘形状待人工审阅 | `72f44b11027f4f00ad6210e6490dc8e7 / 992811325d4240078c3d0ec9919b7c97 / ae68bf96a0ea4cd286f165c1e2f44505` | TOLL：pin1=G、pins2–8=S、散热 tab=D；正式源 CAD pin9 映射 D。e30 是 pin8 纵向焊盘，e31 是同编号 pin8 的横向延伸，不是新增引脚；e31 不… |
| Q_PRE1 | NCE01P30K | `1=QPRE_G, 2=BATT+_FUSED, 3=QPRE_SRC` | PASS：引脚功能与端号对应 | `99ef698ad25848ee9d345c73b8ce27d5 / 850815f6719b4df8915dd18d8390abe2 / aff334dc45be4fb1854d8713e5ed3dd5` | P 沟道 MOS SOT-23：1=G、2=D、3=S；符号 pin 名、封装焊盘编号和正式源网络一致。 |
| Q_PRE2 | NCE01P30K | `1=QPRE_G, 2=QPRE_SW, 3=QPRE_SRC` | PASS：引脚功能与端号对应 | `99ef698ad25848ee9d345c73b8ce27d5 / 850815f6719b4df8915dd18d8390abe2 / aff334dc45be4fb1854d8713e5ed3dd5` | P 沟道 MOS SOT-23：1=G、2=D、3=S；符号 pin 名、封装焊盘编号和正式源网络一致。 |
| D_CAN_TVS | PESD2CANFD24V-TR | `1=CANH, 2=CANL, 3=PACK-` | 缺库源：无法核端号/焊盘 | `76bcd52e6dd845eb950d4f76382feb0e / 75d11002b26f44fb9e5fd84309d4a68c / bdad16194d454cb292a50525f39c3ffd` | 施工清单当前有绑定 UUID，正式源 1=CANH、2=CANL、3=PACK-；但当前 elibz 中没有可读 Device/Symbol/Footprint 源，无法核对数字到焊盘。 |
| U_PSU | SCT2A26STER | `1=B0, 2=CELL_B9, 3=PSU_EN, 4=NC, 5=PSU_FB, 6=B0, 7=PSU_BST, 8=PSU_SW, 9=B0` | PASS：引脚功能与端号对应 | `0b9a375741ff467ab6dc4066c4fab27e / 0886204536bc492596741baf75dae57b / 3fc9ebb0130f47f2a842308f2090c3ef` | 厂家 pin1 GND、2 VIN、3 EN、4 NC、5 FB、6 TM、7 BST、8 SW、9 EP；符号/封装号和正式源网络一致，4 保持未连接，9 为 B0。 |
| U_GATE1 | TI SN74LVC1G11DCKR | `1=SW_RUN, 2=B0, 3=BMS_LOCAL_OK, 4=PRECHARGE_WINDOW, 5=PRE_CTRL_3V3, 6=TMR_ACTIVE` | PASS：引脚功能与端号对应 | `27376aeb8ef24d0493ee69f282c2787f / 680e6e65287c42dd9e2d21b084cd4d03 / 40d544f497914caaa6cbea3bc4eaae91` | DCK 6-pin：1=A、2=GND、3=B、4=Y、5=VCC、6=C；符号 pin 名、pad 编号和当前正式源对应。只核对 pin mapping，不验证 AND 逻辑时序/逻辑设计。 |
| U_SEQ | TI SN74LVC74ADR | `1=MAIN_CLR_N, 2=PRE_CTRL_3V3, 3=MAIN_REQ, 4=PRE_CTRL_3V3, 5=FET_RELEASE, 6=MAIN_HOLD_N, 7=B0, 8=PRE_USED_N, 9=TMR_TRIG, 10=PRE_CTRL_3V3, 11=PRE_REQ, 12=PRE_CTRL_3V3, 13=POR_OK_N, 14=PRE_CTRL_3V3` | PASS：仅编号对应 | `4b9b0b3026db4c3781dcfe76716d53b7 / 83505c9b0810430b94597dea14c21f9f / f1b3106e9d564687958e1016421ed2c5` | 14 个 pin 的双 D 触发器引脚功能号与符号/封装编号相符；当前正式源 pin 1–14 网络逐项使用对应编号。仅编号映射，不验证时序状态机。 |
| U_TIMER | TI TPL5111DDCR | `1=TMR_VDD, 2=B0, 3=TMR_SET, 4=B0, 5=TMR_ACTIVE, 6=B0` | PASS：引脚功能与端号对应 | `c3cf736bc2524174963d78e17a18aa92 / 08374e1fddbe4538b0a98b1e166cc2d2 / 99a529f2657a4db3b518ca8a2cf34b91` | 6-pin pin1=VDD、2=GND、3=DELAY/M_DRV、4=DONE、5=DRVn、6=EN/ONE_SHOT；符号/封装编号与正式源一致。 |
| U_POR | TI TPS3809K33DBVR | `1=B0, 2=POR_OK_N, 3=PRE_CTRL_3V3` | PASS：引脚功能与端号对应 | `730f62007a0a4f4aae50fb25c8097e82 / 8dd2f84cc6fe4e70a895b617093d2881 / 078e43bc93394f49815211c4fd533d19` | SOT-23：1=GND、2=RESET#、3=VDD；正式网络 B0/POR_OK_N/PRE_CTRL_3V3 与符号 pin 名、封装号一致。 |
| Q_QPRE_DRV | onsemi MMBT5551LT1G | `1=QPRE_DRV_B, 2=B0, 3=QPRE_PULL_C` | PASS：引脚功能与端号对应 | `e9aa8bc80a8e495c9e7817b3328be7bc / 907e3848a9fe40e89b2a1beddc8a6491 / 410bf5f266b44ea2820d622d3d20afa8` | SOT-23：1=B、2=E、3=C；符号引脚名与封装 pad 编号一致，当前正式源 1=QPRE_DRV_B、2=B0、3=QPRE_PULL_C。 |
| DZ_QPRE | onsemi MMSZ5242BT1G | `1=QPRE_SRC, 2=QPRE_G` | PASS：引脚功能与端号对应 | `c0336bfa04c5418f8f5ef4d72bbecd49 / 78897fd77ff04664aef989c24570348e / eabc9173c4664f8a90fe92f757e4228f` | SOD-123 符号 pin1=K、pin2=A；封装焊盘有 1/2，正式源 1=QPRE_SRC、2=QPRE_G。与器件极性和端号一致。 |

## 型号级证据与说明

### 254PM-2x12P-V

- 位号：J_CORE_P1, J_CORE_P2
- 结论：结构一致：实体方向待核。2×12 符号与封装 1–24 编号集合齐全，正式源 pin 1–24 与库编号对应；本轮未独立核验排针/排母的实体视角与配对线序。
- 器件端号/功能：本轮只确认 CAD 编号结构，未确认连接器实体方向。
- 当前清单 libUuid：`267440d4c22b4aa5bc805ec75934cfc2`；Symbol UUID：`68b04dc0184145e9afcf619c599cb9d5`；Footprint UUID：`0b8d789b21f44c4e803ba1ca8c2753f1`
- 证据：https://item.szlcsc.com/datasheet/254PM-2x12P-V/56231317.html
- 附注：J_CORE_P1、J_CORE_P2 的网络编号映射已逐 pin 从当前 JSON 读取；实体线序需按具体连接器装配视角确认。

### 2EDGVC-5.08-02P-14-100A(H)

- 位号：J_FAN, J_LOAD_POWER_VALID
- 结论：结构一致：实体方向待核。符号端号、库焊盘端号与正式源端号集合一致（1–2）；未独立核验端子实体视角、朝向和接线侧编号。
- 器件端号/功能：本轮只确认 CAD 编号结构，未确认实体方向。
- 当前清单 libUuid：`e2136f2a86054070a01a98a1b42be9ea`；Symbol UUID：`c040430ea03345fdb7a9a35f8ac3adeb`；Footprint UUID：`cd5a23d6442042e79479e5ec711ddd1c`
- 证据：https://item.szlcsc.com/datasheet/2EDGVC-5.08-02P-14-100A(H)/609129.html
- 附注：J_FAN、J_LOAD_POWER_VALID 保持结构级结论。

### 2EDGVC-5.08-03P-14-100A(H)

- 位号：J_CAN
- 结论：结构一致：实体方向待核。连接器符号端号、库焊盘端号与正式源端号集合一致（1–3）；本轮未以厂家机械图独立确认器件视角、端子朝向及线缆侧编号。
- 器件端号/功能：本轮只确认 CAD 编号结构，未确认实体方向。
- 当前清单 libUuid：`0b02de662cf742e48250e9a484a03710`；Symbol UUID：`bb8a71d7584f4b27bd6e91be07c5e32f`；Footprint UUID：`a915f53f5815402699030ef425256137`
- 证据：https://item.szlcsc.com/datasheet/2EDGVC-5.08-03P-14-100A(H)/609130.html
- 附注：J_CAN 结构可核对；连线物理朝向/接线顺序仍需结合端子厂家图或首件。

### AT3H74-CuH-S

- 位号：U_BAL_OPTO_Q1, U_BAL_OPTO_Q2
- 结论：PASS：引脚功能与端号对应。16 个符号 pin 与封装 pad 编号 1–16 一一对应；当前正式源输入 LED 对和每路 C/E 端号映射与厂家脚图相符。
- 器件端号/功能：四路输入 LED 对：(1,2)、(3,4)、(5,6)、(7,8)；厂家输出脚图：16=C/15=E、14=C/13=E、12=C/11=E、10=C/9=E。
- 当前清单 libUuid：`5c65181be86d494c942ee8f6bcdc1462`；Symbol UUID：`e533973ce9fb44598f9d00c0d10325ce`；Footprint UUID：`73f4598ae01341a999a1eda3d7f94fc0`
- 证据：https://www.aotesemi.com/uploadfile/202405/ffb1a2e2ebecfaf.pdf
- 附注：AT3H74 两颗位号 U_BAL_OPTO_Q1/Q2；源备注给出16C/15E→stage1、14C/13E→stage2、12C/11E→stage3、10C/9E→stage4，各E接对应级BATN、各C接对应控制栅极；这里只确认 pin function/number，不评估CTR裕量或失效行为。

### BAT46W-7-F

- 位号：D_BAT, D_BREG
- 结论：REVIEW：封装极性方向待核。D_BAT 正式源 pin1=BAT_DIODE；D_BREG pin1=BREG_DIODE；两者 pin2=CELL_B9。封装 pad 1/2 与符号 pin 1/2 齐全，但库符号只写数字，当前原始库无法证明 pad1 与器件阴极色带方向一致。
- 器件端号/功能：SOD-123：1=K（阴极，色带端），2=A（阳极）
- 当前清单 libUuid：`a4a5071e96464e3cb64a8e44e5c0c90e`；Symbol UUID：`1f07de743ab94c58be31411b983e2b2c`；Footprint UUID：`eabc9173c4664f8a90fe92f757e4228f`
- 证据：https://www.diodes.com/datasheet/download/BAT46W.pdf
- 附注：Diodes Inc. BAT46 数据表给出 SOD-123 与阴极色带。当前正式源注记也要求复核 footprint 极性带方向；D_BAT、D_BREG 均保持 REVIEW。

### BQ7695202PFBR

- 位号：U1
- 结论：PASS：仅编号对应。48 个符号端号与封装 pad 端号逐一对应，正式源 pin 1–48 的网络/空引脚键完整落在对应编号。保留意图 NC，不代表 AFE 配置、采样链或保护逻辑已验收。
- 器件端号/功能：TI BQ76952 PFB/TQFP 48-pin pinout；按数据表编号逐项匹配。
- 当前清单 libUuid：`11a7ff39fa6a461a89d259faea7aed4a`；Symbol UUID：`9d334130c9d0406ca221f499d6bd292d`；Footprint UUID：`bfef61e59b344d0eb8a970b05c25bc25`
- 证据：https://www.ti.com/lit/ds/symlink/bq76952.pdf
- 附注：仅端号对应性。项目正式源中的 null/NC 及 9S 采样分配不作功能验收。

### CA-IS2062A

- 位号：U_CAN
- 结论：PASS：引脚功能与端号对应。16 个 pin 号与封装焊盘编号对应；符号 pin 名与厂家功能号相符，正式源 CAN_RX/TX、两侧电源/地、CANL/H 映射到相同端号，NC 端保持空接。
- 器件端号/功能：pin1 RXD, 4 TXD, 5 GNDP1, 6 VDDP, 7 VDDL, 8 GND1, 9 GND2, 10 VISOIN, 11 VISOOUT, 12 GNDP2, 14 CANL, 15 CANH；2/3/13/16 NC。
- 当前清单 libUuid：`f0c935430d56463499dcfd5a80ea6af5`；Symbol UUID：`0b6c26fbbbf2439fb20cd11158cc3030`；Footprint UUID：`1bc0f77a3d724950bf13445f1e2cdc53`
- 证据：https://e.chipanalog.com/Public/Uploads/uploadfile/files/20250325/CAIS2062AdatasheetVersion1.01en20250325.pdf
- 附注：只核对 pin function 与数字映射，不评估隔离耐压、爬电距离或应用电路。

### ETA3000D2I

- 位号：U_ETA_BAL1, U_ETA_BAL2, U_ETA_BAL3, U_ETA_BAL4, U_ETA_BAL5, U_ETA_BAL6, U_ETA_BAL7, U_ETA_BAL8
- 结论：PASS：EP焊盘与BATN公共地对应。厂家 pin4=BATN 且说明为负端/公共 GND；厂家要求 exposed pad 接 GND。官方库把中心裸露焊盘编号为 CAD pin9；当前 8 颗 pin4 与 pin9 同网，均对应本级 BATN。pin9 是库编号，厂家未称其为数字脚9。
- 器件端号/功能：1 BATC,2 BATP,3 SW,4 BATN,5 BIAS,6 STAT,7 ISET,8 EN；EP connect to GND。
- 当前清单 libUuid：`15d201998529493cb3fc5dec2536f526`；Symbol UUID：`1122ffc6eef1424fbcda2055c99b4c25`；Footprint UUID：`e9425520e51749b88aba9a4607a4e5af`
- 证据：https://xonstorage.z8.web.core.windows.net/pdf/etasolution_eta3000d2i_xonlink.pdf（ETA Solutions 品牌 PDF，第2/4/11页）
- 附注：中心 pad9 1.2×0.6 mm，落在厂家第11页 D1=1.10–1.30 mm、E1=0.50–0.70 mm 区间。8/8 正式 pin4=pin9 网络已核实；详见独立 ETA EP 证据。

### FCX495TA

- 位号：Q_BREG
- 结论：部分依据。原厂 SOT-89 顶视图标出 B/C/E 与位置、C/tab，但图中没有数字脚号。当前库 symbol 为 1=B、2=C、3=E，footprint pads 1/2/3 齐全；正式源 1=BREG、2=BREG_COLLECTOR、3=REGIN 与库数字映射一致。
- 器件功能依据：原厂顶视图只标 B/C/E：tab=C、上侧引脚 E、中间引脚 C、下侧引脚 B；图中未标数字端号
- 当前清单 libUuid：`ccb55617316e45928d4c73767f613adf`；Symbol UUID：`61246eeac9fc4b8180bdbe1d98f254fb`；Footprint UUID：`56b98ce1f5664118ae6dbbfaa5718ffb`
- 证据：https://www.diodes.com/datasheet/download/FCX495.pdf
- 附注：Diodes Inc. 数据表第1页 Top View Pinout 图只标功能字母与位置。数字映射 1=B、2=C、3=E 来自当前官方库 symbol/footprint；正式源与库编号一致，但厂家数字脚序证据仍为部分。

### HYG022N10NS1TA

- 位号：Q_CHG1, Q_CHG2, Q_CHG3, Q_DSG1, Q_DSG2, Q_DSG3
- 结论：PASS：端号对应；焊盘形状待人工审阅。TOLL：pin1=G、pins2–8=S、散热 tab=D；正式源 CAD pin9 映射 D。e30 是 pin8 纵向焊盘，e31 是同编号 pin8 的横向延伸，不是新增引脚；e31 不触及 pin1 gate。
- 器件端号/功能：厂家图：1=G，2–8=S，Tab=D；项目把散热 tab 编为 CAD pin9。
- 当前清单 libUuid：`72f44b11027f4f00ad6210e6490dc8e7`；Symbol UUID：`992811325d4240078c3d0ec9919b7c97`；Footprint UUID：`ae68bf96a0ea4cd286f165c1e2f44505`
- 证据：BMS模块/厂商资料/HYG022N10NS1TA.pdf（厂家 pinout 图）；BMS模块/检查/当前精确器件库.elibz（当前归档 SHA256）
- 附注：根任务复核 rawPAD：e31 中心约 (0.640,-2.950) mm、尺寸 8.08×0.80 mm，最左约 -3.40 mm；pin1 pad 右边约 -3.775 mm，保留约 0.375 mm 间距。6 颗正式源 1=各自 gate，2–8=同一 source 网，9=FET_MID。电气编号可接受；完整 land/paste 几何仍待人类审阅。

### NCE01P30K

- 位号：Q_PRE1, Q_PRE2
- 结论：PASS：引脚功能与端号对应。P 沟道 MOS SOT-23：1=G、2=D、3=S；符号 pin 名、封装焊盘编号和正式源网络一致。
- 器件端号/功能：SOT-23 pin1=G，pin2=D，pin3=S
- 当前清单 libUuid：`99ef698ad25848ee9d345c73b8ce27d5`；Symbol UUID：`850815f6719b4df8915dd18d8390abe2`；Footprint UUID：`aff334dc45be4fb1854d8713e5ed3dd5`
- 证据：BMS模块/厂商资料/NCE01P30K_V2_Elecfans镜像.pdf；https://item.szlcsc.com/datasheet/NCE01P30K/117243.html
- 附注：厂家资料与当前正式连接 JSON 相符；不据此评估电气额定、SOA或热设计。

### PESD2CANFD24V-TR

- 位号：D_CAN_TVS
- 结论：缺库源：无法核端号/焊盘。施工清单当前有绑定 UUID，正式源 1=CANH、2=CANL、3=PACK-；但当前 elibz 中没有可读 Device/Symbol/Footprint 源，无法核对数字到焊盘。
- 器件端号/功能：本报告不从器件备注推断库端号/焊盘对应关系。
- 当前清单 libUuid：`76bcd52e6dd845eb950d4f76382feb0e`；Symbol UUID：`75d11002b26f44fb9e5fd84309d4a68c`；Footprint UUID：`bdad16194d454cb292a50525f39c3ffd`
- 证据：BMS模块/检查/当前精确器件库.elibz（按当前归档 SHA256）；当前原始库证据 tmp/BMS非RCL_原厂针脚封装功能核对_20261014.json
- 附注：保留项目绑定 UUID 作为线索，不把它当作已核验的库源。需补入当前官方 Device 源后再闭合。

### SCT2A26STER

- 位号：U_PSU
- 结论：PASS：引脚功能与端号对应。厂家 pin1 GND、2 VIN、3 EN、4 NC、5 FB、6 TM、7 BST、8 SW、9 EP；符号/封装号和正式源网络一致，4 保持未连接，9 为 B0。
- 器件端号/功能：1=GND,2=VIN,3=EN,4=NC,5=FB,6=TM,7=BST,8=SW,9=EP
- 当前清单 libUuid：`0b9a375741ff467ab6dc4066c4fab27e`；Symbol UUID：`0886204536bc492596741baf75dae57b`；Footprint UUID：`3fc9ebb0130f47f2a842308f2090c3ef`
- 证据：https://www.silicontent.com/uploads/20250604/31b34e8344440b8ab90865778ff55ab0.pdf
- 附注：只核对 pin number/function mapping，不验证电源控制器应用电路或补偿。

### TI SN74LVC1G11DCKR

- 位号：U_GATE1
- 结论：PASS：引脚功能与端号对应。DCK 6-pin：1=A、2=GND、3=B、4=Y、5=VCC、6=C；符号 pin 名、pad 编号和当前正式源对应。只核对 pin mapping，不验证 AND 逻辑时序/逻辑设计。
- 器件端号/功能：1=A, 2=GND, 3=B, 4=Y, 5=VCC, 6=C
- 当前清单 libUuid：`27376aeb8ef24d0493ee69f282c2787f`；Symbol UUID：`680e6e65287c42dd9e2d21b084cd4d03`；Footprint UUID：`40d544f497914caaa6cbea3bc4eaae91`
- 证据：https://www.ti.com/lit/ds/symlink/sn74lvc1g11.pdf
- 附注：当前 U_GATE1 正式网络分别为 SW_RUN、B0、BMS_LOCAL_OK、PRECHARGE_WINDOW、PRE_CTRL_3V3、TMR_ACTIVE。

### TI SN74LVC74ADR

- 位号：U_SEQ
- 结论：PASS：仅编号对应。14 个 pin 的双 D 触发器引脚功能号与符号/封装编号相符；当前正式源 pin 1–14 网络逐项使用对应编号。仅编号映射，不验证时序状态机。
- 器件端号/功能：1=1CLR#,2=1D,3=1CLK,4=1PRE#,5=1Q,6=1Q#,7=GND,8=2Q#,9=2Q,10=2PRE#,11=2CLK,12=2D,13=2CLR#,14=VCC
- 当前清单 libUuid：`4b9b0b3026db4c3781dcfe76716d53b7`；Symbol UUID：`83505c9b0810430b94597dea14c21f9f`；Footprint UUID：`f1b3106e9d564687958e1016421ed2c5`
- 证据：https://www.ti.com/lit/ds/symlink/sn74lvc74a.pdf
- 附注：U_SEQ pin map 对照 TI pin table；本报告不运行逻辑仿真。

### TI TPL5111DDCR

- 位号：U_TIMER
- 结论：PASS：引脚功能与端号对应。6-pin pin1=VDD、2=GND、3=DELAY/M_DRV、4=DONE、5=DRVn、6=EN/ONE_SHOT；符号/封装编号与正式源一致。
- 器件端号/功能：1=VDD,2=GND,3=DELAY/M_DRV,4=DONE,5=DRVn,6=EN/ONE_SHOT
- 当前清单 libUuid：`c3cf736bc2524174963d78e17a18aa92`；Symbol UUID：`08374e1fddbe4538b0a98b1e166cc2d2`；Footprint UUID：`99a529f2657a4db3b518ca8a2cf34b91`
- 证据：https://www.ti.com/lit/ds/symlink/tpl5111.pdf
- 附注：仅引脚映射，不评估控制模式/定时器行为。

### TI TPS3809K33DBVR

- 位号：U_POR
- 结论：PASS：引脚功能与端号对应。SOT-23：1=GND、2=RESET#、3=VDD；正式网络 B0/POR_OK_N/PRE_CTRL_3V3 与符号 pin 名、封装号一致。
- 器件端号/功能：1=GND, 2=RESET, 3=VDD
- 当前清单 libUuid：`730f62007a0a4f4aae50fb25c8097e82`；Symbol UUID：`8dd2f84cc6fe4e70a895b617093d2881`；Footprint UUID：`078e43bc93394f49815211c4fd533d19`
- 证据：https://www.ti.com/lit/ds/symlink/tps3809.pdf
- 附注：不据此判断复位时序或电源监控阈值。

### onsemi MMBT5551LT1G

- 位号：Q_QPRE_DRV
- 结论：PASS：引脚功能与端号对应。SOT-23：1=B、2=E、3=C；符号引脚名与封装 pad 编号一致，当前正式源 1=QPRE_DRV_B、2=B0、3=QPRE_PULL_C。
- 器件端号/功能：SOT-23 pin1=B，pin2=E，pin3=C
- 当前清单 libUuid：`e9aa8bc80a8e495c9e7817b3328be7bc`；Symbol UUID：`907e3848a9fe40e89b2a1beddc8a6491`；Footprint UUID：`410bf5f266b44ea2820d622d3d20afa8`
- 证据：https://www.onsemi.com/pdf/datasheet/mmbt5551-d.pdf
- 附注：onsemi 厂家数据表确认端号；当前正式源本轮重读。

### onsemi MMSZ5242BT1G

- 位号：DZ_QPRE
- 结论：PASS：引脚功能与端号对应。SOD-123 符号 pin1=K、pin2=A；封装焊盘有 1/2，正式源 1=QPRE_SRC、2=QPRE_G。与器件极性和端号一致。
- 器件端号/功能：1=K（阴极），2=A（阳极）
- 当前清单 libUuid：`c0336bfa04c5418f8f5ef4d72bbecd49`；Symbol UUID：`78897fd77ff04664aef989c24570348e`；Footprint UUID：`eabc9173c4664f8a90fe92f757e4228f`
- 证据：https://www.onsemi.com/download/data-sheet/pdf/mmsz5221bt1-d.pdf
- 附注：onsemi MMSZ52xx 系列数据表；SOD-123 pin1 阴极、pin2 阳极。仅核对引脚与封装编号，不评估稳压值/功耗裕量。

## 特别记录

- **ETA3000D2I（8 颗）**：当前正式源 pin4 与库编号 pin9 对应的中心 EP 同网，8/8 为本级 BATN。厂家 PDF 第4页要求 EP 接 GND；pin9 是库编号，不是厂家数字 pin9。尺寸对照见 [独立 EP 证据](BMS_ETA3000D2I_EP证据_20261014.md)。
- **HYG022N10NS1TA（6 颗）**：CAD pin1=G、2–8=S、9=tab D。pad e31 是同编号 8 的横向延伸，与 gate pad 不接触；端号映射可接受，完整 land/paste 由人类审阅。
- **BAT46W-7-F（D_BAT、D_BREG）**：D_BAT pin1=BAT_DIODE、D_BREG pin1=BREG_DIODE、两者 pin2=CELL_B9；封装编号集合齐全，但库原始 footprint 未证明 pad1 对实体阴极色带的方向，继续 REVIEW。
- **FCX495TA（Q_BREG）**：原厂顶视图仅标 B/C/E 功能字母与位置、C/tab，没有数字端号。当前库 symbol 1=B、2=C、3=E，footprint pad 1/2/3 齐全，正式源网络与该库编号一致；厂家数字脚序证据为部分。
- **连接器（5 颗）**：本轮确认 CAD 端号/焊盘结构和正式 JSON 编号相符；实体朝向与线束视角未验。
- **PESD2CANFD24V-TR（D_CAN_TVS）**：当前清单已有 UUID 绑定，但当前精确 elibz 缺少可读取的器件源；没有把旧绑定或项目备注当成库 pad 证据。

## 范围边界

本报告只核对当前正式源 pin→net 与当前器件库 symbol pin / footprint pad 编号、厂家 pinout 证据的对应关系。PASS 不等于完整原理图功能验收，也不覆盖参数、功耗、热、逻辑时序或布板验收。未运行测试、ERC、DRC、仿真。

JSON 逐项记录（含当前网络、UUID、符号引脚、焊盘编号/坐标、来源文件和 SHA256）：[BMS非RCL_功能核对阶段结论_20261014.json](BMS非RCL_功能核对阶段结论_20261014.json)。

原始过程快照已收入 `归档/历史与调试.zip`，当前结构依据见同名 JSON。
