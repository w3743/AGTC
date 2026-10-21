# RoboMaster 参考与精简裁决

日期：2026-10-10。版本：V1.3。用途：记录公开结构参考及本方案的取舍，不把参考项目指标当作本模块验证。

## 公开来源

- DonotFreeze 的 [RM2024 超级电容控制板 Lite 硬件篇](https://donotfreeze.github.io/robomaster/supercap-controller-lite-hardware/)列出 STM32G431CBT6、EG2181、单半桥 MOS、串联 P-MOS、电感及电流采样等结构。原始[硬件开源发布页](https://bbs.robomaster.com/article/54930)标注 CC BY 4.0；引用或复用该硬件资料须署名并遵守该许可。文章同时说明该板主要依赖软件保护并建议增加保险；这不能证明本方案的软件 BUS dump 已验证，也不能替代源端保险或逐节监测。
- [Lite 固定提交 0c9482f9e2fa320f29d1dad2f92fa85f1684977c](https://github.com/DonotFreeze/RM-PCB-SuperCapControlBoard_Lite/commit/0c9482f9e2fa320f29d1dad2f92fa85f1684977c)与[Plus 固定提交 3d02b40cf8433b0fa90568786ff958e7bf22f620](https://github.com/DonotFreeze/RM-PCB-SuperCapControlBoard_Plus/commit/3d02b40cf8433b0fa90568786ff958e7bf22f620)用于定位公开项目文档版本。Lite 与 Plus 是同一作者/团队的连续迭代，不是两套独立验证。
- [RM-SuperCapCtrlLib 固定提交 4e594415700edb38e2d2f6c126c2049e27b9a173 的源码树](https://github.com/DonotFreeze/RM-SuperCapCtrlLib/tree/4e594415700edb38e2d2f6c126c2049e27b9a173)与[该提交的 MIT 许可证](https://github.com/DonotFreeze/RM-SuperCapCtrlLib/blob/4e594415700edb38e2d2f6c126c2049e27b9a173/LICENSE.txt)。尚未移植该库代码。
- 另存的 [FYT 技术报告](参考资料/RoboMaster2024_FYT技术报告.pdf)属于另一支赛事队伍的资料，仅作阅读参考，不归属 DonotFreeze 项目，也不作为本方案验证证据。

## 本方案采用与保留

采用 WeAct Normal QFN48 STM32G431CBU6 核心板、单半桥、2只主N-MOS、单电感和单P-MOS正常关断；核心板VCC统一5V、板载3.3V供逻辑，优先核定复用本地候选PSU5_EXT（URB4805YMD-10WR3），不把其2A额定当作余量；12V驱动与比较器供电优先评估URB4812YMD-10WR3，尚未定型，RECOM仅作规格备选参考。低压初充（包括从0V开始）属于正常受控充电，不是额外自动恢复功能；优先评估现有双向DC/DC受控限流充电并保留逐节测量/过压保护。当前AFE/驱动从0V上电测量、自举、最小脉宽和门控未获证明，资格闭合前禁止未经验证的自动PWM；外部限流电源仅限装配/台架调试，不是用户指定日常条件。本轮不增加独立恢复模块，正常充电实现待原理核准。DRV8161集成半桥驱动、CSA、VDS监测和驱动保护；国内采购价待询，集成不等于已证明最低总成本。超容均衡采用BW6101每节被动泄放。

本轮母线电阻泄放改为核心板 G431 PB7/TIM4_CH2 软件 PWM 直接控制 UCC27511A（IN+接 PWM、IN−接 GND），不保留 BUS 硬比较器触发、二极管 OR 或硬件启动抑制；因此没有独立硬件 BUS_OV dump，MCU/ADC/PWM 故障可能使泄放失效。底板独立 IC 为 5 种/12 颗；核心板和供电模块总成另列，内部器件/成本仍计入全系统。TLV6700 仅保留 SC 总压比较；SC 逐节监测、三处 NTC、IWDG、源 DC 保险及残压维护仍保留。P-MOS 只作正常关断，不冒称可覆盖上下管短路故障；需证明电流衰减与保险故障能量，否则回退反串双 N-MOS 隔离方案。

## 不能据此推断的事项

公开板或开源库不证明本方案在 9S 22.5–37.8 V、8S 超容、55 A、0.8–1 kW/1 s 下已验证。没有完成同版设计、原理图、固件移植、仿真、热测试、短路关断、过压 dump 或车载测试；赛事波形和对方样机结论不能外推为本模块指标。
