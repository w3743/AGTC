# STM32G431CBU6 核心板参考资料

整理日期：2026-10-28。这里收集芯片官方资料和一款采用该型号的公开核心板资料，供原理图、接口及机械尺寸核对。用户实际核心板的厂家、版本尚未确认；WeAct 图纸仅作为候选参考，使用前应与实物丝印、引脚和尺寸核对。

## 芯片官方资料（STMicroelectronics）

| 本地文件 | 用途 | 官方来源 |
| --- | --- | --- |
| [DS12589-STM32G431.pdf](DS12589-STM32G431.pdf) | 数据手册：型号、引脚复用、电气特性、UFQFPN48 封装及尺寸。 | [DS12589](https://www.st.com/resource/en/datasheet/stm32g431cb.pdf) |
| [RM0440-STM32G4.pdf](RM0440-STM32G4.pdf) | STM32G4 系列寄存器及外设参考手册；留意各型号功能差异。 | [RM0440](https://www.st.com/resource/en/reference_manual/rm0440-stm32g4-series-advanced-armbased-32bit-mcus-stmicroelectronics.pdf) |
| [ES0431-勘误.pdf](ES0431-勘误.pdf) | G431/G441 勘误，使用时按实物芯片修订版核对适用项及规避方法。 | [ES0431](https://www.st.com/resource/en/errata_sheet/es0431-stm32g431xx441xx-device-errata-stmicroelectronics.pdf) |
| [AN5093-硬件设计.pdf](AN5093-硬件设计.pdf) | G4 系列硬件开发：供电、复位、时钟及调试等设计参考。 | [AN5093](https://www.st.com/resource/en/application_note/an5093-getting-started-with-stm32g4-series--hardware-development-boards-stmicroelectronics.pdf) |

STM32G431CBU6 为 STM32G431CB，UFQFPN48 封装；具体引脚和电气条件以 DS12589 对应封装及型号为准。G431 的外设不能直接按 G474 等其他 G4 型号套用。资料更新可查 [STM32G431CB 产品页](https://www.st.com/en/microcontrollers-microprocessors/stm32g431cb.html) 和 [G4 官方文档目录](https://www.st.com/en/microcontrollers-microprocessors/stm32g4-series/documentation.html)。

## WeAct Studio 核心板公开参考

以下来自 [WeActStudio.STM32G431CoreBoard](https://github.com/WeActStudio/WeActStudio.STM32G431CoreBoard)，固定来源提交 `0af1943ee21bf7d5e172a2bfc6480d44aa229de3`。仓库同时包含 CBU6 和 CBT6；这里选取对应 CxUx 的 QFN48 标准版图纸。图纸是板级参考，不代表本工程已有实物或经过电气验收。

| 版本 | 原理图 | 外形 |
| --- | --- | --- |
| 标准版 V1.0 | [原理图](WeAct-QFN48-标准版-原理图.pdf) | [外形图](WeAct-QFN48-标准版-外形.pdf) |

[WeAct 原始中文说明](WeAct-README原文.md) 为仓库原文摘存，文中的相对图片与目录链接需在上游仓库查看。软件例程和其他封装版本也可从上游仓库获取。

## 使用时核对

1. 对照板上型号、版本和尺寸确认是否为上述 WeAct 版本；核对电源输入、稳压器、USB、BOOT、复位、SWD 及排针信号，不把另一块板的排针定义直接用于实物。
2. 确认实际芯片修订版及 ES0431 对所用外设的适用性；供电、引脚允许范围和封装以官方数据手册为依据。
3. 板级接入本工程前，结合目标模块检查供电、电平、接口冲突和保护；资料收集不等于完成设计或实物验证。
