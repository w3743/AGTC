# 预充期间关闭 FOC Buck：双NPN源连接与接口交接

日期：2026-10-14。已把双NPN钳位连接写入 FOC 单板连接源及 FOC 保护/接口交接文档。它仍是待原生工程合并、角落验证和实物验证的设计源；没有改 BOM、PCB 或 EasyEDA 原生工程，也没有运行测试、仿真、ERC/DRC。

## 当前实施连接

FOC 单板连接源：[当前原理图连接.json](../../FOC驱动模块/原理图准备/当前原理图连接.json)，已新增 7 个器件/接口记录：`Q_BUCK_CLAMP`、`R_BUCK_CLAMP_BUS`、`Q_BUCK_RELEASE`、`R_BUCK_VALID_BASE`、`R_BUCK_VALID_PD`、`R_LOAD_VALID_BIAS`、`J_LOAD_POWER_VALID`。源描述一块 FOC 板；整套由两块同款板按 A/B 装配差异复制。

FOC-A 与 FOC-B 的板端和 BMS 侧统一使用 DEGSON `2EDGVC-5.08-02P-14-100A(H)`（LCSC C581328）；每块 FOC 使用一只配套线端 `2EDGK-5.08-02P-14-1000A(H)`（LCSC C581323），共两只。2P 针序一致：pin 1=`LOAD_EN_BIAS_EXT`，pin 2=`LOAD_POWER_VALID`。把两块 FOC 的同号针脚分别接到 BMS 侧 `J_LOAD_POWER_VALID` pin 1/2。pin 1 是接收侧光耦偏置，pin 2 是正逻辑许可。两块 FOC 的接收侧参考地使用现有 `BUS−/GND_SYS`；接口不加 BMS `B0` 地线，也不把 pin 1 当作电源输出。厂商图未标 pin 1 识别点，layout须明确封装/丝印方向并保持此针号映射。

只在 FOC-A 安装 `R_LOAD_VALID_BIAS=10 kΩ`，连接本板 `SAFE_5V` 到 pin 1。FOC-B 相同的上拉焊盘/位号 DNP，pin 1 只随线束接入许可偏置线，不接 FOC-B `SAFE_5V`。严禁两板同时安装上拉或并接两块板的 SAFE_5V。接口对接插头为两只，按每块FOC一个另计；不含BMS已有端的插头。

每块 FOC 的钳位如下：

| 位号 | 器件/封装 | 引脚连接 |
|---|---|---|
| Q_BUCK_CLAMP | onsemi MMBT3904LT1G，SOT-23 | pin 1=B 接 R_BUCK_CLAMP_BUS；pin 2=E 接本板 GND_SYS；pin 3=C 接 U1 DRV8353RS pin 47 `BUCK_RT_SD` |
| R_BUCK_CLAMP_BUS | 100 kΩ，0603，1% | BUS 到 Q_BUCK_CLAMP 基极 |
| Q_BUCK_RELEASE | onsemi MMBT3904LT1G，SOT-23 | pin 1=B 接 R_BUCK_VALID_BASE；pin 2=E 接本板 GND_SYS；pin 3=C 接 Q_BUCK_CLAMP 基极 |
| R_BUCK_VALID_BASE | 10 kΩ，0603，1% | `LOAD_POWER_VALID` 到 Q_BUCK_RELEASE 基极 |
| R_BUCK_VALID_PD | 100 kΩ，0603，1% | Q_BUCK_RELEASE 基极到本板 GND_SYS |
| U1_BUCK_RT | 保留现有 280 kΩ，0603，1% | U1 pin 47 `BUCK_RT_SD` 到 BUS，不拆除 |

FOC 源记录 `R_LOAD_VALID_BIAS` 的 BOM标志为装配默认件；两板系统装配表须把FOC-B同位号标为DNP。没有新增跳线或第二种连接器。FOC接口交接位置：[保护与固件约束.md](../../FOC驱动模块/保护与固件约束.md) 的“预充期间关闭本板 Buck”一节。

## 默认状态和有限电气判断

BUS 上升时，`R_BUCK_CLAMP_BUS` 给 Q_BUCK_CLAMP 提供基流，令其把 `BUCK_RT_SD` 拉低；`LOAD_POWER_VALID` 低、断线或掉电时，Q_BUCK_RELEASE 由 100 kΩ 下拉保持关闭，Buck 默认关闭。只有 BMS 完成预充并进入允许主输出的控制流程，光耦许可才应有效；许可有效时 Q_BUCK_RELEASE 拉低 Q_BUCK_CLAMP 的基极，释放 RT/SD。该单线许可不覆盖信号短到高等任意单故障。

按 BUS=22.5–37.8 V、两支路电阻±1%、VBE约0.7 V作名义估算：Q_BUCK_CLAMP 基流约0.22–0.37 mA；现有280 kΩ RT支路电流约0.08–0.14 mA。许可节点约1.7 V时，每板Q_BUCK_RELEASE基流约0.093 mA，两板共约0.19 mA；共享10 kΩ偏置在SAFE_5V约4.77 V时的可用电流约0.27–0.30 mA（取光耦输出压降0.2–0.4 V）。这是有限名义算术，不是全温保证。onsemi MMBT3904资料的饱和压降最大值在较高测试电流点规定，没有直接保证本电路约0.1–0.4 mA电流下的全温 VCE(sat)；双板许可电平、光耦CTR/老化、RT/SD低于0.4 V的最低关断阈值仍需后续电气角落和首板核验。

此电路只关闭 DRV8353 内部 Buck。独立 TPS7A4001 产生的 SAFE_5V 在预充时仍带电，钳位不移除 LDO、比较器和阈值网络耗电；FOC-A另有共享偏置支路电流。预充预算应继续包含 SAFE_5V 自耗与这条偏置负载，不能把 Buck 关闭写成 FOC 整板无负载。

## 数量

| 范围 | NPN | 每板钳位电阻 | 上拉位置/实装 | 2P板端座 | 安装/数量 |
|---|---:|---:|---:|---:|---|
| 单块 FOC | 2 | 3 | 1个位置 | 1 | 5件本地钳位件；上拉仅FOC-A安装 |
| 两块 FOC 系统 | 4 | 6 | 2个位置、1只实装 | 2 | 11件本地钳位/偏置电子件 + 2个板端座；另有2只FOC对接插头 |
| BMS 板 | 0新增 | 0新增 | 复用现有光耦输出 | 现有1个 | 本轮未改BMS源/BOM |

FOC单板连接源7条新增记录包含单板器件和接口，不代表系统装配最终件数；双板的上拉差异以A安装、B DNP处理。

## 归档和改动边界

改前文件已追加进对应历史归档并解压流读回 SHA256 一致：

- FOC连接源条目 `原理图准备/20261014_051648708_buck_clamp_dual_npn_before.json`，SHA256 `AA57E60DC8D2B7475346EAA2C66FFD6448C070939F77F8E8D383F3097ACD6F21`。
- FOC接口/保护文档条目 `保护与固件约束/20261014_051745937_buck_clamp_dual_npn_before.md`，SHA256 `F3D9677D9A38E337637A298DDD53F55B936D3941C5FF8966C3F950B644D1CC57`。
- 本报告旧稿条目 `检查/20261014_051648708_buck_clamp_dual_npn_before.md`，SHA256 `FF2CADB5888D969FE318ECEF4F5C7B049E7468B64D957BC8B246D8C85375870E`。

本轮统一 DEGSON 端子型号前，也将当时三份文件写入各自历史 ZIP，并逐条读回 SHA256 一致：

- FOC连接源条目 `原理图准备/20261014_052006568_buck_clamp_terminal_before.json`，SHA256 `BA8A252A019075382B9196182ECDCC0F8E77368CE96A248EFE6C56469941ADD1`。
- FOC接口/保护文档条目 `保护与固件约束/20261014_052006568_buck_clamp_terminal_before.md`，SHA256 `6BBB60C2FDBD9F24D103C8E3EF9DEB7026E687EC79400E3FCA39154F4D88A509`。
- 本报告改前条目 `检查/20261014_052006568_buck_clamp_terminal_before.md`，SHA256 `EDDA7AC0ECFD8B8F833DF5AD8597600E7DB0034C4514723D2E30C8DC02B22761`。

归档位置分别为 `FOC驱动模块/归档/历史与调试.zip`、`BMS模块/归档/历史与调试.zip`；三条目的读回哈希均与存档前原文件相同。当前仅修改 FOC 连接 JSON、FOC 保护/接口文档和本检查报告。BMS源/BOM、FOC BOM、总计数、PCB和原生工程均未修改；未运行测试、仿真、ERC/DRC。

## 原厂依据

- [onsemi MMBT3904LT1G 数据手册](https://www.onsemi.com/download/data-sheet/pdf/mmbt3904lt1-d.pdf)：SOT-23 pin 1=B、2=E、3=C及器件电气特性；低电流全温饱和压降未由本设计工作点的保证表覆盖。
- [TI DRV8353 数据手册](https://www.ti.com/lit/ds/symlink/drv8353.pdf)：RS版本集成Buck、RT/SD引脚及约0.4 V最小关断阈值。
- [Vishay VO617A 数据手册](https://www.vishay.com/docs/83425/vo617a.pdf)：隔离光耦输出特性；当前双板电流点不等同数据手册CTR最小测试点。
