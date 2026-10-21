# BREG供电选型收尾研究

日期：2026-10-13。仅核对原厂资料、渠道页面和静态功耗；未修改原理图/BOM/TODO，未运行测试。输入条件采用当前设计记录：风扇恢复 EL357 后，REG1 LED 支路最坏电阻限流估算为 22.73mA；REG1 总负载按 30mA；最高 B9=37.8V；R_BREG_FEED=300Ω、拟用 RC2512FK-07300RL；REGIN 由 Q_BREG 维持约5.5V。

## 结论

- **PCN-2664 没有把 FCX495TA 列为 EOL。** PCN 表3列的是 `FCX495-7 → FCX495TA`、`FCX495-13R → FCX495TC`；现用料号 `FCX495TA` 在 Diodes 2024年6月版数据表仍标为 Active。因此无需因产品页出现 PCN-2664 而更换现用料号。
- **FCX495QTA 是可行的同封装候选，渠道有供货线索，但不是解决热耗散的替代方案。** 原厂资料给出 SOT89、相同的 B-C-E 引脚次序、150V/1A、1W及相同的125°C/W结到环境热阻；QTA另有AEC-Q101/PPAP资格。Mouser检索快照显示有现货并可单颗购买；库存会变化，且本次未核实大陆人民币报价。
- **30mA正常负载处于REG1规格范围内，但Q_BREG高温连续耗散余量不足。** 按标称条件估计约0.69W，保守角落约0.72W。SOT89数据表的1W是在25°C及指定高覆铜测试板条件下；85°C环境的降额上限约0.52W。故如果板内环境可能达到85°C，不能据此认定Q_BREG连续工作合格；改成QTA不会改变结论。
- **RC2512FK-07300RL在30mA正常预算下有较大电阻功率余量。** 含1%公差最坏约0.273W，低于1W@70°C额定值。REG1短路时的47–80mA限流是另一工况：若达到80mA，300Ω电阻瞬时/持续功耗上界为1.92W；数据表没有给出允许持续时间，不能据此认为1W电阻可承受持续短路。

## 核对与推导

### REG1负载

TI BQ76952数据表对REG1设为3.3V的规格为：当 `VREGIN ≥ 4.1V`、负载 `0–45mA` 时，输出范围 `3.0–3.6V`。所以22.73mA单支路低于45mA；按总负载30mA计算，额定电流余量为15mA。22.73mA是给定的“仅按限流电阻”估算，本报告未重新核算EL357输入支路；实际所有REG1负载仍需合计，且须保证REGIN不低于4.1V。

TI另列REG1输出短路电流限值47–80mA；这不是可用的正常负载预算。REG0外部BJT选型章节明确要求BJT考虑REG1最大45mA及最高约80mA短路电流。

### R_BREG_FEED

Yageo对精确料号 RC2512FK-07300RL 标注300Ω、±1%、2512、1W（70°C），最高工作温度155°C；该规格页注明商业级且非AEC-Q200。

- 正常30mA、取最坏电阻303Ω：`P_R = I²R = 0.03² × 303 = 0.273W`。
- REG1正常负载上限45mA、303Ω：`P_R = 0.045² × 303 = 0.614W`。
- 若REG1短路电流达到80mA：`P_R = 0.08² × 303 = 1.94W`（标称300Ω时1.92W）。高于器件1W额定值；持续时间、短路是否由系统及时解除，以及实际电流波形均未由现有资料证明。

### Q_BREG热耗散与SOA

B9经BAT46W和300Ω串联电阻供给FCX495TA集电极，发射极为REGIN。以30mA作为Q集电极电流近似：

`VCE ≈ 37.8V − VF − I×R − VREGIN`

取示例标称 `VF≈0.45V`（BAT46W原厂在10mA给出的最大VF为0.45V；30mA处不把该数值当作保证值）、`R=300Ω`、`VREGIN=5.5V`：`VCE≈22.85V`，`PQ≈0.686W`，与约0.68W估值一致。为估算更高发热角落，取 `VF=0V`、`R=297Ω`、`VREGIN=5.0V`，得 `PQ≈0.717W`。计算把REG1总负载等同于Q集电流，未计REGIN额外负载和启动瞬态；最终应以实际电流与板上温度核对。

Diodes FCX495TA数据表给出：PD=1W、RθJA=125°C/W、TJ(max)=150°C；1W条件基于15mm×15mm×1.6mm FR4、单面高覆盖1oz铜、静止空气。按其从25°C/1W到150°C/0W的降额曲线作一阶估算，85°C环境允许耗散约 `(150−85)/125=0.52W`。0.686W在25°C下对应Tj约111°C，在85°C下约171°C；0.717W约对应115°C和175°C。即使不考虑额外设计余量，30mA连续工况在高环境温度下也越过该热额定边界。实际PCB铜面积、层叠和气流尚未由本研究评估，不能用数据表测试板数据替代实际热验收。

FCX495TA数据表的SOA图包含DC曲线；在本估算的约23V、30mA工作点，总功耗约0.69W，低于25°C指定测试板的1W热额定线。环境升高时，首要约束是热降额；脉冲SOA不能证明该连续工况合格。FCX495QTA同为1W、125°C/W等级，因此只可作为生命周期/车规资格选择，不会提升本电路热能力。

## 料号与供货核查

| 项目 | 核查结果 |
|---|---|
| PCN-2664适用范围 | EOL：FCX495-7（推荐FCX495TA）、FCX495-13R（推荐FCX495TC）；不包含现用FCX495TA作为EOL料。 |
| FCX495TA | 原厂数据表订购表标为Active，SOT89，B-C-E引脚；维持现有料号是最少改动方案。 |
| FCX495QTA | 原厂FCX495Q数据表列出该订购料号；同为SOT89、150V/1A/1W，原厂产品页标注Automotive，资料支持AEC-Q101/PPAP。属于规格匹配候选，PCN-2664并未把它指定为TA的正式PCN替代料。 |
| 供货 | Mouser的FCX495QTA页面/检索快照（约2026年9月）显示773件库存、可单颗购买。渠道库存是动态信息；本次没有确认大陆现货、人民币含税价或正式报价。 |

**建议：** 暂保留FCX495TA，不因PCN误判而换料；如项目要求AEC-Q101资格，可把FCX495QTA作为同封装候选，但采购前复核实时库存与正式报价。封板前应先由PCB设计者按真实铜面积、环境温度和30mA持续负载核实Q_BREG热余量，并单独确定REG1短路期间R_BREG_FEED的持续时间/能量边界；如要求85°C环境连续运行，现有SOT89/1W热等级没有通过纸面核算。

## 来源

- TI：[BQ76952数据表 Rev.B](https://www.ti.com/lit/ds/symlink/bq76952.pdf)，§7.11–7.12、§12.3、§16.2.2。
- Diodes：[PCN-2664 EOL通知](https://www.diodes.com/assets/PCN-Files/Diodes_PCN_2664_-EOL.pdf)；[FCX495数据表](https://www.diodes.com/datasheet/download/FCX495.pdf)；[FCX495Q数据表](https://www.diodes.com/datasheet/download/FCX495Q.pdf)；[FCX495Q产品页](https://www.diodes.com/part/view/FCX495Q)；[Master CoC料号状态表](https://www.diodes.com/assets/Quality-Reliability-Docs/Master_CofC.pdf)。
- Diodes：[BAT46W数据表](https://www.diodes.com/datasheet/download/BAT46W.pdf)。
- Yageo：[RC2512FK-07300RL精确料号规格页](https://www.yageogroup.com/component-documentation/download/specsheet/RC2512FK-07300RL)。
- 渠道：[Mouser FCX495QTA](https://www.mouser.com/en/ProductDetail/Diodes-Incorporated/FCX495QTA?qs=u16ybLDytRZZ46GM8f2mkA%3D%3D)。库存快照不构成报价或长期供货承诺。
