# 稳压模块原理图准备

本目录存放稳压模块的设计输入、BOM、连接/采样合同及器件候选资料。现行工程为 8 颗 LICAP SC0350-270-RSS、8S1P；文件中标注为候选、旧版或来源片段的内容不能据此视为现行原理图或已验证器件。当前状态见[模块 README](../README.md)与[现行全图复核](../检查/全功能与逐脚复核.md)。

## 本目录直接文件

| 文件 | 用途与边界 |
|---|---|
| [BOM.csv](BOM.csv) | 保存当前设计物料清单输入；实际采购、价格和资格状态需结合库索引及模块当前检查核对。 |
| [design-plan.json](design-plan.json) | 保存当前设计计划的结构化内容，包括器件、分区、连线和系统边界。 |
| [power_plan.json](power_plan.json) | 保存原理图页的供电与器件安排数据。 |
| [interface_plan.json](interface_plan.json) | 保存接口网络、器件绑定、别名及未闭合绑定的结构化计划。 |
| [connector_devices.json](connector_devices.json) | 保存连接器器件候选及其来源/校验边界。 |
| [AFE启动配置.json](AFE启动配置.json) | 记录 AFE、超级电容组参数和启动顺序的配置输入；其中资格状态须以文件字段和当前复核为准。 |
| [单页设计注释.json](单页设计注释.json) | 保存原理图单页设计注释的结构化数据。 |
| [接口与采样核对.md](接口与采样核对.md) | 记录接口与采样路径的文字核对；其正文注明对应 V1.3 准备资料，不能代替当前工程检查。 |
| [接口与采样核对.json](接口与采样核对.json) | 与接口/采样核对有关的结构化输入和状态信息；按文件中的版本、范围和未决项使用。 |
| [采样与硬关断实现.md](采样与硬关断实现.md) | 说明逐节测量与本地硬关断实现边界，以及候选/未实施路径。 |
| [磁件选型.md](磁件选型.md) | 记录单电感及大电流端子的候选参数与筛选边界，文件明确说明尚非验证或生产冻结选型。 |
| [母线电容选型.md](母线电容选型.md) | 保存母线大电容纹波电流的候选选型分析。 |
| [增容与保护优化.md](增容与保护优化.md) | 保留增容及逐组保护的计算与约束；16 颗方案已取消，正文标明其计算仅供追溯。 |
| [DRV8161_UCC_TLV_参数核对_20261010.md](DRV8161_UCC_TLV_参数核对_20261010.md) | 作为 2026-10-10 的原理图参数侧车文件，内容针对当时 V1.3 候选，非当前验证结论。 |
| [R76IR4220SE40J_封装源与无封装处理说明.md](R76IR4220SE40J_封装源与无封装处理说明.md) | 说明同目录 R76 电阻封装源码片段的来源和处理边界。 |
| [R76IR4220SE40J_86d1aff7974043ff.epcb2-source.txt](R76IR4220SE40J_86d1aff7974043ff.epcb2-source.txt) | 从 FOC 原生板文件抽取的完整封装文档源码段；是否可用于本模块须按配套说明和机械核对判断。 |
| [URB4812YMD-10WR3_FOC_footprint_review.json](URB4812YMD-10WR3_FOC_footprint_review.json) | 保存从 FOC 资料抽取的 URB4812YMD-10WR3 封装复核结构化记录。 |
| [URB4812YMD-10WR3_FOC_A3_a47ad9c165087638.epcb2-source.txt](URB4812YMD-10WR3_FOC_A3_a47ad9c165087638.epcb2-source.txt) | 保存 URB4812YMD-10WR3 的 FOC 来源封装文档源码段，用于候选复核，不表示已经采用。 |
| [IHXL2000VZEB100M3A_candidate.metadata.json](IHXL2000VZEB100M3A_candidate.metadata.json) | 保存 IHXL2000VZEB100M3A 电感候选的来源、几何与验证状态元数据。 |
| [IHXL2000VZEB100M3A_candidate.footprint.jsonl](IHXL2000VZEB100M3A_candidate.footprint.jsonl) | 保存该 IHXL 电感候选的封装数据记录，和配套 metadata 一起用于候选核查。 |
| [PEH226ML3900QE4_candidate_parameters.json](PEH226ML3900QE4_candidate_parameters.json) | 保存 PEH226ML3900QE4 电容候选的电气、端子、封装和安装参数。 |
| [PEH226ML3900QE4_project_candidate_radial_vertical_15p24.epcb2-source.txt](PEH226ML3900QE4_project_candidate_radial_vertical_15p24.epcb2-source.txt) | 保存该 PEH226 电容径向垂直候选封装的项目源码片段，不等同于当前工程绑定或机械验收。 |
