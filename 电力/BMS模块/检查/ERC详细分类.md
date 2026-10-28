# ERC 详细日志分类

- 来源：`E:\Pioneer\先进通用履带底盘工程\电力\BMS模块\检查\ERC与日志.json`
- 来源 SHA256：`3f023e5a1bcecb59c1eba9d6e0ca278eeff6fcf67b25b5e87995fab53cb8e6a7`
- 生成时间：`2026-10-12T07:41:58+00:00`
- 汇总计数：fatalError 18、error 9、warn 1903
- `logs`记录数：38；按类型：info 37、openProject 1
- 稳定消息模板组：3

## 逐项 ERC 明细可用性

| 类型 | 汇总数量 | 单条日志数 | ref/pin/page/nativeID/原消息 |
|---|---:|---:|---|
| fatalError | 18 | 0 | 当前快照未提供 |
| error | 9 | 0 | 当前快照未提供 |
| warn | 1903 | 0 | 当前快照未提供 |

日志文件的 `result` 仅记录 18/9/1903 的汇总数；`logs` 数组共 38 条，只有 `openProject` 1 条和 `info` 37 条，没有 fatalError/error/warn 单条记录。因此无法可靠列出这些 DRC 项的位号、引脚、页码、nativeID 或原消息，也不能据此归因具体 ERC 原因。

## 网表生成位号通知

以下通知属于网表生成时的 `info` 日志，不是 ERC 错误。

### `info` × 25

- 模板：The Component {objectID} designator is repeated with other component(s), and it is annotated a new designator: {designatorValue} when generating the netlist.
- 记录：Q_CHG1（$2I2；page=82222f7cb9045883；nativeID=2089e130ea1265be）、F1（$2I15；page=82222f7cb9045883；nativeID=59813e85a09b8aef）、Q_CHG2（$2I24；page=82222f7cb9045883；nativeID=c84addcdc0ae2982）、Q_CHG3（$2I28；page=82222f7cb9045883；nativeID=43a29c1434e5db30）、Q_DSG1（$2I32；page=82222f7cb9045883；nativeID=a66a2beef0c760f6）、Q_DSG2（$2I36；page=82222f7cb9045883；nativeID=6a37a4043880e153）、Q_DSG3（$2I40；page=82222f7cb9045883；nativeID=75a322a19588586a）、NTC_CELL1（$18I2；page=5cb4ecec4ea23139；nativeID=b3d65d6bdcb19dbc）、NTC_CELL2（$18I14；page=5cb4ecec4ea23139；nativeID=8ba715e05156ab79）、NTC_CELL3（$18I26；page=5cb4ecec4ea23139；nativeID=502203d88b037a3b）、NTC_CELL4（$19I2；page=8930b73b61f55a8b；nativeID=4bd66a4c6cd5ab40）、NTC_CELL5（$19I14；page=8930b73b61f55a8b；nativeID=b5920dd6961db380）、NTC_CELL6（$20I3；page=931fb1216d355c07；nativeID=81a5740ed1b03079）、NTC_CELL7（$20I9；page=931fb1216d355c07；nativeID=22d725c657f45b45）、NTC_CELL8（$20I15；page=931fb1216d355c07；nativeID=682c4e1854c4a047）、NTC_CELL9（$20I21；page=931fb1216d355c07；nativeID=8187e002d0bb744e）、T_BAL1（$26I2；page=5306dbe872cc1cfc；nativeID=3694ea1314b9d136）、T_BAL2（$27I2；page=d439b52359ab29cb；nativeID=c97586547f992a75）、T_BAL3（$28I2；page=01c5bfb3701d5dfc；nativeID=ca48be49cc1ca2e1）、T_BAL4（$29I2；page=6e339e3b9b98a4e3；nativeID=5a2ddec3ea1cd6ba）、T_BAL5（$30I2；page=c7322d80a4a88fa8；nativeID=3d11376b28a0aa18）、T_BAL6（$31I2；page=a6a6154d5b226719；nativeID=2d87c22316e4291d）、T_BAL7（$32I2；page=9558eb9b89500ac8；nativeID=5b2a3ae160987dc3）、T_BAL8（$33I2；page=133ee40a6d31857b；nativeID=44939d30c0a82c8a）、T_BAL9（$34I2；page=22e5f987ed409c2d；nativeID=85392e886ef5a061）

### `info` × 12

- 模板：The component {objectID} is annotated a designator: {designatorValue} when generating the netlist.
- 记录：J_BAT_HV（$2I12；page=82222f7cb9045883；nativeID=d8a3212c730d98a6）、J_PACK_HV（$2I18；page=82222f7cb9045883；nativeID=d59965ad00aaa6bc）、RSHUNT（$2I44；page=82222f7cb9045883；nativeID=cfee7f9da36b9283）、D_TVS（$2I61；page=82222f7cb9045883；nativeID=2f6a44976cbea854）、SW_RUN_STORE_A（$6I2；page=6d2dd82fa41b09c5；nativeID=e4d1a371353314eb）、SW_RUN_STORE_B（$6I6；page=6d2dd82fa41b09c5；nativeID=7a999a70ec551d4b）、R_PRE（$10I38；page=183acb640045154b；nativeID=6d38817b42d55780）、NTC_BOARD（$19I26；page=8930b73b61f55a8b；nativeID=9b928be75edca222）、NTC_CORE（$19I38；page=8930b73b61f55a8b；nativeID=121789bbdea61ab8）、NTC_MOS（$20I27；page=931fb1216d355c07；nativeID=406480f74cafe05a）、J_BAL_SERVICE（$38I2；page=37e458e6a3230331；nativeID=0a606731201b99c8）、J_CELL_SERVICE（$38I3；page=37e458e6a3230331；nativeID=c65c6ba1aad3a636）

## 其他日志

- `openProject` × 1：`BMS-9S4P`。
- 当前日志快照没有逐条警告消息；1903条warn只在汇总计数中出现。
