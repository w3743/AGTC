# FOC EDA 自动化材料

本目录保存面向嘉立创EDA官方API/CLI的FOC原理图读取、单次修改、导出、持久化和网表核对脚本。若脚本绑定固定页面UUID、位号或对象ID，运行前需先核对目标工程和源状态；保留的修改脚本不是自动化批量更新入口。`logs/`中的API/CLI调用记录属于运行日志，不作为原理图源或当前验收结论。

## 直接文件

| 文件 | 用途 |
|---|---|
| [Current-Source-CLI.py](Current-Source-CLI.py) | 备份后通过官方CLI操作当前FOC源，并管理由该工具创建的会话。 |
| [Persist-FocNative.py](Persist-FocNative.py) | 将API编辑后的FOC页面及校验过的本地/导出库内容持久化到原生项目文件。 |
| [Verify-FocNetlist.py](Verify-FocNetlist.py) | 只读比较官方API读回、官方网表与维护的引脚合同。 |
| [Invoke-Lceda.ps1](Invoke-Lceda.ps1) | 通过本地桥接会话调用嘉立创EDA脚本，支持内联代码或脚本文件输入。 |
| [Run-Lceda.ps1](Run-Lceda.ps1) | 调用嘉立创EDA CLI，并按参数记录命名日志、超时和进程结果。 |
| [Export-FocPreview.ps1](Export-FocPreview.ps1) | 调用官方导出接口生成FOC原理图PDF预览并校验写出文件大小与SHA256。 |
| [Export-CurrentPdf.js](Export-CurrentPdf.js) | 从指定FOC原理图页面导出PDF。 |
| [Export-CurrentSource.js](Export-CurrentSource.js) | 保存指定页面并导出ENET、BOM和DRC结果。 |
| [Export-SingleSheetOfficialNetlist.js](Export-SingleSheetOfficialNetlist.js) | 从指定单页FOC工程读取并返回官方ENET网表。 |
| [Read-SingleSheetActualGeometry.js](Read-SingleSheetActualGeometry.js) | 读取指定单页原理图中的器件、引脚、连线、端口及图面几何信息。 |
| [Read-BusOvLibraries.js](Read-BusOvLibraries.js) | 查询BOV参考链路相关器件在EDA库中的匹配记录。 |
| [Apply-BusOvReference.js](Apply-BusOvReference.js) | 在对象身份/数量前置条件匹配时，对指定页面实施一次BOV参考链路原理图编辑。 |
| [Connect-BusOvReference.js](Connect-BusOvReference.js) | 在指定FOC页面补充BOV参考链路的连线和网络端口，并执行连接阶段重复检查。 |
| [Correct-BusOvProcurement.js](Correct-BusOvProcurement.js) | 核对并更新BOV参考分压器/比较器去耦器件的采购属性。 |
| [0819f05c4eef4c71ace90d822a990e87](0819f05c4eef4c71ace90d822a990e87) | SQLite格式的EDA应用数据文件；包含工程、原理图、器件等表结构，但当前工程/原理图/器件表为空。它不是当前FOC原理图源或必需设计输入。 |

上述脚本只说明各自实现的自动化动作；当前工程入口与验收状态仍以[FOC模块说明](../README.md)和`检查/`当前证据为准。
