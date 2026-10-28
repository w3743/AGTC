# V1.6-BUSFIX 逐节保护模型

## 本目录直接文件

| 文件 | 用途与边界 |
|---|---|
| [cell_protection_qualification.py](cell_protection_qualification.py) | 运行 BUSFIX 超级电容逐节保护状态机模型；参数来自 `../../输入/V1.6-BUSFIX/model_input.json`，持久化模型输出位于 `../../结果/V1.6-BUSFIX/cell_protection_qualification.json`。模型覆盖的软件状态用例不证明 AFE 模拟采样、功率级、自举、失效独立关断或硬件资格，适用边界见[模块当前说明](../../../README.md)。 |
