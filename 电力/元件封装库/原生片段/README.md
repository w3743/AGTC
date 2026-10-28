# 原生库资产片段

本目录按 EasyEDA 资产类型保存从当前工程提取的原生内容片段。资产索引 [资产索引.json](../资产索引.json) 将器件及其符号、封装引用映射到具体文件；片段文件名为内容 SHA256，不能仅凭文件名推断型号。

## 子目录

| 目录 | 直接文件用途 |
|---|---|
| [DEVICE/](DEVICE/README.md) | `.jsonl` 文件保存原生 DEVICE 资产内容记录。 |
| [FOOTPRINT/](FOOTPRINT/README.md) | `.jsonl` 文件保存原生 FOOTPRINT 资产内容记录。 |
| [SYMBOL/](SYMBOL/README.md) | `.jsonl` 文件保存原生 SYMBOL 资产内容记录。 |

这些片段供内容追溯和库索引核对，不是可以单独导入的完整库包。需要更新时使用 `../工具/更新元件库.py`，并核对生成索引；不要直接改写片段文件。
