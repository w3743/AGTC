# 通信工具

本目录直接文件：

| 文件 | 用途 |
|---|---|
| [`calibrate.ps1`](calibrate.ps1) | SERVER 侧现场校准脚本：采集 `[RL] server D=` 串口行，取原始 `D1` 中位数并计算偏移，更新 SDK `.config` 后构建 SERVER 固件、复制产物/构建信息并打印烧录命令。 |

脚本针对其源码中配置的 `D:\Project\NearLink` 目录布局，默认输出名仍为 `range_link_v24`；运行前需确认串口、SDK/工具链路径和输出名与当前环境匹配。它会写入采集日志、SDK 配置、构建输出及固件产物，不是只读检查工具。
