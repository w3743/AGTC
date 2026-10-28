# 通信 —— 小熊派 H2821E 双板 SLE 测距与双向数据

基于海思 BS21E（BearPi-Pico H2821E，星闪 SLE）的两块开发板之间的 **1D 测距 + 双向数据** 工程。
测距使用芯片原生 **SLE Channel Sounding（CS，SDK 中为 HADM）** 能力，距离解算由官方预编译算法库
`libcal_dis.a` 完成；本工程在官方示例 `sle_measure_dis` 之上增加了现场校准、滤波与双向数据链路。

## 验收结论（±10cm 要求）

| 实际距离 | 实测中位 | 误差 | σ | 说明 |
|---|---|---|---|---|
| 0.50 m | 485 mm | **−15 mm** | ≈10 mm | 三点线性校准后 |
| 1.00 m | 981 mm | **−19 mm** | ≈6 mm | 三点线性校准后 |

- 校准参数：`SCALE=462‰ / OFFSET=88 mm`（等价于 `raw = 2.161 × 真实距离 + 190 mm`）；
- 满足「误差不超过 ±10cm」的要求，留有约 5 倍余量；
- 复位/异常：长时间采集 0 次看门狗复位。

> 边界条件：原始距离值本身会随摆放/多径慢漂（实测约 85mm / 10 分钟），换算到距离误差约 ±40mm。
> 漂移累计超过约 200mm 时，在已知距离处补一个校准点（发一条 `AT+RDFIT=<mm>`）重新拟合即可，无需重烧固件。

## 目录结构

```
通信/
├── src/
│   ├── sle_range_link/         # 我们的示例工程源码（可直接放入 SDK 的 application/samples/products/）
│   └── sdk_patches/            # 需要同步到 SDK 的 4 个文件（保持相对路径，说明见其中 README）
├── firmware/
│   ├── server_v31/             # A 板（server / 锚点）冻结固件包与构建元数据，烧到 COM22
│   └── client_v16/             # B 板（client / 标签）冻结固件包，烧到 COM23
├── docs/
│   ├── range_link/             # 使用/校准/实测记录/官方基线
│   ├── design/                 # 设计说明、重构说明（官方源码 + 钩子 + 扩展层）
│   └── research/               # 官方测距算法逆向研究、校准参数研究
└── tools/
    └── calibrate.ps1           # 一键采集→改校准常量→编译→出烧录命令
```

`firmware/` 保存已生成的固件快照：`server_v31/bs21e_all.fwpkg` 是 server 镜像，
`client_v16/bs21e_all.fwpkg` 是 client 镜像；`server_v31/build_info.txt` 记录该 server
产物的标识、构建时间、功能/库版本、AT 命令接口和固件包大小。固件目录是冻结交付物，
其用途与版本信息在本上级说明中维护。

## 工程结构（关键设计）

`src/sle_range_link` 采用「**官方源码 + 少量钩子 + 扩展层**」的结构：

| 文件 | 来源 | 说明 |
|---|---|---|
| `range_link_server/`、`range_link_client/` | 官方 `application/samples/products/sle_measure_dis` | 与上游 master 逐文件 diff 仅 7 行/文件（`#include "rl_ext.h"` + `kzalloc` 修正 + `[RANGE_LINK]` 钩子） |
| `rl_ext.c/.h` | 本工程新增 | 校准、AT 命令、滤波接入、结构化打印、notify、双向数据 |
| `range_link_filter.c/.h` | 本工程新增 | 中值窗口 + 野值门控 + 连续超差重置 |
| `sle_range_link.c` | 基于官方入口改写 | 入口任务：每轮喂狗、扩展初始化、客户端周期发数据 |

搜索 `[RANGE_LINK]` 可定位与官方源码的全部差异；重构说明见 `docs/design/range_link_refactor.md`。

## 编译

1. 将 `src/sle_range_link/` 复制到 SDK 的 `application/samples/products/` 下；
2. 按 `src/sdk_patches/README.md` 同步 4 个 SDK 集成文件（示例 Kconfig/CMakeLists/芯片 sample 配置/任务栈配置）；
3. 在 SDK 根目录执行（目标名必须用连字符）：

```powershell
# 环境（Windows）
$tc='D:\Project\NearLink\ToolChain'
$env:Path="$tc\tools\python;$tc\tools\python\Scripts;$tc\tools\Windows\ninja;$tc\tools\Windows\cc_riscv32_win_env;"+$env:Path
cd SDK\bearpi-pico_h2821e
python build.py -c -j8 standard-bs21e-1100e
```

产物：`tools/pkg/fwpkg/bs21e/bs21e_all.fwpkg`。

## 烧录

```powershell
$hs='<ToolChain>\tools\hsflash\hsflash.exe'
# A 板（server，COM22）
& $hs write -c bs21e -p COM22 -b 750000 -f firmware\server_v31\bs21e_all.fwpkg --after-reset
# B 板（client，COM23）
& $hs write -c bs21e -p COM23 -b 750000 -f firmware\client_v16\bs21e_all.fwpkg --after-reset
```

手法：**按住板载 Reset → 执行命令 → 等工具开始等待后松开 Reset**。

## 使用（运行时校准，不需要重烧固件）

串口：`COM22`，`115200 8N1`，发送需带回车换行。SDK 自带 AT 命令框架（与烧录同一 CH340 口）。

| 命令 | 作用 |
|---|---|
| `AT+RDCLR=1` | 清空标定点 |
| `AT+RDFIT=<实际mm>` | 在某个已知距离记录一个拟合点；记录到 2 个点自动做最小二乘并生效（最多 4 点） |
| `AT+RDFIT?` | 查询：`+RDFIT:<点数>,<SCALE‰>,<OFFSETmm>,<点1真值>,<点1原始>,...` |
| `AT+RDTARE=<实际mm>` | 单点对零（快速，不改变刻度） |
| `AT+RDTARE?` | 查询当前原始中位数/偏移/刻度 |

典型标定流程：

```text
（两块板按实际使用距离摆好，天线面正对平行、同高度、无遮挡，静置 15~20 秒）
AT+RDCLR=1
（放到 0.50m）静置 15~20 秒 → AT+RDFIT=500
（放到 1.00m）静置 15~20 秒 → AT+RDFIT=1000
AT+RDFIT?            # 核对参数
```

测距日志格式：

```text
[RL] server D=981mm D1=2310mm D2=2314mm PROB=999 SN=6 RSSI=-40dBm SC=462 OF=88 RX_SEQ=4032 RX_CNT=121
```

- `D`：校准+滤波后的距离；`D1`：算法库原始平滑值（未校准）；`D2`：未平滑候选值；
- `PROB`：置信度（RSSI 驱动，非精度指标）；`SN`：库内平滑帧数（6 为收敛）；
- `SC/OF`：当前线性校准参数；`RX_SEQ/RX_CNT`：收到的客户端用户数据包序号/计数。

## 已知限制

1. **近场偏差大**：官方算法在 0.5~1m 的原始输出约为真实距离的 2.2 倍 + 150mm 常量偏置，必须现场校准；
2. **原始值慢漂**：几分钟到十几分钟级别漂移 100~300mm，长时间使用建议在已知距离重新对零/拟合；
3. **摆放极敏感**：天线遮挡/姿态变化会让 RSSI 掉 20dB 以上、原始值跳动 1m 以上，此状态下测量不可信；
   建议：天线面正对平行、同高度、离桌面 ≥10cm、避开金属/人体/显示器；
4. **TOF 未启用**：本 SDK 构建未开 CS Mode3，`tof_result` 恒为 0（应用层已实测确认）；
5. 仅 1D 距离，不做坐标/角度；坐标需多锚点自研解算，角度需 SLP（SP31A）方案硬件。

## 参考

- 官方示例：`fbb_bs2x/src/application/samples/products/sle_measure_dis`（HiSpark 官方仓库）
- 官方测距算法研究：`docs/research/sle-ranging-algorithm-analysis.md`
- 官方基线实测：`docs/range_link/official-baseline.md`
- 实测记录：`docs/range_link/demo-evidence.log`
- 重构说明（官方源码 + 钩子）：`docs/design/range_link_refactor.md`
