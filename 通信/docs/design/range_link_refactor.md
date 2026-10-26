# range_link 工程重构说明（基于官方 sle_measure_dis 复用）

> 更新：2026-10-25 22:20｜目标：**不再维护一份被改得面目全非的官方代码副本**，
> 改为「官方源码 + 极少量带标记的钩子 + 我们自己的扩展文件」。
>
> 官方来源：`https://gitcode.com/HiSpark/fbb_bs2x/tree/master/src/application/samples/products/sle_measure_dis`
> （等价镜像：`https://gitee.com/HiSpark/fbb_bs2x`，本文件的对比基线为 gitee master = BS2X 1.0.18 的 sample）

## 1. 目录结构

```
application/samples/products/sle_range_link/
├── CMakeLists.txt                 # 我们的 target：官方源文件 + rl_ext.c + filter.c
├── Kconfig                        # 角色选择 + 滤波/校准/数据周期等配置
├── sle_range_link.c               # 入口（基于官方 sle_measure_dis.c，加喂狗/扩展初始化/客户端发送任务）
├── rl_ext.h / rl_ext.c            # ★我们自己的全部功能（校准/AT 命令/打印/notify/双向数据）
├── range_link_filter.c / .h       # ★中值滤波 + 野值门控
├── range_link_server/             # 官方 server 源码（仅 3 处 [RANGE_LINK] 钩子）
│   ├── sle_measure_dis_server.c/.h
│   ├── sle_measure_dis_server_adv.c/.h
│   └── sle_measure_dis_server_alg.c/.h
└── range_link_client/             # 官方 client 源码（仅 2 处 [RANGE_LINK] 钩子）
    ├── sle_measure_dis_client.c/.h
    └── sle_measure_dis_client_slem.c/.h
```

`★` 标记的是我们新增的文件；其余文件内容 = 官方 master 原样 + 下文列出的钩子行。

## 2. 与官方源码的差异（全部差异，可逐条核对）

重构后逐文件 diff 行数（对照官方 master）：

| 文件 | 差异行数 | 差异内容 |
|---|---|---|
| `range_link_server/sle_measure_dis_server.c` | 7 | `#include "rl_ext.h"` + write 消息里新增 `case RL_MSG_USER_DATA:`（用户数据转发给扩展层） |
| `range_link_server/sle_measure_dis_server_alg.c` | 7 | `#include "rl_ext.h"` + `kmalloc→kzalloc`（修正官方未清零导致 `tof_*` 带堆垃圾）+ 在 `measure_dis_posalg_set_distance()` 之后调用 `rl_ext_on_distance()` |
| `range_link_server/sle_measure_dis_server.h` | 0 | 原样 |
| `range_link_server/sle_measure_dis_server_adv.c/.h` | 0 | 原样 |
| `range_link_client/sle_measure_dis_client.c` | 7 | `#include "rl_ext.h"` + 收到通知时新增 `case RL_MSG_DIST:`（打印 server 回传的距离） |
| `range_link_client/sle_measure_dis_client_slem.c/.h` | 0 | 原样 |
| `sle_range_link.c`（入口） | 与官方 `sle_measure_dis.c` 同构 | 仅加：server 任务每轮 `uapi_watchdog_kick()`、`rl_ext_server_init()`、client 周期发用户数据 |

> 重建/升级官方代码时只需重放以上钩子（搜索 `[RANGE_LINK]` 即可定位）。

## 3. 我们新增的能力（都在 rl_ext.c / range_link_filter.c）

1. **应用层校准**：`cal = raw × SCALE_PM/1000 − OFFSET_MM`（官方 `calib_val` 保持 0，避免双重校准）。
2. **运行时对零**：`AT+RDTARE=<mm>`（把当前距离当真实值，反推偏移）/ `AT+RDTARE?` 查询。
3. **两点/多点线性拟合**：`AT+RDFIT=<mm>` 每个距离点发一次（≥2 点自动最小二乘 `raw=a·true+b`），
   `AT+RDFIT?` 查询、`AT+RDCLR=<任意数>` 清空。
4. **滤波**：中值窗口（`RANGE_LINK_FILTER_WIN`）+ 野值门控 + 连续超差重置（动态场景跟随）。
5. **门限**：丢弃 `dist_first<=0`（库错误帧）、`prob < PROB_MIN`、`smooth_num<3`、`RSSI < -70dBm` 的帧。
6. **诊断打印**：`[RL] server D=... D1=... D2=... PROB=... SN=... RSSI=... SC=... OF=... RX_SEQ=... RX_CNT=...`
7. **双向数据**：client 每 `RANGE_LINK_USER_PERIOD_MS` 发 20B 用户数据（`RL_MSG_USER_DATA`）；
   server 可选 notify 距离回传（`RANGE_LINK_DIST_NOTIFY`，默认关）。
8. **看门狗适配**：SDK 只在 idle 任务喂狗，server 连续处理 IQ 时会饿死 idle（8s CWDT 复位）；
   entry 的 server 任务每轮主动 `uapi_watchdog_kick()`，队列空时才 `osal_msleep(1)`。

## 4. 与官方行为的关系（哪些是官方行为、哪些是我们的）

| 行为 | 来源 |
|---|---|
| SLE 连接/广播/扫描、CS 使能、IQ 上报、`slem_alg_calc_smoothed_dis()` 距离计算 | 官方（未改） |
| `dis_smoothed`（D1）/`dis_ori`（D2）/`prob`/`smooth_num` | 官方库输出（未改） |
| 校准/滤波/打印/notify/AT 校准/用户数据 | 我们（rl_ext.c） |
| 官方 1m 单点校准（`calib_val`） | 保留但保持 0；我们在应用层做（可运行时改） |

## 5. 如何重新同步官方代码（升级 SDK 时）

1. 从上游下载 `samples/products/sle_measure_dis` 的 12 个源文件；
2. 覆盖 `range_link_server/`、`range_link_client/` 对应文件（**不要**覆盖 `rl_ext.c`、`sle_range_link.c`、`CMakeLists.txt`、`Kconfig`）；
3. 在新文件里重新插入 `[RANGE_LINK]` 钩子（搜索本仓库备份可对照）：
   - `server_alg.c`：`#include "rl_ext.h"`、`osal_kzalloc`、`rl_ext_on_distance(measure_dis_temp);`
   - `server.c`：`#include "rl_ext.h"`、`case RL_MSG_USER_DATA: rl_ext_on_user_data(...)`
   - `client.c`：`#include "rl_ext.h"`、`case RL_MSG_DIST: rl_ext_on_client_msg(...)`
4. 编译；若上游结构体/API 变化，只需改 `rl_ext.c` 的适配层。

## 6. 备份

重构前的旧代码完整备份：`Artifacts/backup_range_link_src_20261025/`
