# SLE Range Link Sample（1D 测距 + 双向数据）

基于 SDK 自带 `application/samples/products/sle_measure_dis`（SLE Channel Sounding 测距）
与 `application/samples/products/sle_uart`（SSAP 数据）实现的 Sample：两块 BearPi-Pico H2821E，
**SERVER = 锚点**（广播 + 测距 + 回传距离 + 收用户数据），**CLIENT = 标签**（扫描 + 连接 + 发起 CS 测距 + 周期发用户数据 + 收距离回传）。
两块板之间只有 1 个距离值（1D），不做坐标。

## 目录结构

```
sle_range_link/
├─ Kconfig
├─ CMakeLists.txt
├─ sle_range_link.c                  # 入口：sle_range_link_entry()（唯一 app_run）
├─ rl_ext.c/.h                       # 本工程扩展层：校准、AT 命令、距离回传和用户数据
├─ range_link_filter.c/.h            # server 侧滑动中值+均值滤波、野值剔除
├─ range_link_server/                # 复制自 sle_measure_dis_server（内部符号名保持 measure_dis_*）
│   ├─ sle_measure_dis_server.c      # 属性 NOTIFY、距离回传、用户数据 RX、[RL] 日志
│   ├─ sle_measure_dis_server.h      # RL 报文结构/消息类型/RL_LOG
│   ├─ sle_measure_dis_server_adv.c/.h   # 未改
│   └─ sle_measure_dis_server_alg.c/.h   # kzalloc/负值/距离回调/安静日志（calib_val 保持 0）
└─ range_link_client/                # 复制自 sle_measure_dis_client
    ├─ sle_measure_dis_client.c      # 周期发用户数据、收到 DIST 打印、[RL] 日志
    ├─ sle_measure_dis_client.h      # RL 报文结构
    ├─ sle_measure_dis_client_slem.c/.h    # 仅把逐帧日志改为 RL_LOG（逻辑未改）
```

## 本目录直接文件

| 文件 | 用途 |
|---|---|
| `Kconfig` | 声明 sample 的 SERVER/CLIENT 角色、数据周期、滤波、校准、notify 与日志配置项。 |
| `CMakeLists.txt` | 将本 sample 的入口、扩展层、滤波器以及所选角色的官方样例源文件加入构建。 |
| `sle_range_link.c` | 注册唯一 `app_run` 入口并启动角色任务；SERVER 任务处理测距队列、喂看门狗并在空闲时让出 CPU，CLIENT 任务按配置周期发送用户数据。 |
| `rl_ext.c` | 实现本工程扩展逻辑：距离校准/多点拟合和 AT 命令、距离过滤/日志/notify、用户数据收发及客户端距离通知处理。 |
| `rl_ext.h` | 定义扩展层使用的 SSAP 消息类型与载荷结构，并声明 SERVER/CLIENT 钩子接口。 |
| `range_link_filter.c` | 实现 server 距离样本的窗口中值、均值输出、野值门控和连续异常后的窗口重置。 |
| `range_link_filter.h` | 暴露滤波器初始化、重置、压入样本接口及窗口/野值门限配置。 |

`range_link_client/` 和 `range_link_server/` 的直接源码文件均列在上方目录树中，
它们从海思 `sle_measure_dis` 官方 sample 派生；本工程修改点通过 `[RANGE_LINK]` 钩子标记，
源码用途和来源集中在本 README 说明，不在这些上游派生目录中重复放置 README。

## 通信协议（SSAP，沿用样例帧头 `u32 type; u32 len; u8 data[]`）

| 类型 | 值 | 方向 | 载荷 |
|---|---|---|---|
| RL_MSG_IQ | 0xFFFFFFEA | client → server | IQ 帧（沿用样例，测距必需） |
| RL_MSG_USER_DATA | 0xFFFFFFE1 | client → server | `rl_user_payload_t`（20B：seq/uptime_ms/payload[12]） |
| RL_MSG_DIST | 0xFFFFFFE0 | server → client | `rl_dist_payload_t`（36B：seq/dist_mm/raw_mm/prob_pm/rssi_dbm/smp_num/rx_seq/rx_count/rx_len） |

距离回传使用 `measure_dis_server_write_client()`（原样例存在但从未被调用）；属性
`operate_indication` 同时置 `READ | WRITE_NO_RSP | WRITE | NOTIFY`（既允许 client 的
`write_cmd` 写 IQ/用户数据，又允许 server notify 距离）。

## Kconfig 选项

| 选项 | 默认 | 说明 |
|---|---|---|
| `SAMPLE_SUPPORT_SLE_RANGE_LINK` | n | 打开本 sample（products Kconfig 中） |
| `SAMPLE_SUPPORT_SLE_RANGE_LINK_SERVER/_CLIENT` | SERVER | 角色二选一 |
| `RANGE_LINK_USER_PERIOD_MS` | 5000 | client 发送用户数据周期；**周期太小会干扰 IQ 上传导致测距偏小**（实测 500ms 使 0.5m 原始值 1583→1420mm），追求精度用 5000ms |
| `RANGE_LINK_FILTER_WIN` | 1 | server 距离滤波窗口（1=直通，使用库自身 Kalman 平滑值，与官方一致） |
| `RANGE_LINK_CALIB_MM` | 0 | 校准偏移（mm）：`d_cal = d_raw * SCALE_PM/1000 - CALIB_MM`；按官方方法填 `mean(dist_first@1m) - 1000` |
| `RANGE_LINK_CALIB_SCALE_PM` | 1000 | 校准比例（千分比，1000=1.0）；官方方法只偏移，比例一般保持 1000 |
| `RANGE_LINK_DIST_NOTIFY` | n | server 把距离 notify 回 client（已硬件验证可用；本工程出厂配置为 y） |
| `RANGE_LINK_RX_PRINT` | n | 在 SSAP 写回调里逐包打印用户数据（默认关，避免回调上下文大量打印） |
| `RANGE_LINK_VERBOSE` | n | 打开原样例 IQ/中间日志（默认只打 `[RL]` 行） |

## 构建

```powershell
$tc='D:\Project\NearLink\ToolChain'
$env:Path="$tc\tools\python;$tc\tools\python\Scripts;$tc\tools\Windows\ninja;$tc\tools\Windows\cc_riscv32_win_env;"+$env:Path
cd D:\Project\NearLink\SDK\bearpi-pico_h2821e
python build.py -c -j8 standard-bs21e-1100e
```

`.config` 必需项（每版一次）：
`CONFIG_SAMPLE_ENABLE=y`、`CONFIG_ENABLE_PRODUCTS_SAMPLE=y`、`CONFIG_SAMPLE_SUPPORT_SLE_RANGE_LINK=y`、
角色二选一、`CONFIG_PRODUCT_SLE_MEASURE_DIS=y`、`# CONFIG_SUPPORT_SLE_BLE_CENTRAL_DEFAULT is not set`、
`# CONFIG_SAMPLE_SUPPORT_SLE_MEASURE_DIS is not set`（**禁止与原 measure_dis sample 同时打开**）。

可用 `build\script\usr_config.py --command=setconfig ...` 落盘配置，例如：

```powershell
python build\script\usr_config.py --command=setconfig --chip=bs21e --core=acore --target=standard-bs21e-1100e `
  SAMPLE_ENABLE=y ENABLE_PRODUCTS_SAMPLE=y ENABLE_PERIPHERAL_SAMPLE=n `
  SAMPLE_SUPPORT_SLE_MEASURE_DIS=n SAMPLE_SUPPORT_SLE_RANGE_LINK=y `
  SAMPLE_SUPPORT_SLE_RANGE_LINK_SERVER=y PRODUCT_SLE_MEASURE_DIS=y SUPPORT_SLE_BLE_CENTRAL_DEFAULT=n
```

## 串口输出（115200 8N1，无需回车）

```
[RL] boot role=SERVER build=...
[RL] connected conn_id=%d
[RL] server D=1234mm RAW=1250mm PROB=900 RSSI=-45dBm N=8 RX_SEQ=42 RX_CNT=43
[RL] server RX seq=42 len=12 data=ABCDEFGHIJKL
[RL] disconnected, adv restart disc_reason=0x...
```

```
[RL] boot role=CLIENT build=...
[RL] connected conn_id=%d
[RL] client TX seq=42 len=20
[RL] client D=1234mm PROB=900 RSSI=-45 N=8
[RL] disconnected, rescan disc_reason=0x...
```

- 距离行最多 2 行/秒（`RL_DIST_REPORT_MS=500` 节流，打印与 notify 同节流）。
- 用户数据行按 client 发送周期（默认 500ms）。
- `CONFIG_RANGE_LINK_VERBOSE=y` 才输出 IQ/中间日志。

## 校准与使用（现场步骤，官方方法）

1. 环境：锚点周围 3m 内空旷、无墙体/柱体/金属遮挡；板载天线同向竖直、离地约 1m。
2. **官方校准方法（单点偏移）**：两板相距 **1.00m**，静置 ≥5s 待 `SN=6` 收敛，记录 30 帧
   `D1=`（库原始 `dist_first`，未校准）取中位数 `M`（单位 mm）；
   设 `CONFIG_RANGE_LINK_CALIB_SCALE_PM=1000`、`CONFIG_RANGE_LINK_CALIB_MM=M-1000`（例如 M=2073 → 1073）。
3. 重新编译烧录；现场看 `[RL] server D=`（校准后距离）、`D1/D2`（原始值，用于复标）、`PROB`、`SN`。
4. 若需多点/更远距离精度：分别在 0.5m/1.0m（或 1.0/2.0m）记录 D1 中位数，做线性拟合
   `SCALE_PM=1000*Δ真值/ΔD1`、`CALIB_MM=SCALE_PM*D1_1m/1000-1000`。
5. 注意：**测距原始值会随环境/摆位漂移**（实测同一 1m 不同会话 D1 约 2.07~2.29m），换环境后建议重标；
   数据业务周期建议 5000ms（见 Kconfig 说明），否则会干扰 IQ 导致偏小。

## 稳定性说明（重要）

SDK 的看门狗只在 **idle 任务**里喂狗（`drivers/chips/bs2x/main_init/idle_config.c`）。server 任务处理 CS/IQ 时
会长时间占用 CPU，若不让出 CPU 会导致 idle/log 饿死，约 8 秒后触发 CWDT NMI 复位
（`APP|exception:8000000c / Oops:NMI`）。本 sample 的 server 任务已修复：
每批消息处理完 `uapi_watchdog_kick()` + `osal_msleep(1)`；队列溢出打印默认静默；队列 16 条 + 批量排空。
实测修复后连续 90 秒以上无复位/无断连。

## 硬件联调建议（两块板烧不同镜像）

1. 先烧 SERVER 到 A 板、CLIENT 到 B 板，串口 115200 观察上电 banner（确认角色/镜像没烧反）。
2. 两块板相距 ~1m，等待 `[RL] connected`，确认 client 出现距离行、server 出现 `RX seq`、client 出现 `TX seq`。
3. 若 client 收不到 `[RL] client D=`：检查 server 串口是否持续有 `[RL] server D=`（server 侧打印必成功）；
   若 server 侧 notify 失败会打印 `[RL] server send to client failed, ret=0x...`，属已知的 notify 订阅风险，见报告。
4. 距离校准：按上一节流程得到偏移量，改 `CONFIG_RANGE_LINK_CALIB_MM` 重烧后复测。
5. 拉远/遮挡场景观察稳定性：滤波会丢弃 >max(500mm, 30%) 的野值，`N=` 为累计有效样本。

## 已知限制

- 只做 1D 距离，无坐标（SDK 仅提供 `libcal_dis.a` 1D 解算）。
- notify 回传依赖 SSAP notify 订阅在硬件上可用；不可用时 server 侧打印仍可读距离，client 侧距离行会缺失。
