# SLE Range Link 设计说明（两块 H2821E：数据传输 + 测距）

> 版本：v1（2026-10-25）｜目标芯片：BS21E（BearPi-Pico H2821E）｜构建 target：`standard-bs21e-1100e`
> 依据：SDK 自带 `application/samples/products/sle_measure_dis`（SLE HADM/Channel Sounding 测距）+ `sle_uart`（SSAP 数据）验证结果。

## 1. 目标与边界

**做**：
- 板 A = **SERVER**（锚点角色）：广播、接受连接、参与 Channel Sounding、用 SDK `libcal_dis.a` 算距离，串口输出距离；接收板 B 周期性发来的用户数据并打印。
- 板 B = **CLIENT**（标签角色）：扫描并连接 A，发起 CS 测距；周期性向 A 写用户数据；接收 A 回传的「距离+状态」并打印。
- 双向数据都走**同一条 SLE 连接**（SSAP），与 CS 测距并存（SDK 原样例就是用 SSAP 传 IQ 帧）。

**不做（本期）**：
- 坐标 (x,y,z)。两块板只有 1 个距离值；坐标需 ≥3 块坐标已知锚点 + 解算节点，SDK 无坐标解算库（只有 1D `libcal_dis.a`），属后续项目期。
- 多锚点/GTTT、BLE 兼容、OTA。

**已确认的关键事实（来自只读调研）**：
- 官方样例：server=广播方、client=扫描方；**CS 参数与使能只由 client 在连接后发起**，server 只注册回调；距离只在 server 端计算/打印；距离**未回传** client（回传封装存在但从未调用）。
- 构建必须 `CONFIG_PRODUCT_SLE_MEASURE_DIS=y`（决定链接 `protocol/bt/*/bs21e-1100e-sle-measure-dis/lib*.a`），否则链接失败/回退错误库。
- 打开 `CONFIG_SAMPLE_SUPPORT_SLE_MEASURE_DIS` 会由 `drivers/chips/bs2x/bs2x_sample_config.cmake` 给全工程追加 `-DSLEM_CARKEY -DMEASURE_DIS -DEM_32K_SUPPORT`（影响 ITCM/栈大小/AT 分支），自研 sample 必须**同样追加**这三个宏。
- 样例已知坑：`alg.c` 里校准真正生效的位置是 `slem_posalg_set_base_para()`（文档图指向的是死函数）；`osal_kmalloc` 未清零导致 `tof_*` 用堆垃圾；负距离打印异常；每帧打印 3~5 行日志。
- 资源：SERVER 版 FLASH 已用 **94.6%**（525,792 / 555,520 B），新增代码必须保持小（目标 ≤ 6 KB），不得新增大数组/大栈。

## 2. 目录与文件（新增，不动原 sample 一个字节）

```
application/samples/products/sle_range_link/
├─ Kconfig                      # 新 sample 的开关 + 角色 choice + 参数
├─ CMakeLists.txt               # 复制原文件，路径改为本目录
├─ sle_range_link.c             # 复制 sle_measure_dis.c，仅改：入口函数名 sle_range_link_entry、
│                               #   任务名/栈/打印 banner；角色宏改成本 sample 的 CONFIG_*
├─ range_link_server/           # 复制 sle_measure_dis_server/，文件名不变，内部符号名保持 measure_dis_*（同时只编一个 sample，不会冲突）
│   ├─ sle_measure_dis_server.c        # 改：属性加 NOTIFY；write 回调加用户数据分发；距离回传调用
│   ├─ sle_measure_dis_server.h        # 改：新增消息类型/报文结构
│   ├─ sle_measure_dis_server_adv.c/.h # 不改
│   └─ sle_measure_dis_server_alg.c/.h # 改：算完距离后回调回传 + 修 kzalloc/负值
└─ range_link_client/           # 复制 sle_measure_dis_client/，同上
    ├─ sle_measure_dis_client.c        # 改：周期发用户数据；收到 DIST 通知打印
    ├─ sle_measure_dis_client.h        # 改：新增报文结构/周期配置
    ├─ sle_measure_dis_client_slem.c/.h# 不改
    └─ range_link_filter.c/.h          # 新增：滑动中值+均值滤波、野值剔除（server 侧用）
```

### 需要改的 3 处 SDK 注册文件（除此之外不动其它 SDK 文件）
1. `application/samples/products/Kconfig`：仿 `SAMPLE_SUPPORT_SLE_MEASURE_DIS`（:265-277）新增：
   ```
   config SAMPLE_SUPPORT_SLE_RANGE_LINK
       bool "Support SLE Range Link sample (distance + data)."
       depends on ENABLE_PRODUCTS_SAMPLE
       default n
       help ...
   if SAMPLE_SUPPORT_SLE_RANGE_LINK
   osource "application/samples/products/sle_range_link/Kconfig"
   endif
   ```
2. `application/samples/products/CMakeLists.txt`：仿 :154-156 加
   `if(DEFINED CONFIG_SAMPLE_SUPPORT_SLE_RANGE_LINK) add_subdirectory_if_exist(sle_range_link) endif()`
3. `drivers/chips/bs2x/bs2x_sample_config.cmake`：仿 :52-63 的 measure_dis 块，给
   `CONFIG_SAMPLE_SUPPORT_SLE_RANGE_LINK` 追加 `SLEM_CARKEY / MEASURE_DIS / EM_32K_SUPPORT` 三个 define（否则 ITCM 0x14000、栈尺寸、AT 分支与 measure_dis 不一致，可能跑飞）。

**禁止**：同时打开 `CONFIG_SAMPLE_SUPPORT_SLE_MEASURE_DIS` 与 `CONFIG_SAMPLE_SUPPORT_SLE_RANGE_LINK`（全局 `measure_dis_*` 符号+两个 `app_run` 会冲突）。构建只开 `CONFIG_PRODUCT_SLE_MEASURE_DIS=y`（库变体）+ 本 sample。

## 3. 通信协议（沿用样例帧头：`u32 type; u32 len; u8 data[]`）

```c
#define RL_MSG_IQ        0xFFFFFFEAu  /* 沿用样例：client 上报 IQ 帧 */
#define RL_MSG_DIST      0xFFFFFFE0u  /* server → client：距离+状态 */
#define RL_MSG_USER_DATA 0xFFFFFFE1u  /* client → server：用户数据 */

typedef struct {                       /* 20B payload */
    uint32_t seq;                      /* 用户数据包序号，从 0 递增 */
    uint32_t uptime_ms;                /* client 本地上电毫秒（osal_get_time 类 API，可简化用计数） */
    uint8_t  payload[12];              /* 用户数据（默认可打印 ASCII/递增字节） */
} rl_user_payload_t;

typedef struct {                       /* 32B payload，全部用整数避免浮点解析 */
    uint32_t seq;                      /* 测距结果序号 */
    int32_t  dist_mm;                  /* 滤波后距离 (mm) */
    int32_t  raw_mm;                   /* 原始距离 (mm) */
    int32_t  prob_pm;                  /* 置信度千分比 0..1000 */
    int32_t  rssi_dbm;                 /* 服务器侧 RSSI */
    uint32_t smp_num;                  /* 累计有效样本数 */
    uint32_t rx_seq;                   /* 最近收到的用户数据 seq */
    uint32_t rx_count;                 /* 累计收到用户数据包数 */
    uint32_t rx_len;                   /* 最近一包 payload 长度 */
} rl_dist_payload_t;
```

## 4. 代码修改点（按函数名定位，复制后行号会漂移）

### 4.1 SERVER 侧
1. **属性支持通知**：`measure_dis_server_add()` 里 `property.permissions = READ|WRITE` 附近，补
   `property.operate_indication = SSAP_OPERATE_INDICATION_BIT_READ | SSAP_OPERATE_INDICATION_BIT_NOTIFY;`
   （参考 `sle_uart_server.c:187-188`）。否则回传 notify 会被栈拒绝。
2. **用户数据接收**：`measure_dis_server_msg_proc()`（原 server.c:143-164 的 type 分发）新增
   `case RL_MSG_USER_DATA:` → 解析 `rl_user_payload_t`，打印一行，更新 `g_rl_rx_seq/g_rl_rx_count/g_rl_rx_len`；
   未知类型保持原样打印（原代码会把首字节当长度，注意别越界）。
3. **距离回传**：在 `measure_dis_proc_remote_iq()` 调用 `measure_dis_posalg_get_distance()` 成功后，
   组装 `rl_dist_payload_t`（滤波后/原始/置信度/RSSI/接收计数），调用**已存在但未使用**的
   `measure_dis_server_write_client(RL_MSG_DIST, &pkt, sizeof(pkt))`（原 server.c:340-360 → notify 封装 :319-338）。
4. **滤波/校准**：在 `sle_measure_dis_server_alg.c` 复制版中：
   - 距离生效校准：`slem_posalg_set_base_para()` 里的 `calib_val`（原 alg.c:32-33；**不要**改文档图指向的死函数 alg.c:61）；
   - `calib_val` 由 `CONFIG_RANGE_LINK_CALIB_MM/1000.0f` 得到（默认 0）；
   - 新增 `range_link_filter.c`：环形窗口 `CONFIG_RANGE_LINK_FILTER_WIN`（默认 8），先中值去野值（|raw−中值| > max(500mm, 30%) 丢弃），窗口内均值输出 `dist_mm`；
   - 打印改为单行（见 §5），原每帧多行 IQ 日志改为 `CONFIG_RANGE_LINK_VERBOSE` 才打印；
   - `osal_kmalloc` → `osal_kzalloc`；打印负值处理（先判 `dis < 0` 打 `-`）。

### 4.2 CLIENT 侧
1. **周期发用户数据**：在 client 任务/或由连接回调起一个 `osal_timer`/主循环里，每
   `CONFIG_RANGE_LINK_USER_PERIOD_MS`（默认 500ms）调用 `ssapc_write_cmd(0, conn_id, &param)`，
   `param.handle = g_property_handle`（发现到的属性句柄，原 client.c 已保存），payload=`rl_user_payload_t`，
   `param.type = 0`（沿用样例 IQ 写法）；seq 自增。
2. **接收距离回传**：`measure_dis_client_msg_proc()`（原 client.c:220-239）新增 `case RL_MSG_DIST:` →
   解析并打印单行（见 §5）。
3. **连接/扫描参数不变**：地址保持样例写死的 `1.1.1.1.1.1`（server）/`2.2.2.2.2.2`（client），两块板烧不同镜像即可。

## 5. 串口输出（115200 8N1，UART-L2；必须单行、可被脚本解析）

```
[RL] boot role=SERVER build=...            # 上电 banner
[RL] connected conn_id=%d
[RL] server D=%ldmm RAW=%ldmm PROB=%ld RSSI=%lddBm N=%lu RX_SEQ=%lu RX_CNT=%lu
[RL] server RX seq=%lu len=%u data=%s       # 收到用户数据（server 端）
[RL] client TX seq=%lu len=%u
[RL] client D=%ldmm PROB=%ld RSSI=%ld N=%lu # 收到 server 回传后（client 端）
[RL] disconnected, rescan/adv restart
```
打印节流：距离行最多 2 行/秒；用户数据行按接收速率（默认 2 包/秒）。`CONFIG_RANGE_LINK_VERBOSE=y` 才输出原样例的 IQ/中间日志。

## 6. Kconfig 选项（新 sample 目录）

```
choice "SLE Range Link role"
  config SAMPLE_SUPPORT_SLE_RANGE_LINK_SERVER   # default
  config SAMPLE_SUPPORT_SLE_RANGE_LINK_CLIENT
endchoice
config RANGE_LINK_USER_PERIOD_MS   int  default 500   range 50 10000
config RANGE_LINK_FILTER_WIN       int  default 8     range 1 32
config RANGE_LINK_CALIB_MM         int  default 0     range -5000 5000
config RANGE_LINK_VERBOSE          bool default n
```

## 7. 构建与验收

```powershell
$tc='D:\Project\NearLink\ToolChain'
$env:Path="$tc\tools\python;$tc\tools\python\Scripts;$tc\tools\Windows\ninja;$tc\tools\Windows\cc_riscv32_win_env;"+$env:Path
cd D:\Project\NearLink\SDK\bearpi-pico_h2821e
python build.py -c -j8 standard-bs21e-1100e
```

`.config` 需要（每版一次）：
`CONFIG_SAMPLE_ENABLE=y`、`CONFIG_ENABLE_PRODUCTS_SAMPLE=y`、`CONFIG_SAMPLE_SUPPORT_SLE_RANGE_LINK=y`、
角色二选一、`CONFIG_PRODUCT_SLE_MEASURE_DIS=y`、`# CONFIG_SUPPORT_SLE_BLE_CENTRAL_DEFAULT is not set`、
`# CONFIG_SAMPLE_SUPPORT_SLE_MEASURE_DIS is not set`。

验收标准：
1. SERVER / CLIENT 两版均 `python build.py -c -j8 standard-bs21e-1100e` 通过（0 error、0 warning 新增），产物 `application_sign.bin` + `tools\pkg\fwpkg\bs21e\bs21e_all.fwpkg` 生成，分别复制到 `Artifacts\range_link\server\`、`Artifacts\range_link\client\`；
2. 链接使用 `bs21e-1100e-sle-measure-dis/lib*.a`（搜 build.ninja/map），`application.mem` FLASH_PROGRAM < 550,000B（留安全余量）；
3. `nm` 两版符号不同（各自 role 函数存在），`measure_dis_server_alg` 只在 server 版；
4. 代码自检：无两个 app_run；`sle_range_link_entry` 唯一入口；无全局符号冲突；打印格式与 §5 一致；
5. 通过后由 Lead 安排硬件联调（两块板烧录 + 1m 校准 + 距离/数据验证）。

## 8. 校准与使用（写给最终用户，实现后落到 README）

1. 环境：锚点周围 3m 内空旷、无墙体/柱体/金属遮挡；板载天线同向竖直，离地约 1m。
2. 校准：把两块板放在 **1m** 距离，按 5 个方向各测 5 次，取平均距离减 1m = 校准值（mm）。
3. 把校准值写入 `CONFIG_RANGE_LINK_CALIB_MM`（menuconfig 或 .config），重新编译烧录。
4. 现场使用：server 串口看 `[RL] server D=...`；client 串口看 `[RL] client D=...` 与 `TX seq`；server 串口还能看到 `RX seq`（证明数据双向通）。
