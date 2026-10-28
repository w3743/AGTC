# regulator_driver 引脚对照表

稳压模块（sc-regulator V1.6-BUSFIX）· STM32G431CBU6 / WeAct Normal QFN48（UFQFPN48）

**这份表是 `Bsp/Inc/bsp_board.h` 的人读版本。改引脚时两个文件必须一起改。**

- 排针号：`J_CORE_P1` = 核心板 P1，`J_CORE_P2` = 核心板 P2，各 2×12 @ 2.54 mm
- 依据：`电力/稳压模块/原理图准备/接口与采样核对.md`（2026-10-22 接口映射纠正版）
- ADC/TIM 通道：ST 官方器件库 `STM32G431CBUx.xml`（UFQFPN48）逐脚核对
- 器件值：`电力/稳压模块/原理图准备/BOM.csv`

---

## 1. 功率级

| 排针 | MCU | 脚号 | 信号 | 复用 | 方向 | 去向 |
|---|---|---|---|---|---|---|
| P1.10 | PA8 | 30 | PWM_H | TIM1_CH1 (AF6) | 输出 | DRV8161 INH |
| P1.6 | PB13 | 26 | PWM_L | TIM1_CH1N (AF6) | 输出 | DRV8161 INL |
| P1.22 | PB7 | 45 | DUMP_PWM | TIM4_CH2 (AF2) | 输出 | UCC27511A IN+ |
| P1.21 | PB6 | 44 | PMOS_CTRL | GPIO 推挽输出 | 输出 | MMBT5551 基极 → Q_BUS_P 栅极 |
| P1.20 | PB5 | 19 | PERMIT | GPIO 推挽输出 | 输出 | 经 1 kΩ 接 active-low 的 DRV_FAULT_N |

## 2. 通信与诊断

| 排针 | MCU | 脚号 | 信号 | 复用 | 方向 | 说明 |
|---|---|---|---|---|---|---|
| P1.13 | PA11 | 33 | FDCAN1_RX | AF9 | 输入 | ⚠️ 同脚 = USB_DM |
| P1.14 | PA12 | 34 | FDCAN1_TX | AF9 | 输出 | ⚠️ 同脚 = USB_DP |
| P2.5 | PB10 | 22 | TEST | GPIO 推挽输出 | 输出 | 平时 LOW；诊断时短时拉高 |

## 3. ADC1 · 规则组（8 通道，连续扫描 + DMA 循环）

顺序 = 转换顺序 = `reg1[]` 下标，**不能乱**。

| idx | 排针 | MCU | 脚号 | 通道 | 信号 | 分压 / 换算 |
|---|---|---|---|---|---|---|
| 0 | P2.20 | PA0 | 8 | ADC1_IN1 | AD_T1 | 68k/10k/47k，K=9.2468 c=1.4468 |
| 1 | P2.17 | PA1 | 9 | ADC1_IN2 | AD_T2 | 同上 |
| 2 | P2.18 | PA2 | 10 | ADC1_IN3 | AD_T3 | 同上 |
| 3 | P2.15 | PA3 | 11 | ADC1_IN4 | AD_T4 | 同上 |
| 4 | P1.7 | PB14 | 27 | ADC1_IN5 | SC_SUM / AD_T8 | 68k/10k，×7.8 |
| 5 | P1.5 | PB12 | 25 | ADC1_IN11 | NTC_DUMP | 10k 上拉 + B3950 |
| 6 | P2.10 | PB1 | 18 | ADC1_IN12 | DRV_SO | 10 mV/A，零点 1.65 V |
| 7 | P2.9 | PB0 | 17 | ADC1_IN15 | BUS_ADC | 150k/10k，×16 |

## 4. ADC2 · 规则组（6 通道）

| idx | 排针 | MCU | 脚号 | 通道 | 信号 | 分压 / 换算 |
|---|---|---|---|---|---|---|
| 0 | P2.16 | PA4 | 12 | ADC2_IN17 | AD_T5 | 68k/10k/47k |
| 1 | P2.13 | PA5 | 13 | ADC2_IN13 | AD_T6 | 同上 |
| 2 | P2.14 | PA6 | 14 | ADC2_IN3 | AD_T7 | 同上 |
| 3 | P2.6 | PB11 | 24 | ADC2_IN14 | NTC_MAIN | 10k 上拉 + B3950 |
| 4 | P2.12 | PC4 | 16 | ADC2_IN5 | NTC_SC | 10k 上拉 + B3950 |
| 5 | P1.8 | PB15 | 28 | ADC2_IN15 | TEST_FB | 10k/10k，×2 |

## 5. 只进比较器、不进 ADC 的脚

| 排针 | MCU | 脚号 | 信号 | 用途 |
|---|---|---|---|---|
| P2.11 | PA7 | 15 | DRV_SO 第二分支 | 只接 COMP2_INP（该脚另有 ADC2_IN4 能力，本方案不用） |

配成模拟输入只是为了不浮空、不拾噪。**不参与任何 ADC 序列。**

---

## 6. 为什么超容抽头被劈成两半

`T1–T4` 在 `PA0–PA3`（只有 ADC1 有这些通道），`T5–T7` 在 `PA4–PA6`（只有 ADC2 有）。
这不是设计选择，是原理图布线决定的 —— G431 没有把 8 个抽头都送到同一个 ADC。
所以本工程必须同时驱动 ADC1 与 ADC2，缺一个就少一片采样。

## 7. 五个复用陷阱（全部已在代码里规避，改板前先读）

1. **PA11/PA12 = FDCAN1_RX/TX + USB_DM/DP**
   核心板把这两个脚同时引到 Type-C。接 CAN 时不能同时插 USB 数据线。
   器件库原文：`PA11: FDCAN1_RX, USB_DM`；`PA12: FDCAN1_TX, USB_DP`。

2. **PB6 = PMOS_CTRL，但核心板 SB3 接 PA9、SB6 接 CC1**
   必须先用万用表确认两个焊桥**实物开路**，否则 PB6 会顺带驱动 PA9 和 Type-C 的 CC1。

3. **PB12 = NTC_DUMP(ADC1_IN11) + TIM1_BKIN**
   本板 PB12 只做模拟输入 → **TIM1 的 BreakState 必须保持 DISABLE**。
   真开了，NTC 分压上的低电平会被当成刹车，半桥反复掉输出。

4. **PA6 = AD_T7(ADC2_IN3) + TIM1_BKIN**
   同上，绝不要把 TIM1 的 BKIN 源配到 PA6。

5. **P2.7 = PB2 是核心板自己的 VBUS 检测**（经 SB4 + 100k/10k/10nF）
   不要复用、不要改板。P1.19 = PB4 常被误认成 VBUS 检测，它不是。

## 8. 封装硬限制

UFQFPN48 上**不存在**：`PC0` `PC1` `PC2` `PC3` `PC5` `PC12`。
原理图核对时若看到这些脚名，说明看错了封装列（DS12589 有 5 种封装，只有 UFQFPN48 那列算数）。

## 9. 保留脚（本工程不配置、不驱动）

`PC6` `PA9` `PA10` `PA15` `PC10` `PC11` `PB3` `PB4` `PB8-BOOT0` `PB9` `PC13` `PC14` `PC15`

⚠️ `PB8-BOOT0` 尤其不能动：它是 BOOT0，拉高会让芯片从 System Memory 启动而不是 Flash。

---

## 10. 与 foc_driver 的差异

两块板**共用同一颗 G431 核心板**，但插在不同的功率板上，固件各自独立。

| 信号 | 稳压板 | FOC 板 |
|---|---|---|
| PA8 | PWM_H（TIM1_CH1） | PWM_UH（TIM1_CH1） |
| PB13 | PWM_L（TIM1_CH1N） | PWM_UL（TIM1_CH1N） |
| PB5 | PERMIT（输出） | CAN_STB（输出） |
| PB6 | PMOS_CTRL（输出） | 未用 |
| PB7 | DUMP_PWM（TIM4_CH2） | 未用 |
| PB12 | NTC_DUMP（ADC1_IN11） | BREAK_N（TIM1_BKIN） |
| PB11 | NTC_MAIN（ADC2_IN14） | DRV_ENABLE（输出） |
| PB0 | BUS_ADC（ADC1_IN15） | SAFE_ADC（ADC1_IN15） |
| PC4 | NTC_SC（ADC2_IN5） | BUS_ADC（ADC2_IN5） |
| PA4 | T5（ADC2_IN17） | DRV_SPI_CS_N（GPIO） |
| PA5 | T6（ADC2_IN13） | SPI1_SCK |
| PA6 | T7（ADC2_IN3） | SPI1_MISO |
| PA7 | COMP2_INP | SPI1_MOSI |
| PB1 | DRV_SO（ADC1_IN12） | TEMP_MOT_ADC（ADC1_IN12） |
| PB14 | SC_SUM / T8（ADC1_IN5） | PWM_VL（TIM1_CH2N） |
| PB15 | TEST_FB（ADC2_IN15） | PWM_WL（TIM1_CH3N） |
| PB10 | TEST（输出） | HALL_C（TIM2_CH3） |

**只有 PA8 / PB13 / PA11 / PA12 四个脚在两块板上同功能**，其余全部不同。
这就是为什么两块板必须各有各的固件，不能合并成一个工程。

---

## 11. 三层结构与调用链

两个工程用的是同一套骨架。对照着看能省很多时间 —— 模块名、命名规则、边界规则完全一致。

```
Core/Src/main.c           只做三件事：HAL 初始化 → bsp_init/dl_init/bl_init → 主循环按 tick 调 BL
        │
        ├─ BusinessLayer  （决策）
        │     bl.c          编排：初始化顺序、1 ms 节拍转发、100 ms CAN 状态上报
        │     bl_regulator.c 5 状态机（INIT/IDLE/ARMED/ACTIVE/FAULT）、10 个故障位、前馈占空比
        │
        ├─ DataLayer      （换算）
        │     dl.c          总入口
        │     dl_bus.c      母线 ×16 / 超容 ×7.8
        │     dl_cell.c     8 节抽头、差分、TEST_FB 反推
        │     dl_temp.c     3 路 NTC（B 值公式 + 开路/短路哨兵）
        │     dl_power.c    泄放功率估算（外置电阻阻值必须由调用者给）
        │     dl_can.c      接收交接槽 + 状态帧打包
        │
        └─ Bsp            （原语）
              bsp.c / bsp_board.h / 6 个子模块（Bridge/Dump/Gpio/Adc/Can/Callback）
```

| 层 | 可以做什么 | 不可以做什么 |
|---|---|---|
| BSP | 读写寄存器、配外设、给原语 | 不做换算、不做决策、不知道什么是 PI |
| DL | 调 BSP 取数，换算成物理单位 | 不直接碰寄存器、不做控制决策 |
| BL | 读 DL 的工程量，调 BSP/DL 的执行原语 | 不出现分压比/上拉阻值/CSA 增益这类数字 |

**命名规则**：`bsp_xxx_yyy` / `dl_xxx_yyy` / `bl_xxx_yyy`；静态函数前置下划线（`_bsp_xxx_yyy`）；
引脚宏只在 `bsp_board.h` 定义一次。

### 两个工程的对照表

| 关注点 | regulator_driver | foc_driver |
|---|---|---|
| 功率级开关 | `bsp_power_stage_enable()`：中点 → 合 P-MOS → PERMIT → MOE | `bsp_power_stage_enable()`：查故障 → 零矢量 → MOE → ENABLE |
| 断电为什么要等 | **必须等电感电流衰减**，电感能量无处可去 | **不用等**，绕组电流经体二极管续流回母线 |
| 调制 / 控制 | 前馈占空比 `D = V_SC / V_BUS` | SVPWM + 可选电流环 |
| 闭环 | 未做（「环路稳定性待联合定型」） | 未做（L/R 未实测，增益默认 0 = 旁路） |
| 采样缺口 | SC− 没有采样通道，第 1 节电压需要外部提供 `sc_minus_mv` | 角度只有 60° 分辨率（霍尔） |
| CAN 协议 | 未做，ID 全是占位符 | 未做，ID 全是占位符 |
| 上板自测 | 需手动调 `bl_regulator_request_enable()` | `main.c` 的 `BL_DEMO_ENABLE` 改 1 即自动跑一轮 |

### 两个工程共同的坑

`bl_task_handler()` **不能放进 SysTick_Handler()**。
它最终会走到 `bsp_power_stage_enable()`，那个函数用 `HAL_Delay()` 轮询硬件状态，
而 `HAL_Delay()` 依赖 SysTick 递增 —— 放进 SysTick 里会等一个永远不会推进的计数器，直接死锁。
两个工程的 `main.c` 都是"主循环里按 `HAL_GetTick()` 对齐 1 ms 调用"。


---

## 12. `.ioc` 与 CubeMX 生成通道

本工程的 `regulator_driver.ioc` 是新建的（此前没有），已验证能被 CubeMX 命令行生成：

```bash
cd "E:/system/Desktop/新建文件夹"
cmd //c "STM32CubeMX.exe -q D:/path/to/s.txt"
```

`s.txt` 三行：`config load <正斜杠路径>` / `project generate` / `exit`。
实测 22 秒出 `CMakeLists.txt`、`CMakePresets.json`、`Core/`、`Drivers/`、
`cmake/`、`STM32G431xx_FLASH.ld`、`startup_stm32g431xx.s`。

### 12.1 生成结果

本次生成实际产出 `Core/Src`：`gpio.c`、`main.c`、`stm32g4xx_hal_msp.c`、
`stm32g4xx_it.c`、`syscalls.c`、`sysmem.c`、`system_stm32g4xx.c`、`tim.c`。

`.ioc` 里被 CubeMX 认可的 IP：`NVIC`、`RCC`、`SYS`、`TIM1`、`TIM4`。

### 12.2 被静默丢弃的部分

| 项目 | 现象 | 为什么不影响**引脚与功能设计** |
|---|---|---|
| `ADC1` / `ADC2` | 引脚信号（14 个 `ADCx_INy`）保留，但 IP 被丢，不生成 `adc.c` | `bsp_adc.c` 自己 `HAL_ADC_Init` + `HAL_DMA_Init` + `HAL_ADC_ConfigChannel`，从不调用 `MX_ADCx_Init` |
| `FDCAN1` | `PA11/PA12` 信号保留，IP 被丢 | `bsp_can.c` 自己配 FDCAN 与过滤器 |
| `TIM1` 通道 / 死区 / Break | 只生成时基 | `bsp_bridge.c` 完整重配 |

⚠️ 但**生成物本身会受影响**：CubeMX 会同时裁掉 `stm32g4xx_hal_conf.h` 里的模块宏
和 `cmake/stm32cubemx/CMakeLists.txt` 里对应的 HAL 源文件，导致重新生成后编译失败。
兜底办法见 12.4。

CubeMX 丢外设时**不报错**，只在日志留 `IP not ready for code generation`。
本工程的 `.ioc` 是在 `foc_driver.ioc` 基础上按 `bsp_board.h` 重排引脚得到的，
两者的 `.ioc` 格式坑完全一致，细节见 `foc_driver/BSP引脚对照.md` 第 6 节，要点：

1. `S_TIM1_CH1` / `S_TIM4_CH2` 这类共享信号必须配 `SH.xxx.0` + `SH.xxx.ConfNb=1`；
2. 定时器必须有 `VP_TIMx_VS_ClockSourceINT` 虚拟引脚并进 `Mcu.PinN` 列表；
3. 互补输出的通道选择器名字带 `N`：`PWM Generation1 CH1 CH1N`；
4. G4 的周期/比较值键名是 `PeriodNoDither` / `PulseNoDither_N`；
5. `DeadTime` 是 DTG 原值（200 ns → 34），不是纳秒；
6. `MxDb.Version=DB.6.0.181` + `AskForMigrate=false`，否则迁移对话框把 script 模式卡死。

### 12.3 一句话定位

`.ioc` 负责**引脚与时钟的单一来源 + 可重复生成工程骨架**；外设配置一律在 BSP。
想补齐 ADC/FDCAN/TIM 通道，用 CubeMX 图形界面打开 `regulator_driver.ioc`，
在 Pinout 里勾选对应外设模式再生成即可 —— CubeMX 会自己写回正确的
`SH` / `VP` / `CH1N` 形式。

### 12.4 重新生成前必须知道的两件事

1. **生成位置**：CubeMX 把产物写到 `.ioc` 所在目录。直接对
   `D:/Hal_code/regulator_driver/regulator_driver.ioc` 执行 `project generate`
   会覆盖本工程的 `Core/`。要试生成，先把 `.ioc` 复制到临时目录。

2. **两处会被按 `.ioc` 裁剪**（已在顶层 CMakeLists 兜底）：
   - `Core/Inc/stm32g4xx_hal_conf.h`：ADC/FDCAN 的 `HAL_*_MODULE_ENABLED` 会被注释；
   - `cmake/stm32cubemx/CMakeLists.txt`：`hal_adc.c` / `hal_adc_ex.c` / `hal_fdcan.c`
     会从 `STM32_Drivers_Src` 里消失。

   顶层 `CMakeLists.txt` 头部写明 `generated only once`，CubeMX 不覆盖它，
   因此在 `add_subdirectory` 之前加了：

   ```cmake
   add_compile_definitions(HAL_ADC_MODULE_ENABLED HAL_FDCAN_MODULE_ENABLED)
   ```

   ⚠️ 这个兜底只解决**宏**。如果某次重新生成后 HAL 源文件列表被裁掉了，
   还要按 `foc_driver/CMakeLists.txt` 的写法，在 `add_subdirectory` 之后补：

   ```cmake
   target_sources(STM32_Drivers PRIVATE
       ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/STM32G4xx_HAL_Driver/Src/stm32g4xx_hal_adc.c
       ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/STM32G4xx_HAL_Driver/Src/stm32g4xx_hal_adc_ex.c
       ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/STM32G4xx_HAL_Driver/Src/stm32g4xx_hal_fdcan.c)
   ```

   只补宏不补源的典型症状是：编译通过，链接报
   `dangerous relocation: unsupported relocation`。

### 12.5 根因已定位并修复（2026-10-28 晚）

之前"ADC/FDCAN 在 .ioc 里无法实例化、根因未定位"的结论**已作废**。

根因与 foc 工程的 SPI1 完全同类（详见 `foc_driver/BSP引脚对照.md` §6.8）：
**ADC1/ADC2/FDCAN1 既没登记进 `Mcu.IP*` 列表，也没有任何参数块**——
`.ioc` 里的 `PA0.Signal=ADC1_IN1`、`PA11.Signal=FDCAN1_RX` 全是悬空引用，
CubeMX 加载时静默丢弃 IP、只留引脚信号名。

修复内容（`regulator_driver.ioc`）：

```
Mcu.IP 列表加入 ADC1 / ADC2 / FDCAN1（IPNb 5→8）
ADC1/ADC2/FDCAN1 参数块照 bsp_adc.c / bsp_can.c 补齐：
  ADC1 = 8 通道（IN1/2/3/4/5/11/12/15，247.5 周期，连续+扫描）
  ADC2 = 6 通道（IN17/13/3/14/5/15）
  FDCAN = 经典帧 500k（20 分频，Seg1=14，Seg2=2，SJW=2，自动重传）
ADC/FDCAN 引脚补 Mode 行（INx-Single-Ended / FDCAN_Activate）
NVIC.FDCAN1_IT0_IRQn=false 显式声明（中断归 bsp_can.c，优先级 5,0）
```

**PA7 的处理**：`.ioc` 原来写着 `PA7.Signal=ADC2_IN4`，但 BSP 注释明确
PA7 只接 COMP2_INP、不进任何 ADC 序列。已把 PA7 从 `.ioc` 移除——
无 IP 引脚的默认态就是模拟输入，与 BSP 的 GPIO_MODE_ANALOG 配置等价。
代价是 Pinout 视图里 PA7 显示为空闲脚，引脚归属以本文档和 bsp_board.h 为准。

**⚠️ KeepUserCode 吞尾括号坑（实测踩中）**：原地重新生成时，CubeMX 对
`main.c` 的 USER CODE WHILE 段做文本手术，把 `main()` 的收尾 `}` 吞掉，
报 `expected declaration or statement at end of input`。诱因是原来的
while 区域外面还包了一层 `{ ... }` 块（`U32 last_tick` 的作用域块）。
已把该区域重构为标准形态（`while(1)` 直接跟在 `BEGIN WHILE` 后，与
foc 工程一致，语义不变）。**今后再原地生成，若 main.c 编译报尾括号
错误，对照 `_fix/backup/reg_main.c.bak` 检查这一段。**

验证结果：

| 检查项 | 结果 |
|---|---|
| `adc.c` 生成，ADC1 8 通道 / ADC2 6 通道，顺序与 bsp_adc.c 一致 | ✅ |
| `fdcan.c` 生成，位时序 500k，无 NVIC 使能（归 BSP） | ✅ |
| `HAL_ADC/FDCAN_MODULE_ENABLED` 激活，hal 源码进 cmake 列表 | ✅ |
| USER CODE 2 的 bsp_init/dl_init/bl_init 保留 | ✅ |

CMakeLists 调整与 foc 相同：`add_compile_definitions` 已撤销（会撞
`macro redefined`），另加防御性 `target_sources`（adc/adc_ex/fdcan）。

代价：FLASH 27540 → ~31700 B (24.22%)。`MX_ADC1/2_Init`、
`MX_FDCAN1_Init` 与 BSP 初始化并存，后者完整重配（含 DMA/DMAMUX，
`.ioc` 里刻意不配 DMA，与 foc 工程口径一致）。
