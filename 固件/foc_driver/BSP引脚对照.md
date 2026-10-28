# FOC 驱动模块 BSP 引脚对照与上板验证清单

- 工程：`D:/Hal_code/foc_driver`
- 芯片：STM32G431CBU6（WeAct Normal QFN48 核心板）
- 数据来源：`D:/PCB/FOC驱动模块/原理图准备/当前原理图连接.json`（169 器件 / 506 脚 / 102 网络，2026-10-21 快照）
- 脚序依据：`D:/PCB/mcu_pins.json`（WeAct 官方原理图 WeAct_normal_QFN48_V1.0.pdf，SHA256 `57ECA2E3…C640D`）
- MCU 复用功能依据：STM32G431 数据手册 DS12589 Rev6 Table 12
- 日期：2026-10-28

---

## 0. 为什么是三块板而不是一块

一个 G431 装不下。数一下就清楚：

| 资源 | FOC 板需要 | 水泵板需要 | 稳压板需要 |
|---|---|---|---|
| 高级定时器（互补 PWM） | TIM1 六路 + BKIN | 1 路 PWM | — |
| ADC 通道 | 3 相电流 + 母线 + 2 温度 + 安全 | 电流 + 母线 + NTC | 多路电压 |
| SPI | 1（DRV8353 配置） | — | — |
| CAN | 1 | 1 | 1 |
| 霍尔 | 3 | — | — |

水泵板的 `.ioc` 里 TIM1 只用了 CH1/CH1N，ADC 只开了 2 个通道，根本没有 SPI —— 那是按有刷电机 + DRV8262 配的。把 FOC 的六路互补 PWM、三路同步采样、SPI 配置总线塞进去，引脚会互相抢（PA8/PA9/PA10 在水泵板上已经被占用）。

所以电力系统本来就是**四颗 G431 走 CAN 组网**：三块功率板 + 一块主控，各管各的功率级，只通过 CAN 交换指令和状态。本工程 = 其中的 FOC 驱动板。

---

## 1. 引脚总表（27 个信号脚）

`J_MCU_L` = 核心板 P1，`J_MCU_R` = 核心板 P2。

### 1.1 功率级（bsp_inverter）

| 排针 | 网络 | MCU 脚 | 复用 | 去向 | 备注 |
|---|---|---|---|---|---|
| L.10 | PWM_UH | **PA8** | TIM1_CH1 (AF6) | DRV8353 第 34 脚 INHA | 上桥 U |
| L.11 | PWM_VH | **PA9** | TIM1_CH2 (AF6) | 第 36 脚 INHB | 上桥 V |
| L.12 | PWM_WH | **PA10** | TIM1_CH3 (AF6) | 第 38 脚 INHC | 上桥 W |
| L.6 | PWM_UL | **PB13** | TIM1_CH1N (AF6) | 第 35 脚 INLA | 下桥 U |
| L.7 | PWM_VL | **PB14** | TIM1_CH2N (AF6) | 第 37 脚 INLB | 下桥 V |
| L.8 | PWM_WL | **PB15** | TIM1_CH3N (AF6) | 第 39 脚 INLC | 下桥 W |
| L.5 | BREAK_N | **PB12** | TIM1_BKIN (AF6) | 第 28 脚 nFAULT ⊕ U3 比较器 | 线与，R_BREAK_PU 2.2k 上拉 |
| R.6 | DRV_ENABLE | **PB11** | GPIO 输出 | 第 33 脚 ENABLE | R_DRV_EN_PD 10k 下拉，复位即禁用 |

- 时钟：TIM1 中心对齐，ARR = 4250，170 MHz / (2×4250) = **20 kHz**
- 死区：DTG 编码值 35 → 35 × 5.882 ns ≈ **206 ns**（目标 200 ns，向上取整）
- ADC 触发：TIM1 CH4（OC4REF，PWM2 模式，CCR4 = ARR−8）→ TRGO，只在计数器顶点打一拍

### 1.2 电流与电压采样（bsp_adc）

| 排针 | 网络 | MCU 脚 | 通道 | 组别 | 说明 |
|---|---|---|---|---|---|
| R.17 | SO_U_ADC | **PA1** | ADC1_IN2 | 注入 rank1 | 0.5 mΩ 低边采样，CSA 20 V/V |
| R.15 | SO_V_ADC | **PA3** | ADC1_IN4 | 注入 rank2 | 同上 |
| R.20 | SO_W_ADC | **PA0** | ADC1_IN1 | 注入 rank3 | 同上 |
| R.18 | TEMP_MOS_ADC | **PA2** | ADC1_IN3 | 规则 rank1 | NTC 分压 |
| R.10 | TEMP_MOTOR_ADC | **PB1** | ADC1_IN12 | 规则 rank2 | NTC 分压 |
| R.9 | SAFE_ADC | **PB0** | ADC1_IN15 | 规则 rank3 | 安全阈值监测 |
| R.12 | BUS_ADC | **PC4** | ADC2_IN5 | 规则 rank1 | 180k/10k 分压 |
| R.8 | VREF_CORE | **VREF+** | — | — | 同时接 DRV8353 第 26 脚 VREF |

**⚠️ PC4 只有 ADC2 通道**（ADC1 没有映射），所以母线电压必须由 ADC2 采，这也是本层要同时驱动 ADC1 和 ADC2 的唯一原因。

**⚠️ ADC 满量程 = VREF_CORE，不是想当然的 3.3 V。** 板上没有任何器件主动产生这个电压，它依赖核心板把 P2-8 内部接到 3V3。上电第一件事就是用万用表量 P2-8 对 GND；若不是 3300 mV，改 `bsp_adc.h` 里的 `BSP_ADC_VREF_MV`，否则所有电流/电压换算按比例全错。

### 1.3 DRV8353 配置总线（bsp_spi）

| 排针 | 网络 | MCU 脚 | 复用 | DRV8353 脚 |
|---|---|---|---|---|
| R.16 | DRV_SPI_CS_N | **PA4** | 普通 GPIO（软件片选） | 32 SCS |
| R.13 | DRV_SPI_SCK | **PA5** | SPI1_SCK (AF5) | 31 SCLK |
| R.14 | DRV_SPI_MISO | **PA6** | SPI1_MISO (AF5) | 29 SDO（**开漏**，R_SPI_SDO_PU 4.7k 上拉） |
| R.11 | DRV_SPI_MOSI | **PA7** | SPI1_MOSI (AF5) | 30 SDI |

- 模式 1（CPOL=0 / CPHA=2EDGE），16 位帧，MSB 先行，BAUDRATEPRESCALER_32 → 170/32 ≈ **5.31 MHz**
- 帧间隔：nSCS 拉高后空转 100 个 NOP ≈ 1.8 µs（手册要求 ≥ 400 ns）

### 1.4 霍尔（bsp_hall）

| 排针 | 网络 | MCU 脚 | 复用 | 说明 |
|---|---|---|---|---|
| L.15 | HALL_A | **PA15** | TIM2_CH1 (AF1) | 本层当普通输入读，R_HALL_PU_A 4.7k 在 DIRECT 支路 |
| L.18 | HALL_B | **PB3** | TIM2_CH2 (AF1) | 同上 |
| R.5 | HALL_C | **PB10** | TIM2_CH3 (AF1) | 同上 |

- 三脚正好凑成 TIM2 的 CH1/2/3，将来要测速可直接切「霍尔接口模式」（SMS=100），本层不用改
- **绝不开内部上拉**：跳线打 5V_DIV 时内部上拉会把分压点顶高
- PA15/PB3 复位后是 JTDI/JTDO，配成普通输入后 JTAG 让位；SWD（PA13/PA14）不受影响

### 1.5 其他 GPIO 与总线

| 排针 | 网络 | MCU 脚 | 方向 | 说明 |
|---|---|---|---|---|
| L.20 | CAN_STB | **PB5** | 输出 | TCAN1042 STB：低=正常，高=待机。R_CAN_STB_PU 10k 上拉 |
| L.24 | DUMP_THERM_OK_GPIO | **PB9** | 输入 | 泄放电阻热许可，U4 比较器经 R_THERM_SER 100k 推挽驱动 |
| L.13 | FDCAN1_RX | **PA11** | AF9 | TCAN1042 RXD（推挽输出） |
| L.14 | FDCAN1_TX | **PA12** | AF9 | TCAN1042 TXD |

---

## 2. 与水泵板的关键差异：PB8 → PA11/PA12

水泵板把 FDCAN1_RX 放在了 **PB8**（P1-23 = `PB8_BOOT0`）。这是一颗雷：

- TCAN1042 的 **RXD 是推挽输出**，总线开路时输出高电平（手册 Table 9-5 "Open → RXD H"），待机态也是高
- PB8 在 G431 上同时是 **BOOT0**。复位期间 MCU 采样 BOOT0，如果 RXD 正把它顶高，芯片会进 System Memory 而不是 Flash —— 表现为"上电不跑程序"，而且换块板子有时好有时坏（取决于收发器上电时刻）

FOC 板的原理图把 P1-23 空着，CAN 走 **PA11/PA12**（同一组 FDCAN1，AF9），从根上绕开。

**代价**：WeAct 核心板上 P1-13/P1-14 同时是 USB-C 的 D−/D+。用了 FDCAN 就用不了 USB CDC。本工程走 SWD（ST-LINK）下载调试，不需要 USB，所以这个代价值得付。

---

### 2.1 三个必须知道的引脚复用冲突

这些都是同一根脚挂了两三个外设，配错了不会报错、只会"功能莫名失效"。

| 引脚 | 复用 | 本板用法 | 冲突后果 |
|---|---|---|---|
| **PA11** | FDCAN1_RX(AF9) / **TIM1_CH4**(AF6) / **TIM1_BKIN2**(AF6) | FDCAN1_RX | ① TIM1_CH4 到不了引脚 —— 但这正是我们要的，ADC 触发只用片内 OC4REF→TRGO；② 如果打开了 `Break2State`，CAN 总线上的每一帧都可能被当成刹车输入把 MOE 打掉 |
| **PA6** | SPI1_MISO(AF5) / **TIM1_BKIN**(AF6) | SPI1_MISO | 把 PA6 改成 AF6 想多接一路刹车，SPI 立刻收不到数据，现象像"DRV8353 坏了" |
| **PB8** | **BOOT0** / FDCAN1_RX | **不用** | 见上一节。本板原理图把 P1-23 留空，这是对的 |

对应的代码约束：
- `bsp_inverter.c` 里 `bdt_init.Break2State = TIM_BREAK2_DISABLE` 不能改
- `bsp_spi.c` 里 PA6 必须保持 `GPIO_AF5_SPI1`
- `bsp_can.c` 里 PA11/PA12 必须保持 `GPIO_AF9_FDCAN1`

---

## 3. 上板验证清单（按顺序做，别跳）

代码通过了编译，但这只证明语法和类型对，**不证明硬件能用**。下面每一条都是一个可独立证伪的检查。

| # | 检查项 | 怎么测 | 通过标准 | 不通过说明什么 |
|---|---|---|---|---|
| 1 | VREF_CORE | 万用表量 P2-8 对 GND | 3.30 V ± 0.05 V | 核心板没把 VREF+ 接到 3V3，所有电流换算按比例错 |
| 2 | DVDD | 量 U1 第 40 脚 | 5.0 V ± 0.25 V | 外部 5V 没进 DRV8353，SPI 和栅极驱动全死 |
| 3 | 5V 电源 | 量 L.1/L.2 | 5.0 V 左右 | 前级 buck 没工作 |
| 4 | DRV8353 通信 | 上电后看 `bsp_init()` 返回值 | `TYPE_RET_SUCCESS` | 返回 FAIL = SPI 不通或芯片没焊。用逻辑分析仪看 PA4/PA5/PA6 |
| 5 | TIM1 六路波形 | 示波器量 6 个 PWM 脚，MOE 未开 | 全部为**低** | 有高电平说明 OSSI/OISx 配错 |
| 6 | PWM 频率 | `bsp_inverter_start()` 后量 PA8 | 20 kHz ± 1%，占空比 50% | 频率不对查 ARR / 时钟树 |
| 7 | 死区 | 量 PA8 与 PB13 的交叠 | 约 200 ns 无交叠 | 交叠 = 直通 = 炸管 |
| 8 | ADC 触发 | `bsp_adc_get_phase_raw()` 连续读，电机静止 | 三相都在 2048 附近（±50） | 差很多 = 零点偏；完全不变 = 触发没到 |
| 9 | 电流零点 | 功率级关闭时读三相 | 2047~2049 | 偏得多需要在 DL 层做零点标定 |
| 10 | 霍尔 | 手转电机，看 `bsp_hall_read()` | 6 个合法码值循环，不出现 0 或 7 | 出现 0/7 = 接线错或跳线选错 |
| 11 | CAN | `bsp_can_start()` 后接 CAN 分析仪 | 500 kbit/s 能收到帧 | 收不到先查 PA11/PA12 是否被 USB 器件拉住 |
| 12 | 硬件刹车 | 手动把 BREAK_N 对地短一下 | `bsp_inverter_break_occurred()` 返回 TRUE，MOE 掉 | 没反应 = BKIN 没配或滤太狠 |

### 顺序上的硬约束

- **第 5 项必须在第 6 项之前做**。MOE 一开，六路就开始按占空比输出；如果 off-state 电平配错了，上电瞬间就是直通。
- **第 8 项必须在第 12 项之前做**。刹车一旦触发会锁死 MOE，得先 `bsp_inverter_clear_break()` 才能继续。

---

## 4. 三层结构与已做 / 未做

### 4.1 分层与文件对照

```
Core/Src/main.c           只做三件事：HAL 初始化 → bsp_init/dl_init/bl_init → 主循环按 tick 调 BL
        │
        ├─ BusinessLayer  （决策：该不该开、开多大、出事了怎么办）
        │     bl.c          编排：初始化顺序、1 ms 节拍转发、100 ms CAN 状态上报
        │     bl_motor.c    状态机（IDLE/ARMED/RUN/FAULT）、9 个故障位、电压斜坡、软停
        │     bl_foc.c      调制：Clarke / Park / 反Park / SVPWM + 霍尔扇区角
        │
        ├─ DataLayer      （换算：原始码 → 工程量）
        │     dl.c          总入口
        │     dl_motor.c    功率级状态、占空比下发与钳位、COAST/BRAKE 语义
        │     dl_sense.c    三相电流 / 母线电压 / 两路 NTC
        │     dl_can.c      接收交接槽 + 状态帧打包
        │
        └─ Bsp            （原语：配引脚、配外设、给安全可用的接口）
              bsp.c / bsp_board.h / 7 个子模块（见下表）
```

**边界规则**（改代码前先看这三条）

| 层 | 可以做什么 | 不可以做什么 |
|---|---|---|
| BSP | 读写寄存器、配外设、给原语 | 不做换算、不做决策、不知道什么是电流环 |
| DL | 调 BSP 取数，换算成物理单位 | 不直接碰寄存器、不做控制决策 |
| BL | 读 DL 的工程量，调 BSP/DL 的执行原语 | 不出现分压比/采样电阻/CSA 增益这类数字 |

**命名规则**：`bsp_xxx_yyy` / `dl_xxx_yyy` / `bl_xxx_yyy`；静态函数前置下划线（`_bsp_xxx_yyy`）；
引脚宏只在 `bsp_board.h` 定义一次，子模块里最多写短别名。

**关键调用链**（自测时按这条链排查最快）

```
TIM1 计数器顶点 ──TRGO──> ADC1 注入组 ──(硬件锁存)──> bsp_adc_get_phase_raw()
                                                            │
bl_task_handler() ──1ms──> bl_motor_task_1ms()               │
   ├─ dl_sense_update() ─────────────────────────────────────┘
   ├─ _bl_motor_eval_interlocks()   母线/电流/温度/霍尔/DRV
   ├─ _bl_motor_task_active()  ──> bl_foc_run()
   │                                  ├─ Clarke（Iu,Iv → αβ）
   │                                  ├─ bl_foc_get_angle_rad()（霍尔扇区中心）
   │                                  ├─ Park（αβ → dq）
   │                                  ├─ [电流环，默认旁路]
   │                                  ├─ 反 Park（dq → αβ）
   │                                  ├─ bl_foc_svpwm()（零序注入）
   │                                  └─ dl_motor_set_duty() ──> bsp_inverter_set_duty()
   └─ 故障时 _bl_motor_enter_fault() ──> dl_motor_enable(FALSE)（COAST）
```

### 4.2 已做

| 文件 | 内容 |
|---|---|
| `Bsp/bsp.c` / `bsp.h` | 总入口：安全时序初始化、功率级使能、DRV8353 探针 |
| `Bsp/Inc/bsp_board.h` | **全工程唯一的引脚 / 复用号 / ADC 通道定义处** |
| `Bsp/Src/Gpio/bsp_gpio.c` | DRV_ENABLE / CAN_STB / THERM_OK / BREAK_N |
| `Bsp/Src/Inverter/bsp_inverter.c` | TIM1 20 kHz 中心对齐六路互补 + 死区 + 刹车 + CH4 触发 |
| `Bsp/Src/Adc/bsp_adc.c` | ADC1 注入（相电流）+ 规则（温度/安全）+ ADC2（母线）+ 双 DMA |
| `Bsp/Src/Hall/bsp_hall.c` | 三路霍尔原始码读取 |
| `Bsp/Src/Spi/bsp_spi.c` | SPI1 16 位模式 1 + DRV8353 寄存器读写 |
| `Bsp/Src/Can/bsp_can.c` | FDCAN1 500 kbit/s + 内核时钟源 + NVIC + 收发 |
| `Bsp/Src/Callback/bsp_callback.c` | 回调注册与分发 |
| `DataLayer/Src/dl_*.c` | 功率级状态、占空比钳位、电流/母线/温度换算、CAN 交接槽与状态帧 |
| `BusinessLayer/Src/bl_*.c` | 状态机、9 个故障位、电压斜坡、软停、Clarke/Park/SVPWM、霍尔扇区角 |

**编译结果**：0 错误 0 告警，FLASH 43152 B / 128 KB（32.92%），RAM 2888 B / 32 KB（8.81%）。
（FLASH 从 28144 B 涨到 43152 B，主要是 `sinf`/`cosf`/`sqrtf`/`logf` 从 newlib-nano 拉进来，
以及 SVPWM 与状态机的代码。要省空间可把三角函数换 256 点查表。）

### 4.3 未做（有依据，不是漏掉）

| 缺口 | 为什么不做 | 补的时候动哪里 |
|---|---|---|
| 电流环 PI 增益 | 相电感 L、相电阻 R 未实测，编系数上板不是振荡就是炸管 | `bl_foc_set_current_gains()`，环路代码已写好，增益非 0 即生效 |
| 角度细分（60° → 更细） | 霍尔只能给 60° 分辨率，插值需要扇区驻留时间 | 只换 `bl_foc_get_angle_rad()`，Clarke/Park/SVPWM 一行不用动 |
| 自启动策略 | 静止时扇区不变，定子磁场停在原地 | 加 I/F 强拖，或等角度细分/电流环到位 |
| 转速环 | 没有可靠的转速来源（缺角度细分） | 在 `bl_motor_task_active()` 里加外环 |
| CAN 指令解析 | 整车协议表未冻结，字段布局/缩放/字节序都没依据 | `bl.c` 的 `_bl_handle_command()` 已留好入口 |
| CSA 零点标定 | DRV8353 的 `CSA_CAL_x`（0x6 的 bit4/3/2）与 `CAL_MODE`（0x7 的 bit0）配套，属于标定流程 | 单独写一个标定函数，不在 BSP 里做 |
| DRV8353 工作参数下发 | 栅极电流、OCP 档位、CSA 增益要按实际 MOS 和电机定 | BSP 保持芯片复位默认值（6x PWM / 100 ns / 20 V/V / 双向 VREF） |

### 4.4 首次上板的两个坑

1. **电机不会自己转**。刷进去之后状态机停在 IDLE、功率级断电 —— 这是对的。
   要让电机动，把 `Core/Src/main.c` 的 `BL_DEMO_ENABLE` 改成 1 重新编译，
   它会延时 2 s → 使能 → 4 s 内把 q 轴电压从 0 加到 2 V → 保持 2 s → 软停。
2. **温度联锁可能挡住使能**。两路 NTC 的 R25 与 B 值还是占位值（10k / 3950），
   读数落到了电气上不可能的区间就会置 `TEMP_INVALID` 并锁存。
   首次上板可以先把 `bl_motor.h` 的 `BL_MOTOR_TEMP_PROTECT_ENABLE` 设 0，
   标定完 NTC 再设回 1 —— **设 0 期间必须有人盯着，且不要让电机堵转**。

---

## 5. 已知限制

1. **PWM 占空比上限约 96.5%**。低边采样要求采样窗口落在下桥导通期间；注入组 3 个通道 × 25 个 ADC 时钟 = 1.76 µs，对应约 3.5% 的占空比裕量。超过就得换采样策略（移相采样或改用高边采样）。
2. **电流量程 ±165 A 对 N5065 偏大**。灵敏度 100 mA/mV，而 3.3 V 满量程对应 ±165 A。实际工作时相电流几十安，分辨率浪费了。要改善就把 `CSA_GAIN` 调到 40 V/V（`bsp_spi.h` 的宏 + `bsp_adc.h` 的 `BSP_ADC_CSA_GAIN` 两处必须同时改），或者换更大阻值的采样电阻。
3. **`bsp_can_init()` 会把 FDCAN 内核时钟设成 PCLK1**。改 APB1 分频会直接改掉波特率。
4. **NRST（P2-19）在原理图上是空脚**。没有外部复位按钮，SWD 只能靠软件复位。调试上不影响，但板子上没有物理复位键。

---

## 6. `.ioc` 与 CubeMX 生成通道

工程里的 `foc_driver.ioc` 是**能用 CubeMX 命令行直接生成**的版本。生成方式：

```bash
# 建一个临时目录放脚本；别在工程目录里生成，Core/ 会被覆盖
mkdir -p D:/Hal_code/_gen
printf 'config load D:/Hal_code/foc_driver/foc_driver.ioc\nproject generate\nexit\n' > D:/Hal_code/_gen/s.txt
cd "E:/system/Desktop/新建文件夹"
cmd //c "STM32CubeMX.exe -q D:\Hal_code\_gen\s.txt > D:\Hal_code\_gen\run.log 2>&1"
```

脚本文件三行，`config load` 的路径用正斜杠、行尾用 LF：

```
config load D:/Hal_code/foc_driver/foc_driver.ioc
project generate
exit
```

实测 22~24 秒出全套产物：`CMakeLists.txt`、`CMakePresets.json`、`Core/`、
`Drivers/`、`cmake/`、`STM32G431xx_FLASH.ld`、`startup_stm32g431xx.s`。
启动器会把 Java 子进程 detach，父进程立刻返回 0，**必须轮询生成物判断完成**，
不能看退出码。

### 6.1 一个必须先满足的前提

`.ioc` 里必须有这两行，否则 CubeMX 会弹数据库迁移对话框并在 script 模式下
**永久阻塞**（日志停在 `Begin LoadConfig()`，不报错、不退出）：

```
MxDb.Version=DB.6.0.181
ProjectManager.AskForMigrate=false
```

`DB.6.0.181` 是安装目录 `db/` 的版本（用户库 `~/.stm32cubemx/databases/` 里是旧的
`DB.6.0.150`）。写成旧版本号一定会卡。

### 6.2 手写 `.ioc` 的三条硬规则（都是本轮实测得出，文档里查不到）

CubeMX 对 `.ioc` 的容错很差：**参数不被接受时不会报错，而是把整个外设悄悄丢掉**，
只在日志里留一行 `IP not ready for code generation`。三条必守规则：

1. **共享信号（`S_` 前缀）必须配 `SH` 段**，否则整个定时器不被实例化。
   引脚写 `PA8.Signal=S_TIM1_CH1`，就必须同时有：

   ```
   SH.S_TIM1_CH1.0=TIM1_CH1,PWM Generation1 CH1 CH1N
   SH.S_TIM1_CH1.ConfNb=1
   ```

   缺了这两行，CubeMX 把 `ConfNb` 算成 0，判定"该共享信号无可用配置"，
   引脚绑定失败 → TIM1 消失，且日志里一条相关报错都没有。
   （这也是 `water_car_motor.ioc` 从来没生成过 `tim.c` 的原因。）

2. **定时器必须有内部时钟源虚拟引脚**，并且要出现在 `Mcu.PinN` 列表里：

   ```
   VP_TIM1_VS_ClockSourceINT.Mode=Internal
   VP_TIM1_VS_ClockSourceINT.Signal=TIM1_VS_ClockSourceINT
   ```

3. **互补输出的通道选择器名字要带 `N`**，且与 `SH` 里的描述串逐字一致：

   | 位置 | 正确写法 |
   |---|---|
   | 键名 | `TIM1.Channel-PWM\ Generation1\ CH1\ CH1N` |
   | 值 | `TIM_CHANNEL_1` |
   | IPParameters 里的显示名 | `Channel-PWM Generation1 CH1 CH1N` |
   | SH 描述串 | `PWM Generation1 CH1 CH1N` |

   写成 `PWM Generation1 CH1`（不带 N）时，CubeMX 报
   `Parameter (Channel) has Id (...) currently not set`，通道配置整段不生成。

另外两个键名坑：G4 支持 PWM 抖动，所以**周期与比较值的键是
`PeriodNoDither` / `PulseNoDither_N`**，不是 `Period` / `Pulse-PWM Generation1 CH1`。
写错会报 `Invalid parameter (Pulse)`。

### 6.3 死区：`DeadTime` 是 DTG 寄存器原值，不是纳秒

`BSP_BOARD_INV_DEADTIME_NS = 200`（200 ns）。170 MHz 下 tDTS = 5.882 ns，
`200 / 5.882 ≈ 34` 个 tDTS，落在 DTG[7:5]=0xx 段，所以 **`TIM1.DeadTime` 必须填 34**。

历史上这里填过 200 —— 那按 DTG 编码是 `(32+8)×16×tDTS ≈ 3.76 µs`，
在 20 kHz（周期 50 µs）下相当于吃掉 7.5% 的调制范围，是错的。
实际生效值由 `bsp_inverter_get_dtg()` 算，`.ioc` 只是文档，但文档写错一样会误导。

### 6.4 CubeMX 仍然不肯实例化的部分

| 项目 | 状态 | 为什么不影响使用 |
|---|---|---|
| `SPI1` | 引脚信号（`PA5/PA6/PA7 = SPI1_*`）保留，但 IP 被丢，不生成 `spi.c` | `bsp_spi.c` 自己 `HAL_SPI_Init` + 配 DMA，从不调用 `MX_SPI1_Init` |
| `TIM1` 通道/死区/Break | 只生成时基（`MX_TIM1_Init` 里没有 `HAL_TIM_PWM_ConfigChannel`） | `bsp_inverter.c` 完整重配通道、死区、Break、TRGO |
| `ADC1` 注入组 / DMA | `.ioc` 里不配（配了会被判非法：`DMAContinuousRequests` 无 DMA 时非法） | `bsp_adc.c` 手动 `HAL_DMA_Init` + `__HAL_LINKDMA` + 注入组配置 |

结论：`.ioc` 在本工程里的职责是**引脚与时钟的单一来源 + 可重复生成工程骨架**；
外设细节一律以 BSP 代码为准。这与 `bsp-dl-bl` 的分层约定一致 —— BSP 是唯一的
寄存器/外设配置处。

要在图形界面里把这些补齐：用 CubeMX 打开 `foc_driver.ioc`，
在 Pinout 里点开 SPI1 → Mode 选 Full-Duplex Master → 生成即可，
CubeMX 会自己写回 `SH`/`VP`/`CH1N` 的正确形式。

### 6.5 两个会被 CubeMX 按 `.ioc` 裁掉的地方（本工程已在 CMakeLists 兜底）

上一节的"不影响使用"这个判断是**错的**，已修正。BSP 调用的是 HAL，
而 CubeMX 会按 `.ioc` 里实例化成功的外设，**同时裁掉两样东西**：

1. `Core/Inc/stm32g4xx_hal_conf.h` 里的 `HAL_xxx_MODULE_ENABLED` 宏
   —— 未实例化的外设，宏被写成注释。
2. `cmake/stm32cubemx/CMakeLists.txt` 的 `STM32_Drivers_Src`
   —— 未实例化外设的 HAL 源码**根本不进编译列表**。

后果链（本工程实测复现）：

| 现象 | 报错 |
|---|---|
| 只有宏被注释 | `bsp_spi.c` 里 `SPI_HandleTypeDef` 未知类型、`HAL_SPI_TransmitReceive` 隐式声明 |
| 只补了宏、没补源码 | 编译过，链接报 `bsp_spi.c: dangerous relocation: unsupported relocation` |
| 宏 + 源码都补上 | 通过 |

**兜底写在顶层 `CMakeLists.txt`**，因为该文件头部写着
`This file is generated only once, and is not re-generated`，CubeMX 不会覆盖它。
位置有讲究：

```cmake
# add_subdirectory 之前：add_compile_definitions 要能传到 STM32_Drivers
add_compile_definitions(HAL_SPI_MODULE_ENABLED)

add_subdirectory(cmake/stm32cubemx)

# add_subdirectory 之后：目标 STM32_Drivers 此时才存在
target_sources(STM32_Drivers PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/STM32G4xx_HAL_Driver/Src/stm32g4xx_hal_spi.c)
```

### 6.6 FDCAN 中断归 BSP，CubeMX 侧必须关掉

`bsp_can.c` 的 FDCAN 句柄是模块私有的 `s_bsp_can_module.handle`，
而 CubeMX 生成的中断入口调用的是 `Core/Src/fdcan.c` 的全局 `hfdcan1`。
两者同名会重复定义。所以 `.ioc` 里：

```
NVIC.FDCAN1_IT0_IRQn=false:5:0:false:false:true:true:true:true
```

首字段 `false` 表示不生成该中断。中断的优先级设置与使能由
`bsp_can_init()` 里的 `HAL_NVIC_SetPriority` + `HAL_NVIC_EnableIRQ` 完成（5, 0）。

### 6.7 生成位置：千万不要在工程目录里直接生成

**CubeMX 把产物写到 `.ioc` 所在目录**，不是当前工作目录。
所以 `config load D:/Hal_code/foc_driver/foc_driver.ioc` + `project generate`
会直接覆盖本工程的 `Core/Src`、`Core/Inc`。要试生成，先把 `.ioc` 复制到临时目录。

另外两套初始化现在是并存的：main.c 里 `MX_ADC1_Init()`、`MX_ADC2_Init()`、
`MX_TIM1_Init()`、`MX_FDCAN1_Init()` 先跑，随后 `USER CODE 2` 里 `bsp_init()`
完整重配。代价是 FLASH 从 43152 B 涨到 46188 B（约 +3 KB）。

### 6.8 根因已定位并修复（2026-10-28 晚）：SPI 引脚缺 Mode 行 + IP 未登记

前面几节里"根因未定位"的结论**已作废**。对比官方例程
`STM32Cube_FW_G4_V1.6.3/Projects/NUCLEO-G431KB/Examples/SPI/...` 的 `.ioc`
后确认，CubeMX 实例化一个外设需要**两个条件同时成立**：

1. **登记**：外设名出现在 `Mcu.IP*` 列表里（`Mcu.IPNb` 计数要一致）；
2. **引脚 Mode 行**：占用其信号的每个引脚都要声明工作模式。

之前"照抄 8110_foc 的 SPI1 参数块也照样被丢"的谜底：参数块抄了，
但 ① 没登记进 `Mcu.IP`，② 引脚上没有 `Mode` 行。CubeMX 加载时对
悬空信号 + 未登记 IP 的处理是**静默丢弃 IP、保留引脚信号名**——
这就是 `.ioc` 里能看到 `PA5.Signal=SPI1_SCK` 却没有 `spi.c` 的原因。

修复内容（`foc_driver.ioc`）：

```
Mcu.IP 列表加入 SPI1（IPNb 7→8）
PA5.Mode / PA6.Mode / PA7.Mode = Full_Duplex_Master    ← 关键缺项
SPI1.* 参数块按官方键集补齐（值照 bsp_spi.c：16 位、模式 1、软 NSS、32 分频）
ProjectManager.functionlistsort 加入 MX_SPI1_Init
```

验证结果（临时目录 + 原地生成各一次）：

| 检查项 | 结果 |
|---|---|
| `Core/Src/spi.c` 生成，`hspi1` 参数与 bsp_spi.c 一致 | ✅ |
| `stm32g4xx_hal_conf.h` 的 `HAL_SPI_MODULE_ENABLED` 激活 | ✅ |
| `cmake/stm32cubemx/CMakeLists.txt` 包含 `stm32g4xx_hal_spi.c` | ✅ |
| `main.c` 按序调用 `MX_SPI1_Init()` | ✅ |
| `stm32g4xx_it.c` 不生成 FDCAN1 处理函数（NVIC=false 维持） | ✅ |

CMakeLists 兜底相应调整：`add_compile_definitions(HAL_SPI_MODULE_ENABLED)`
**已撤销**——hal_conf 里宏已激活，命令行再定义会报 40 处
`macro redefined`；`target_sources(... hal_spi.c)` 保留为防御性保险
（与生成列表重复时 CMake 自动去重，零成本）。

代价：FLASH 46280 → 46488 B。`MX_SPI1_Init()` 与 `bsp_spi_init()` 并存，
后者完整重配，与 ADC/TIM/FDCAN 的处理方式一致。
