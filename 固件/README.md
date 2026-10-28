# 固件

两块功率板的 STM32G431CBU6 固件，**v1.0（2026-10-28）**。

| 工程 | 板子 | 功率级 | 内容 |
|---|---|---|---|
| `foc_driver` | FOC 驱动板 | DRV8353 + 6×100 V MOS | 三相无刷：TIM1 六路互补 PWM、注入组相电流、霍尔、SPI 寄存器诊断、CAN |
| `regulator_driver` | 稳压板 | DRV8161 双向半桥 | 8S 超容 Buck/Boost：双向半桥、泄放 PWM、8+6 通道 ADC、CAN |

MCU 不在板上：两板都通过 J_MCU 排针（2×12 母座）插 **WeAct Normal QFN48
（STM32G431CBU6）** 核心板。`Bsp/Inc/bsp_board.h` 是各自引脚与 ADC 通道的**唯一真源**；
排针号与原理图接口合同逐脚对应，核对方法与结论见各工程的 `BSP引脚对照.md`
及仓库 `电力/` 下对应模块的 BOM / 网表。

## 结构

```
<工程>/
├── Bsp/            板级支持包（引脚真源、各外设驱动、安全态初始化）
├── BusinessLayer/  业务层（控制与状态机）
├── DataLayer/      数据层（执行与反馈）
├── Core/           CubeMX 生成（main / gpio / 外设初始化 / 中断）
├── cmake/          工具链与 CubeMX 源文件列表
├── <工程>.ioc      CubeMX 工程（外设与引脚配置的真源）
└── BSP引脚对照.md  引脚、复用陷阱、改板约束
```

## 构建

依赖：CMake ≥ 3.22、Ninja、`arm-none-eabi-gcc`、STM32CubeMX（G4 固件包 V1.6.3）。

> `Drivers/`（HAL + CMSIS 第三方驱动）**不入库**：用 CubeMX 打开对应 `.ioc`，
> 选择 CMake 工具链生成代码，会补齐 `Drivers/` 及 `cmake/stm32cubemx/CMakeLists.txt`
> 的源文件列表。顶层 `CMakeLists.txt` 只生成一次，可安全保留本地修改。

```bash
cmake --preset Debug -G Ninja
cmake --build build/Debug
```

v1.0 基线：两工程均 **0 error 0 warning**；`foc_driver` 46460 B Flash，
`regulator_driver` 31848 B Flash。

## 状态与限制

- 仅完成原理图阶段，**固件未上板验证**；`bsp_board.h` 的电气结论来自网表与数据手册。
- 两板共用同一颗核心板但**不能混烧**：P1/P2 在各板上的功能分配不同。
- FOC 板 PA11/PA12 与核心板 Type-C 的 D−/D+ 硬连：接 CAN 时不得插 USB 数据线。
- 稳压板 PERMIT 为单向输出，DRV8161 的 nFAULT 无法回读——驱动故障软件不可观测。
- FOC 刹车滤波 BKF=5 约 1.5 µs 延迟，首次上板需示波器确认后再定档。
- FOC ADC 注入组的 JSQR 组装错误已在 v1.0 修复；上板后仍需读 `ADC1->JSQR` 复核
  （预期 `JL=2`、JSQ1..3 = 2/4/1）。
