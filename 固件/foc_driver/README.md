# 电机 FOC 驱动固件（foc_driver）

三相无刷电机 FOC 驱动板固件，包含逆变器、霍尔、SPI 诊断和 CAN 接口。

## 子目录

- [板级支持包（Bsp/）](Bsp/README.md)：板级支持包，负责板卡引脚、外设驱动和初始化。
- [业务层（BusinessLayer/）](BusinessLayer/README.md)：业务层，负责控制策略、状态机及调度。
- [构建配置（cmake/）](cmake/README.md)：交叉编译工具链配置和 CubeMX 构建配置。
- [芯片核心代码（Core/）](Core/README.md)：STM32 核心启动、外设初始化和中断相关代码。
- [数据层（DataLayer/）](DataLayer/README.md)：数据层，负责采样数据换算、状态及通信数据的组织。

## 文件

- [BSP引脚对照.md](BSP引脚对照.md)：引脚对照及硬件约束。
- [CMakeLists.txt](CMakeLists.txt)：CMake 构建配置。
- [CMakePresets.json](CMakePresets.json)：CMake 构建预设。
- [foc_driver.ioc](foc_driver.ioc)：CubeMX 工程配置。
- [startup_stm32g431xx.s](startup_stm32g431xx.s)：芯片启动代码。
- [STM32G431xx_FLASH.ld](STM32G431xx_FLASH.ld)：Flash 链接脚本。

构建方法、共同依赖及上板限制见[固件总说明](../README.md)；板卡引脚和硬件约束见 [BSP 引脚对照](BSP引脚对照.md)。
