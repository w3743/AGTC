# 芯片核心源代码（Core/Src）

STM32 核心、外设初始化、中断和运行时支持代码。

## 文件

- [adc.c](adc.c)：模数转换外设初始化实现。
- [fdcan.c](fdcan.c)：FDCAN 外设初始化实现。
- [gpio.c](gpio.c)：通用输入输出初始化实现。
- [main.c](main.c)：程序入口和初始化。
- [stm32g4xx_hal_msp.c](stm32g4xx_hal_msp.c)：硬件抽象层底层支持实现。
- [stm32g4xx_it.c](stm32g4xx_it.c)：中断处理实现。
- [syscalls.c](syscalls.c)：运行时系统调用实现。
- [sysmem.c](sysmem.c)：运行时内存分配支持实现。
- [system_stm32g4xx.c](system_stm32g4xx.c)：芯片系统初始化实现。
- [tim.c](tim.c)：定时器初始化实现。

修改 CubeMX 生成代码时，请留意用户代码区域和重新生成的影响；工程配置见上级工程的 `.ioc` 文件。

[返回上级目录](../README.md)
