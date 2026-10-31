# 芯片核心头文件（Core/Inc）

STM32 核心及外设初始化相关头文件。

## 文件

- [adc.h](adc.h)：模数转换外设初始化接口声明。
- [fdcan.h](fdcan.h)：FDCAN 外设初始化接口声明。
- [gpio.h](gpio.h)：通用输入输出初始化接口声明。
- [main.h](main.h)：程序入口和初始化接口声明。
- [stm32g4xx_hal_conf.h](stm32g4xx_hal_conf.h)：硬件抽象层配置接口声明。
- [stm32g4xx_it.h](stm32g4xx_it.h)：中断处理接口声明。
- [tim.h](tim.h)：定时器初始化接口声明。

修改 CubeMX 生成代码时，请留意用户代码区域和重新生成的影响；工程配置见上级工程的 `.ioc` 文件。

[返回上级目录](../README.md)
