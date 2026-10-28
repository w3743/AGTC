cmake_minimum_required(VERSION 3.22)

#
# This file is generated only once,
# and is not re-generated if converter is called multiple times.
#
# User is free to modify the file as much as necessary
#

# Setup compiler settings
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)


# Define the build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

# Set the project name
set(CMAKE_PROJECT_NAME foc_driver)

# Enable compile command to ease indexing with e.g. clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

# Core project settings
project(${CMAKE_PROJECT_NAME})
message("Build type: " ${CMAKE_BUILD_TYPE})

# Enable CMake support for ASM and C languages
enable_language(C ASM)

# ---- HAL 模块开关的历史兜底，已撤销（2026-10-28）---------------------------
# 根因已修复：.ioc 丢 SPI1 是因为 SPI 引脚缺 Mode 行（PA5/6/7 必须
# Mode=Full_Duplex_Master）且 SPI1 未登记进 Mcu.IP 列表。修好后 CubeMX 正常
# 实例化 SPI1，hal_conf 里 HAL_SPI_MODULE_ENABLED 恢复激活，无需命令行兜底。
# ⚠️ 若再在此处 add_compile_definitions(HAL_SPI_MODULE_ENABLED)，会与 hal_conf
#    里已激活的 #define 冲突，每个编译单元报 "macro redefined"。
# 完整机制见 BSP引脚对照.md §6.5。

# Create an executable object type
add_executable(${CMAKE_PROJECT_NAME})

# Add STM32CubeMX generated sources
add_subdirectory(cmake/stm32cubemx)

# ---- HAL 源文件保险（防御性保留）------------------------------------------
# .ioc 修复后 cmake/stm32cubemx/CMakeLists.txt 已正常包含 stm32g4xx_hal_spi.c，
# 这行 target_sources 与其重复，CMake 会自动去重、零成本。
# 保留它：万一将来 .ioc 被旧版本覆盖导致 SPI1 再被丢，至少链接不炸。
# 完整机制见 BSP引脚对照.md §6.5。
target_sources(STM32_Drivers PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/STM32G4xx_HAL_Driver/Src/stm32g4xx_hal_spi.c
)

# Link directories setup
target_link_directories(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined library search paths
)

# Add sources to executable
target_sources(${CMAKE_PROJECT_NAME} PRIVATE
    # 用户代码：BSP 层（板级支持包）
    # 顺序只是为了好读，实际由 bsp/bsp.c 里的调用顺序决定
    Bsp/bsp.c
    Bsp/Src/Callback/bsp_callback.c
    Bsp/Src/Gpio/bsp_gpio.c
    Bsp/Src/Inverter/bsp_inverter.c   # 三相逆变器（无刷）：TIM1 六路互补 PWM + 刹车 + ADC 触发
    Bsp/Src/Adc/bsp_adc.c
    Bsp/Src/Hall/bsp_hall.c
    Bsp/Src/Spi/bsp_spi.c
    Bsp/Src/Can/bsp_can.c

    # 用户代码：DL 层（数据层）—— 原始采样 → 工程量
    DataLayer/Src/dl.c
    DataLayer/Src/dl_motor.c          # 功率级状态 + 占空比下发（含 COAST/BRAKE 语义）
    DataLayer/Src/dl_sense.c          # 三相电流 / 母线 / 两路 NTC 换算
    DataLayer/Src/dl_can.c            # 接收交接槽 + 占位状态帧

    # 用户代码：BL 层（业务层）—— 决策
    BusinessLayer/Src/bl.c            # 编排：初始化顺序 + 1ms 节拍 + 100ms 上报
    BusinessLayer/Src/bl_foc.c        # 调制：Clarke/Park/反Park/SVPWM + 霍尔角度
    BusinessLayer/Src/bl_motor.c      # 策略：状态机 + 联锁 + 电压斜坡

    # 注：本工程没有 bsp_hbridge。直流有刷 H 桥属于 water_car_motor 工程。
)

# Add include paths
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    Bsp
    Bsp/Inc
    Other/Type/Inc
    DataLayer
    DataLayer/Inc
    BusinessLayer
    BusinessLayer/Inc
)

# Add project symbols (macros)
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined symbols
)

# Remove wrong libob.a library dependency when using cpp files
list(REMOVE_ITEM CMAKE_C_IMPLICIT_LINK_LIBRARIES ob)

# Add linked libraries
target_link_libraries(${CMAKE_PROJECT_NAME}
    stm32cubemx

    # libm：bl_foc.c 用 sinf/cosf/sqrtf，dl_sense.c 用 logf。
    # 少了这一行会在链接期报 undefined reference to `sinf'。
    # 用的是 --specs=nano.specs 的 newlib-nano，libm 也是精简版，
    # 代价是这几个函数精度略低于标准版（对调制来说完全够）。
    m
)
