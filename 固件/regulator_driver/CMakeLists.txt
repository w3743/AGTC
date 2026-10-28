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
set(CMAKE_PROJECT_NAME regulator_driver)

# Enable compile command to ease indexing with e.g. clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

# Core project settings
project(${CMAKE_PROJECT_NAME})
message("Build type: " ${CMAKE_BUILD_TYPE})

# Enable CMake support for ASM and C languages
enable_language(C ASM)

# ---- HAL 模块开关的历史兜底，已撤销（2026-10-28）---------------------------
# 根因已修复：.ioc 丢 ADC1/ADC2/FDCAN1 是因为它们既没登记进 Mcu.IP 列表、
# 也没有参数块（只有引脚信号是悬空引用）。补齐后 CubeMX 正常实例化三者，
# hal_conf 里 HAL_ADC/FDCAN_MODULE_ENABLED 恢复激活，无需命令行兜底。
# ⚠️ 若再在此处 add_compile_definitions，会与 hal_conf 里已激活的 #define
#    冲突，每个编译单元报 "macro redefined"。
# 另：PA7 上的 ADC2_IN4 已从 .ioc 移除——它只接 COMP2_INP，不进 ADC 序列；
#    无 IP 引脚的默认态就是模拟输入，BSP 运行时也会配。
# 完整机制见 BSP引脚对照.md §12.4。

# Create an executable object type
add_executable(${CMAKE_PROJECT_NAME})

# Add STM32CubeMX generated sources
add_subdirectory(cmake/stm32cubemx)

# ---- HAL 源文件保险（防御性保留）------------------------------------------
# .ioc 修复后 cmake/stm32cubemx/CMakeLists.txt 已正常包含这三个源文件，
# target_sources 与其重复，CMake 会自动去重、零成本。保留以防 .ioc 回退。
target_sources(STM32_Drivers PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/STM32G4xx_HAL_Driver/Src/stm32g4xx_hal_adc.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/STM32G4xx_HAL_Driver/Src/stm32g4xx_hal_adc_ex.c
    ${CMAKE_CURRENT_SOURCE_DIR}/Drivers/STM32G4xx_HAL_Driver/Src/stm32g4xx_hal_fdcan.c
)

# Link directories setup
target_link_directories(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined library search paths
)

# Add sources to executable
target_sources(${CMAKE_PROJECT_NAME} PRIVATE
    # 用户代码：BSP 层（板级支持包）
    # 顺序只是为了好读，实际由 Bsp/bsp.c 里的调用顺序决定
    Bsp/bsp.c
    Bsp/Src/Callback/bsp_callback.c
    Bsp/Src/Gpio/bsp_gpio.c
    Bsp/Src/Bridge/bsp_bridge.c        # 双向半桥：TIM1 CH1/CH1N 互补 + 死区（稳压板）
    Bsp/Src/Dump/bsp_dump.c            # 泄放：TIM4_CH2 单路 PWM（稳压板）
    Bsp/Src/Adc/bsp_adc.c              # ADC1 八通道 + ADC2 六通道，DMA 循环
    Bsp/Src/Can/bsp_can.c

    # 用户代码：DL 层（物理量换算）
    DataLayer/Src/dl.c
    DataLayer/Src/dl_bus.c             # 母线 150k/10k、超容 68k/10k 分压
    DataLayer/Src/dl_cell.c            # 8 节累积抽头（含 TEST 网络修正）
    DataLayer/Src/dl_temp.c            # 3 路 NTC，Beta 方程（用到 logf）
    DataLayer/Src/dl_power.c           # 电流、泄放功率估算
    DataLayer/Src/dl_can.c             # CAN 收发封装

    # 用户代码：BL 层（策略与状态机）
    BusinessLayer/Src/bl.c
    BusinessLayer/Src/bl_regulator.c   # 稳压策略状态机 + 安全联锁

    # 注：本工程没有 bsp_inverter / bsp_hall / bsp_spi。
    #     三相无刷 + 霍尔 + DRV8353 寄存器配置全部属于 foc_driver 工程。
    #     两块板共用同一颗 G431 核心板，但功率板不同、固件也各自独立。
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

    # 数学库：DataLayer/Src/dl_temp.c 用 logf() 解 NTC 的 Beta 方程。
    # 不加 m 会在链接期报 undefined reference to `logf'。
    m
)
