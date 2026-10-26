# SDK 集成文件（需要同步到 SDK）

`src/sle_range_link/` 是示例工程本体；除此之外还需要在 SDK 里做 4 处集成改动，对应本目录下保持相对路径的文件：

| 文件（相对 SDK 根目录） | 改动内容 |
|---|---|
| `application/samples/products/Kconfig` | 引入 `source "samples/products/sle_range_link/Kconfig"`，把我们的 sample 挂进 menuconfig |
| `application/samples/products/CMakeLists.txt` | 把 `sle_range_link` 子目录加入构建（`add_subdirectory_if_exist` 形式） |
| `drivers/chips/bs2x/bs2x_sample_config.cmake` | 为 `bs21e-1100e` 目标追加 `SLEM_MEASURE_DIS`/`SLEM_CARKEY` 等所需特性开关 |
| `drivers/chips/bs2x/main_init/app_os_init.c` | 加大 BT/HADM 相关任务栈（`BT_STACK_SIZE 0xC00`、`BTH_SERVICE_STACK_SIZE 0xB00`），避免 IQ 上报时栈溢出 |

使用方式：把这 4 个文件**覆盖**到 SDK 对应路径（它们是官方文件的少量修改版，可 `diff` 对照官方版本确认改动范围），
再按根目录 `README.md` 的「编译」步骤构建。

> 说明：这些文件包含 SDK 原始版权头，仅作为「集成补丁」随工程提供；真正随工程开发的是 `src/sle_range_link/`。
> 若换成新版 SDK（如 BS2X 1.0.18），建议以上述 4 处改动为参考重新集成，而不是直接覆盖（不同版本的示例注册方式可能变化）。
