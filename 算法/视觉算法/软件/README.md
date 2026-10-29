# 视觉软件入口

| 文件 | 用途 |
| --- | --- |
| `旧版SGBM启动.cmd` | 从 Windows 启动、更新或停止旧版双目拍照与 SGBM 深度服务；操作前参阅 `stereo/使用说明.md`。 |
| `README.md` | 本目录的软件入口说明。 |

`stereo/` 包含旧版采集与深度程序；`LightStereo源码/` 和 `LightStereo实验/` 包含相关交付和实验材料。交付清单、第三方运行库及拍摄、标定证据应保持原样，具体用途和验证范围见各自的使用说明。

`LightStereo实验/` 同目录材料包括：`实验说明.md` 与 `manifest.json` 记录实验和文件清单；`benchmark_cpu.py`、`benchmark_parallel.py`、`board_check.py` 用于板端与性能检查；`convert.py` 和 `convert.log` 分别保存模型转换程序和日志；`evaluate.py`、`evaluate_portrait.py`、`photometric_quality.py` 用于评估；`export_portrait.py`、`portrait.py` 处理纵向图像。`models/` 和 `runtime/` 保存对应模型及运行依赖。

`LightStereo源码/` 交付目录中的 `使用说明.md` 是操作和验证入口，`manifest.json` 是文件清单，`install.sh`、`run.sh` 分别是安装和运行入口，`verify.py` 检查交付内容。其子目录保存应用、模型、依赖及证据；交付内容以清单为准。
