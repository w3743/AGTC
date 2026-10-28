# 旧版双目采集与深度程序

完整操作、标定方法和验证限制见 `使用说明.md`。本目录直接文件如下：

| 文件 | 用途 |
| --- | --- |
| `使用说明.md` | 拍照、SGBM 深度和标定操作说明。 |
| `annotate_capture.py` | 对指定历史拍摄组进行棋盘角点标注，结果仍需人工核对。 |
| `calibrate.py` | 处理双目棋盘图像并计算标定参数。 |
| `calibration-board.png` | 棋盘标定图。 |
| `depth.py` | 双目视差和深度处理。 |
| `index.html` | 拍照与深度网页界面。 |
| `rk-stereo.service` | 板端 systemd 服务配置。 |
| `server.py`、`start.sh` | 板端服务程序及启动入口。 |
| `test_capture.py`、`test_depth.py` | 采集和深度处理测试。 |
| `README.md` | 本目录文件索引。 |

`calibration/`、`captures/` 和 `diagnostic/` 保存标定、拍摄与诊断记录；不在这些证据目录内添加文件。历史操作说明不代表当前服务或现场精度已验收。
