# 新固定状态标定与独立验收

状态：**geometry_pass_metric_accuracy_unverified**。参数已部署到板端米制深度服务；实物距离和建图精度尚未验证。

本次实际收到17组新屏幕棋盘照片，左右各54个角点全部检出。检测和拟合前已固定分组：前12组计算，后5组独立验收。没有混入历史照片、按验收误差删图或重新拟合验收图。格距24.2 mm。

| 指标 | 结果 |
|---|---:|
| 左 / 右单目拟合RMS | 0.235 / 0.221 px |
| 双目拟合RMS | 0.665 px |
| 独立验收左右对齐误差中位数 | 1.038 px |
| 独立验收左右对齐误差95分位 | 1.767 px |
| 独立验收左右对齐误差最大值 | 2.071 px |
| 独立验收跨相机预测RMS | 1.241 px |
| 估计光心间距 | 96.29 mm |

误差均为3840×2160原图像素。预设门槛沿用上一批：单目和双目RMS小于1 px、独立验收对齐95分位小于2 px、跨相机预测RMS小于3 px，且内参/畸变合理。结果：`{"mono_training_under_1px": true, "stereo_training_under_1px": true, "heldout_epipolar_p95_under_2px": true, "heldout_crossview_rms_under_3px": true, "camera_model_sanity": true}`。

采用五项畸变模型，先以本批训练图计算内参，再固定内参估计双目关系。手工测量的基线未参与约束。图片SHA256、角点、逐组误差和全部参数分别保存于detections.json与report.json；parameters.npz可由程序读取。

原图极线方向为vertical，参数必须与原始图像坐标一起使用；输入旋转或缩放时需同步转换，不可直接套到旧SGBM输入。角点及校正预览分别为corner-review.jpg、rectified-validation.jpg。

复现：`python -B stereo/calibrate.py batch --output stereo/calibration/current`。原始照片全部保留，历史参数保留；当前参数已接入实时服务。

实时深度：板端加载参数校验通过，计算接受范围0.3～5 m，默认自动色标、可切换固定色标，360×640转正画面；计算优化后实机观察约5.8～6.0帧/秒。采集、匹配、投影及色标相关9项测试通过，照片数量仍为70组，未自动拍照。运行性能不代表实物距离精度验证通过。
