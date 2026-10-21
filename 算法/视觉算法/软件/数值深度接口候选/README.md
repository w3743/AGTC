# RK3588 统一深度输出

本目录名沿用“数值深度接口候选”，当前服务已部署。2026-10-21 按用户最新要求，常态运行改为 ISP 按需尺寸直出，浏览器仅保留一份实时深度图；相机预览、设备状态和标定保留。

## 当前运行方式

- 双路 `/dev/video33`、`/dev/video42` 实际协商为 **640×360、NV12、约 30 FPS/眼**，由 RK3588 ISP 完成缩小，应用不持续接收 4K 帧再缩图。
- 当前传感器驱动原生模式仍为 **3864×2192、10-bit Bayer**。640×360 是 ISP 输出，不是新增加的传感器原生模式。ISP 枚举下限 32×32；本轮实际配置、验证的是 640×360。
- 已关闭持续 4K MPP 编码与传输。标定、手动拍摄原图仍可显式按需切到 3840×2160，结束后恢复常态采集。
- 唯一主深度后端为当前 CPU SGBM，scale 0.375、2 线程，结果 **131×240**，Z 深度筛选 **0.3–5 m**。范围设置不代表独立实物精度验收。
- 测距标注框、点击读值、点云、远距 ROI 服务与独立 NPU 深度预览均已移除，相关工作线程不再启动。离线算法/几何实验模块保留用于复现，不接入生产服务。

[深度图](http://127.0.0.1:8877/?view=depth) · [相机预览](http://127.0.0.1:8877/?view=raw) · [设备状态](http://127.0.0.1:8877/hardware) · [标定](http://127.0.0.1:8877/calibration)

界面共用 [ui.css](app/ui.css) 和 [ui.js](app/ui.js)，使用一套导航、字体、按钮和面板；调试信息放入折叠区。默认深度视图没有近/远分支切换、框选或点云入口。

## 唯一生产者与数值接口

生产者为 [server.py](app/server.py) 中 `Live.metric_work → Live.publish`，编码与帧记录实现为 [depth_wire.py](app/depth_wire.py)。`GET /depth.jpg` 是该深度的伪彩色显示，`GET /depth.bin` 是同一生产链路的数值编码；不启动第二个估深模型。唯一语义来源仍为[深度接口契约](../../深度接口契约.md)。

| 项目 | 当前值 |
|---|---|
| 电脑转发 URL | `http://127.0.0.1:8877/depth.bin` |
| 板端 HTTP 端口 | `8088`；ADB 转发 `tcp:8877 → tcp:8088` |
| schema | `depth-z-u16le-v1` |
| 载荷 | C 行优先、uint16 小端毫米、131×240，共 **62,880 bytes** |
| 无效与量程 | `0` 无效；有效值 300..5000 mm，范围外不夹成有效端点 |
| 坐标意义 | 左相机校正光轴的 Z 深度，不是光心到点的直线距离 |
| 帧元数据 | HTTP `X-Depth-Metadata`，与 payload 位于同一不可变缓存记录 |
| 同帧绑定 | seq、session、epoch、K、标定/预处理ID、左右源PTS、发布单调时间 |
| 指定缓存帧 | `/depth.bin?sequence=N`；过期409、非法序号400 |
| 真实硬件 ready | PTS时钟映射与机身/云台外参未验证，相关字段保持null，consumer必须 `legacy_ready=false` |

单独抓取一次真实帧及其原始 HTTP 头：

```powershell
adb forward tcp:8877 tcp:8088
curl.exe --fail --silent --show-error --dump-header depth.headers.txt --output depth.bin http://127.0.0.1:8877/depth.bin
```

直接复用已抓取的同帧 fixture：

- [current-sample.depth.bin](evidence/current-sample.depth.bin)：真实板端 payload，SHA-256 `e995be0040da04608279a71232307a6d9c628c1d091e41caf7e79f8126a0e43d`。
- [current-sample.metadata.json](evidence/current-sample.metadata.json)：原始 `X-Depth-Metadata`，SHA-256 `a09668b741e30667490fb73c4c0e08a7d36711fb836af7611ce86cdbef0d9e30`。
- [current-sample.headers.txt](evidence/current-sample.headers.txt)：这次响应的原始头字段。

这组数据仅证明真实接口连通、字节序和帧绑定。实测距离、跨设备曝光时钟和真实安装外参不从 fixture 自动获得。

## 消费适配与回归入口

共享消费适配器为 [depth_adapter.py](app/depth_adapter.py)，LASSC桥接由训练对话维护 `算法/仿真训练环境/scripts/lassc_consumer.py`。本模块不复制训练侧适配器，也不写其生产源。

本轮已修复共享 adapter 的拒绝帧状态提交问题：未来时间或其他未就绪帧不再污染 seq、epoch 和时钟高水位，后续正常帧仍可消费。另支持显式 `target_k`；默认保持原有 `TARGET_K`，不能将理论缩放 K 自动等同真实渲染 K。几何、时钟、位姿校验仍保持拒绝未知条件。供本次 QA 复现的接口源版本为：

- `app/depth_adapter.py` SHA-256：`1561ae01bf63420afc19296cadfbda75a628226e2cccec0d99d39133a3deeee5`。
- `app/depth_wire.py` SHA-256：`c99a2fec72160ac4cfe9a2cc18752064df2309b832f6c9a4ba02d9e20a0d4dfa`。

这两个接口源不随后续页面排版修改。adapter 修正供电脑侧消费者复用，未包含在本次板端 UI 安装白名单中；板端 producer 的 wire 源保持不变。

截至 NPU 监控修复，当前完整本地套件 **78/78 通过，6.647 s**；包括契约跨层、状态回退、HTTP、几何和保留的离线实验回归。较早的 ARM 63项与23项记录分别属于当时源版本，不冒充当前完整板测。新增契约验收文件由契约对话维护，不由本对话复制一份：

```powershell
python -B -m unittest discover -s '算法\视觉算法\软件\数值深度接口候选\tests' -v
```

关键入口：[契约验收](tests/test_depth_contract_acceptance.py)、[adapter回归](tests/test_depth_adapter.py)、[producer原子发布](tests/test_depth_producer_http.py)、[真实HTTP传输](tests/test_depth_http_transport.py)、[退役路由回归](tests/test_far_routes.py)。

## 部署与实际验证

直出/单深度切换时的证据为 [ui-refactor-result.json](evidence/ui-refactor-result.json)，界面截图为 [ui-depth-final.png](evidence/ui-depth-final.png)。该历史切换记录中包含当时板端源 SHA、两路 V4L2 实际布局、服务状态、真实路由结果和 fixture hash。板端墙钟仍为2024年，记录日期使用主机 UTC 并按北京时间解释。

本轮短时观测：相机约 **30 FPS/眼**，producer 约 **27.30 FPS**；一帧处理快照约20.1 ms。服务 active，`hardware_video.enabled=false`，状态不再提供 far/NPU预览分支。短测不代表长时间热稳定性或 sensor-to-action 延迟验收。

旧 `/measure`、`/pointcloud`、`/far` 跳转到唯一深度视图；旧 `/stereo4k`、`/measure4k` 跳转到普通相机预览。标注图、点云、远距、独立NPU和4K视频数据接口返回410，不启动对应任务。已经从应用目录删除相关HTML，旧参数、标定、模型、runtime和systemd配置未改动。

最终切换前回滚包：`算法/视觉算法/备份/直出与单深度前_20261021_1034/direct-single-20261021-1034.tar.gz`，SHA-256 `a5823532546e4c47a427f90e1859dd5bf47619f2b71c9cfa6aaa14ea7448973f`。板端原件 `/opt/rk-lightstereo/backups/direct-single-20261021-1034` 包含 manifest、旧文件及 installer.py。安装前已拉回并核对归档SHA；本轮未执行回滚演练。最终模板文字修正后的活动SHA以本轮证据为准，prepare manifest保留切换时的版本。

如需恢复切换前的状态（会重启视觉服务）：

```powershell
adb shell python3 -B /opt/rk-lightstereo/backups/direct-single-20261021-1034/installer.py rollback --backup /opt/rk-lightstereo/backups/direct-single-20261021-1034
adb shell systemctl is-active rk-lightstereo.service
```

历史 `deploy_candidate.py` 仅对应早期9文件候选，不能用于本次收敛后的页面安装。原始完整算法备份未改动。

## NPU 性能监控（2026-10-21）

设备状态页将 NPU 三核利用率、最近60次采样曲线、三核平均/最忙核心、驱动报告频率、电源和 NPU 温度放在首屏。当前深度后端仍为 CPU SGBM，独立 NPU 预览保持退役；监控页面不会启动深度计算，也不会启动推理测试。

负载唯一来源为 `/sys/kernel/debug/rknpu/load`，按 Core0/1/2 标签解析并保留核心位置。缺核、非法、重复、超范围数据保持 `null`；只有三核完整时计算平均/最大值。频率来自 devfreq `cur_freq`；不再将 NPU devfreq `load` 解释为利用率。板端已观察到 NPU 电源 `off`、三核0%，但 devfreq 仍为 `100@1000000000Hz`，所以频率与功耗状态必须分别解释。实现与测试入口为 [realtime.py](app/realtime.py)、[hardware.html](app/hardware.html)、[采样回归](tests/test_hardware_monitor.py)。驱动采样方式也可参照 [Rockchip 官方 rknpu_debugger.c](https://github.com/rockchip-linux/kernel/blob/develop-5.10/drivers/rknpu/rknpu_debugger.c)。

`/hardware/status` 带单调采样时间、`sample_seq`、`sample_age_ms`，约1秒内复用同一采样，避免多客户端反复读取资源文件。浏览器按采样序号记录趋势；后台暂停、连接失败或样本过期时清空当前数值并留曲线断点。深度状态接口失败不会阻断 NPU 采样。RSS/PSS 只是视觉进程总内存，不能解释为 NPU 专用内存；利用率也不能换算成实际TOPS或模型吞吐。

部署证据与**当前活动源SHA**以 [npu-monitor-result.json](evidence/npu-monitor-result.json) 为准。本次仅部署 `server.py`、`realtime.py`、`hardware.html`，不改模型、标定、wire/adapter或服务配置。78项本地Python测试、7项页面状态检查通过，新增6项采样测试也在板端Python 3.9使用生产代码通过。实际连续12次监控请求中，深度帧计数保持0，证明监控没有唤醒深度任务；该短测HTTP中位响应约5.79 ms，不代表长期稳定性验收。

为核对监控确实能随任务变化，单独运行现成保存输入的RKNN前向测试约13.5秒（不切换生产算法）：三核从0/0/0升到80/79/83%，结束后回到0/0/0，稍后电源off。完整原始采样见 [npu-monitor-load-test.json](evidence/npu-monitor-load-test.json)，模型测试结果见 [npu-monitor-forward-check.json](evidence/npu-monitor-forward-check.json)。这只验证计数响应；1秒浏览器采样可能漏掉短脉冲，不构成模型加速或距离精度改进。最终页面截图为 [npu-monitor-final.png](evidence/npu-monitor-final.png)。

当前监控修复回滚包为 `算法/视觉算法/备份/NPU监控修复前_20261021-120543/npu-monitor-20261021-120543.tar.gz`，SHA-256 `f53967945b9dbcc8c902c112beb94f1c7d9603ce611c040ecb0fc49fbfedab3f`；安装前已拉回校验。备份 manifest 记录初次安装SHA，最终排版调整后的SHA见本节活动证据。需要恢复本次监控改动前的状态时：

```powershell
adb shell python3 -B /opt/rk-lightstereo/backups/npu-monitor-20261021-120543/installer.py rollback --backup /opt/rk-lightstereo/backups/npu-monitor-20261021-120543
adb shell systemctl is-active rk-lightstereo.service
```

## 历史实验与未验收项

[optimization-result.json](evidence/optimization-result.json) 保存上一阶段32组CPU、RKNN及原生ROI对比和当时部署SHA。其远距/4K并行数据是历史实验结果，相关生产分支现已按用户要求移除；不能当作当前仍有远距输出。原生ROI、C++/NPU吞吐也不构成真实距离精度。

当前仅具备室内条件，尚未完成独立量具误差验收、户外运动植物与光照突变、联动短曝光/硬同步/HDR、相机到车体/云台外参、时钟映射、长期热稳定性和 sensor-to-action 延迟验证。本轮不操作训练进程；训练和整链验收状态由相应责任对话维护。