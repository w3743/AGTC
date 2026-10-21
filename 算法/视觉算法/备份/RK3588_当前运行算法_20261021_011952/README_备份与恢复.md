# RK3588 当前运行算法备份

备份时间：2026-10-21 01:19:52（Asia/Shanghai）  
设备：`quarkpi-ca2`，ADB serial `b1371e33e9f6e49e`。归档：`rk-lightstereo.tar.gz`（76,113,801 字节）。

## 运行环境

- Debian GNU/Linux 11 (bullseye)，Linux 5.10.209，aarch64。
- Python 3.9.2、OpenCV 4.10.0、NumPy 1.26.4；隔离依赖在 `/opt/rk-lightstereo/runtime/vendor`，其中 RKNN Lite 2.3.2；运行库为 `librknnrt.so`。
- Rockchip RKNPU 内核驱动版本 0.9.6；`rkaiq_3A.service` 正在运行，相关 camera udev 规则及服务启动脚本已收入归档。

## 当前服务

服务的状态、ExecStart 和 WorkingDirectory 见 `服务清单.txt`。`rk-lightstereo.service` 为 active/enabled，工作目录 `/opt/rk-lightstereo`，入口 `/opt/rk-lightstereo/run.sh serve`；HTTP 服务监听 8088。`rk-stereo.service` 为 inactive/disabled，指向旧路径 `/opt/rk-stereo`。

## 备份范围与校验

归档包含正在运行的 `/opt/rk-lightstereo` 程序、模型、标定、隔离 runtime/vendor、examples、evidence 和本机备份材料；还包含活动/旧版 RK 服务 unit、rkaiq unit/init 脚本及 Rockchip 相机 udev 规则。保留了 `validation` 中的 meta 和 NPZ 参考数据。

为避免带入相机大文件，排除了 `/opt/rk-lightstereo/captures/`、`/opt/rk-lightstereo/output/`（如存在）和 `/opt/rk-lightstereo/validation/` 内的 PNG/JPG/JPEG/BMP 照片。恢复后若要运行基于保存照片的自检，需要另行重新采集或提供左右验证照片。没有备份整个操作系统、内核、系统包，也没有复制未运行的 `/opt/rk-stereo` 程序/runtime 或未引用的 `/opt/lightstereo`。

归档前后算法文件、服务配置和符号链接清单一致；归档内 1,232 个常规文件逐项 SHA-256 与设备源文件一致，2 个符号链接目标一致。设备临时 tar 与本地归档 SHA-256 相同。详细结果见 `backup-info.json`、`manifest-*.sha256`、`symlinks-*.txt` 和 `关键文件SHA256.txt`。

## 恢复

在兼容的 RK3588 Debian 11 系统上，将归档解到根目录以恢复原绝对路径。先把文件传到板端，再执行：

```sh
sudo tar --numeric-owner --same-owner --same-permissions --xattrs --acls -xzf /tmp/rk-lightstereo.tar.gz -C /
sudo systemctl daemon-reload
sudo systemctl enable rk-lightstereo.service
sudo systemctl start rk-lightstereo.service
sudo systemctl status rk-lightstereo.service --no-pager
```

恢复后检查相机节点、`rkaiq_3A.service`、标定 ID 和 `/status`。归档不含 captures 和 validation 照片；恢复硬件驱动需使用与目标板内核匹配的 Rockchip 系统镜像/驱动包。