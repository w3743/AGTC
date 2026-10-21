# ATK-IMU901 厂商手册本地参考

用户要求将厂家资料下载保存在本地。本目录原样保存 `ATK-IMU901_User_Guide_v1.2.pdf`。

- 官方资料入口：https://github.com/alientek-openedv/Products/blob/master/zdyz_docs/modules/other/ATK-IMU901.rst ，其下载链接指向百度网盘。
- 实际取得的PDF镜像：https://raw.githubusercontent.com/SuperMarioSF/IMU901ComServer/master/datasheet/ATK-IMU901%20User%20Guide%20v1.2%20.pdf
- SHA256：`96a05814b5b3a09825528e988629c1c7af9fbc9ac5843decb6b3b7ae6f12f3aa`。
- 下载时间：2026-10-20（北京时间）。镜像文件名写v1.2，但实际封面是“正点原子 ATK-IMU901 角度传感器模块用户手册 V1.0”，共35页；不将其误称为已经确认的最新v1.2手册。封面已渲染检查，原始文件没有修改。
- 来源是公开镜像，非从厂商网盘直接取得；版次与实际十轴固件的一致性仍须上机核对。

`manual_text.txt`是完整只读提取，`cover.png`是封面渲染。接入代码根据第6章的逐字节表定义解析：`55 55 ID N DATA SUM8`，小端有符号数据，校验按实际头部N计算。手册部分文字有明显笔误（小端示例后写成高字节在前、ID4数据长度8而校验示例写6等），不照抄矛盾的校验例句。

气压/海拔ID5整帧不进入策略，ID4磁力计同时携带的温度保留。磁力计只证实输出有符号原始计数，没有取得每计数多少μT的校准证据；代码和模型接口使用counts并明确缩放。D0–D3扩展端口未确认接有传感器，保留诊断解析而不补零伪装为额外物理测量。

串口加速度与陀螺仪换算依赖设备已配置的量程；手册默认4g、2000°/s不等于实机已经核实。接收配置必须记录实际量程与安装坐标变换。资料本身不是修改设备配置或进行任何硬件操作的授权。
