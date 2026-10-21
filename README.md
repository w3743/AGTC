# 先进通用履带底盘工程

AGTC（Advanced General-purpose Tracked Chassis）面向户外应用，围绕履带底盘开展感知、定位、行驶与云台联合规划及电控研发。
算法、硬件和机械资料集中在本仓库维护，模块入口如下。
具体设计与操作要求以各模块文档为准。

## 目录入口

| 模块 | 入口 | 内容 |
|---|---|---|
| 算法 | [算法入口](算法/README.md) | 仿真、视觉、定位与 RK3588 |
| 电力 | [电力工程](电力/README.md) | 电池、电驱、供电与控制 |
| 器件库 | [元件封装库](电力/元件封装库/README.md) | 全项目器件与封装资料 |
| 机械 | [参数基线](机械/参数基线.json) · [车体装配](机械/车.SLDASM) | 机械参数与整车 CAD |

## 获取工程

安装 Git 和 Git LFS，并确认具有仓库访问权限。Windows 克隆需启用长路径并初始化第三方 submodule：

```powershell
git lfs install
git -c core.longpaths=true clone --config core.longpaths=true --recurse-submodules https://github.com/w3743/AGTC.git
cd AGTC
git lfs pull
```

CAD、模型等大型二进制由 Git LFS 管理；第三方源码通过 submodule 固定版本。克隆后按各模块说明准备运行环境。

## 关键约束与文档

- 硬件约束：保留现有相机与 RK3588；超容固定为 8 颗 LICAP SC0350-270-RSS（8S1P），用于启动瞬态补能和母线支撑。
- 项目进展与分工统一记录于[项目协作](项目协作.md)；完整技术要求见[电力基本设计](电力/基本设计.md)及各模块文档；贡献和 Git 约定见[AGENTS.md](AGENTS.md)。
- 项目仍处于开发与验证阶段；模型结果或 DRC 记录不等同于实车验收。
