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
| 通信 | [通信入口](通信/README.md) | 测距、双向数据和板端通信材料 |

## 根目录文件

| 文件 | 用途 |
| --- | --- |
| `README.md` | 工程入口、目录导航和获取方法。 |
| `AGENTS.md` | 项目协作与 Git 工作约定。 |
| `项目协作.md` | 项目协作与工程状态相关说明。 |
| `.gitattributes` | Git LFS、文本和原始证据空白处理规则。 |
| `.gitignore` | 不纳入版本管理的本地文件规则。 |
| `.gitleaksignore` | 历史凭据扫描忽略指纹；PR 检查使用独立扫描策略。 |
| `.gitmodules` | 第三方 Git 子模块的路径和来源。 |

`.github/` 保存 GitHub 工作流配置，说明见[工作流目录](.github/workflows/README.md)。

## 获取工程

安装 Git 和 Git LFS，并确认具有仓库访问权限。Windows 克隆需启用长路径并初始化第三方 submodule：

```powershell
git lfs install
git -c core.longpaths=true clone --config core.longpaths=true --recurse-submodules https://github.com/w3743/AGTC.git
cd AGTC
git lfs pull
```

CAD、模型等大型二进制由 Git LFS 管理；第三方源码通过 submodule 固定版本。克隆后按各模块说明准备运行环境。
