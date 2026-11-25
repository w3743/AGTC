# 贡献指南

感谢关注并参与 **AGTC（先进通用履带底盘工程）**。提交贡献前，请先阅读以下约定，以保证仓库结构整洁、硬件资料可追溯且自动检查顺利通过。

## 1. 克隆仓库与环境准备

仓库包含 CAD/模型等 Git LFS 大文件及第三方源码子模块，克隆时请启用 LFS 与长路径支持（Windows 推荐如下命令）：

```powershell
git lfs install
git -c core.longpaths=true clone --config core.longpaths=true --recurse-submodules https://github.com/w3743/AGTC.git
cd AGTC
git lfs pull
```

如仅需拉取后续更新：

```powershell
git pull
git submodule update --init --recursive
git lfs pull
```

## 2. 分支与 Pull Request 流程

- `main` 为受保护分支，禁止直接向 `main` 推送提交。
- 请从最新的 `main` 创建功能或修复分支（例如 `feat/...`、`fix/...`、`chore/...`）。
- 完成修改并在本地通过自检后推送分支，参照 [.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md) 创建 Pull Request。
- PR 必须通过必需 CI 检查 **`Repository checks`** 后才可合并。

## 3. 提交前本地自检

提交及推送前，请在仓库根目录执行以下检查，确保与 CI 规则一致：

```powershell
python -X utf8 -B .github/scripts/repository-self-check.py --range origin/main..HEAD
git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --check origin/main HEAD
```

同时注意以下基础规范：

- 文本文件统一使用 **UTF-8（无 BOM）** 与 **LF（`\n`）** 换行，避免行尾空格和文件末尾多余空行。
- 非 Git LFS 跟踪的单个文件不得超过 **10 MiB**。
- 严禁提交真实或疑似的密钥、Token、私钥、口令等敏感凭据（CI 会运行 `gitleaks` 扫描）。

## 4. 工程与模块约束

- **车载软件语言与性能要求**：凡涉及随车运行的程序（主控、感知、定位、规划、推理、通信、驱动、监控及板卡固件/启动程序等），须严格遵守 [AGENTS.md](AGENTS.md)，优先采用 **C、C++ 或 Rust** 等编译型高性能语言，不得新增 Python 或脚本胶水作为车端运行时，并附上对应基线与回归验证说明。
- **电力硬件文档与冻结资产**：电力模块每块板卡只维护 `设计需求.md` 与 `设计细节.md` 两份现行设计文档，统一入口见[电力工程](电力/README.md)。请勿擅自改动已冻结板卡文件及联合仿真/采购归档清单。
- **文档同步**：新增、重命名或删除顶层文件及模块入口时，请同步更新根目录 [README.md](README.md) 或对应模块文档，并确保相对链接有效。

## 5. 问题反馈

- 缺陷报告、功能建议与方案讨论请优先通过仓库 Issue 表单提交，见 [.github/ISSUE_TEMPLATE/](.github/ISSUE_TEMPLATE/)。
- 安全隐患或漏洞请遵循 [SECURITY.md](SECURITY.md) 私密上报，勿在公开 Issue 披露高风险细节。

## 6. 行为守则与开源协议

- 参与本项目的所有贡献者与讨论者均需遵守 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。
- 本仓库采用 **GNU General Public License v3.0（GPL-3.0）** 开源，全文见 [LICENSE](LICENSE)。向本仓库提交贡献即表示您同意您的贡献按相同协议对外发布。
