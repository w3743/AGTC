# 仓库工作流

| 文件 | 用途 |
| --- | --- |
| `repository-checks.yml` | 对目标为 `main` 的拉取请求运行原生训练 CPU 回归、空白与 Git/LFS 文件检查、子模块元数据检查和凭据扫描。 |
| `../scripts/repository-self-check.py` | 本地与定期自检入口：空白、Git/LFS 大文件策略、Python/JSON 语法、失效机器路径、Markdown 相对链接和原生训练 CPU 回归；只读运行，不修改仓库。 |
| `../scripts/run-repository-self-check.ps1` | Windows 包装脚本，自动定位 git 与 Python 后调用自检，可带 `-Range`、`-SkipNative`、`-SkipLinks`、`-Verbose`。 |
| `README.md` | 本目录文件说明。 |

## 本地自检

```powershell
powershell -ExecutionPolicy Bypass -File .github/scripts/run-repository-self-check.ps1
```

默认比较 `origin/main..HEAD`；退出码 0 表示无硬性失败，1 表示存在失败项，警告不影响退出码。定期执行同一命令即可持续发现格式、引用和语法类错误。
