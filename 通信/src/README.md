# 通信源码

本目录组织 sample 本体和将它接入海思 SDK 所需的补丁：

| 子目录 | 用途 |
|---|---|
| [`sle_range_link/`](sle_range_link/README.md) | 本工程维护的 SLE Range Link sample：双板 1D 测距、双向数据、运行时校准和距离过滤。该 README 逐项说明本目录源文件及官方 sample 派生文件。 |
| [`sdk_patches/`](sdk_patches/README.md) | 按 SDK 相对路径保存的四个集成补丁，涉及 sample 菜单/构建注册、BS2x 功能开关和 BT/HADM 任务栈配置。 |

SDK 派生补丁的具体文件和适用边界由 `sdk_patches/README.md` 维护；请结合目标 SDK 版本对照后集成。
