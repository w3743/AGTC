# 后续解释：overflow计数与角度分支

原始JSON、NPZ、日志、源码副本和copy_manifest不改。

MuJoCo Warp3.15源码对`Data.overflow`执行OR置位，`step/forward`不自动清除。旧探针中的4999/4999/4699或849/849/299是flag-positive采样数，不是独立线搜索耗尽次数；首次置位后可能持续被重复读到。容量位与outer-iteration位仍须分别解读，不能把LS位当接触缓冲截断。仓库实现将每个物理子步求解前在设备清零，求解后立刻把本步flags累计到设备buffer，只在策略边界回读，避免漏掉瞬时截断或重造逐子步主机同步。

原始eval_ik裸角差可跨4π分支，不能直接称物理mimic漂移。v4b保存的真实continuous state joint q与MJWarp qpos一致，mimic最大raw残差约0.000169rad；它不能补造旧v3缺失的逐对坐标。

本文件在归档后新增，不属于原copy_manifest列出的复制项。最新平台状态、最终数据与边界见`native/MIGRATION_ACCEPTANCE.md`。
