# 自由转动前导向轮方案归档

用户于2026-10-20要求改为导向轮与同侧负重轮传动，并整理/适当归档。这里保留被替代的自由转动阶段、旧rough准备失败记录与未启动的旧viewer包装器。

移动了6个目录、130个文件，移动前后逐文件SHA-256与字节数完全一致；原始模型、协议、快照、日志、失败退出和成绩未重写。实际结果见 [archive_verification.json](archive_verification.json)，原路径→归档路径及每个文件SHA见 [archive_manifest_before.json](archive_manifest_before.json)。路径映射保留原有results目录层级。

旧原始记录内的绝对路径维持原字节；查阅时按清单映射到本目录，恢复时可按original_relative回到原路径，须先确认目标未被占用。不要把自由轮接触探针作为新联动方案的结果。旧rough目录只prepare过，preflight因tuple/list合同比较失败，未创建模型或训练；它不是已训练实验。

当前方案与验收入口见项目根目录ISAAC_LAB.md和 results/project_history/front_guide_coupled_cleanup_20261020/README.md。资产缓存与共享运行时保留原位，旧正式训练模型及其协议/成绩也保留原位。
