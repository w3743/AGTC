# 仿真训练测试

本目录保存自动化测试和测试辅助工具。名称中的主题对应被检查的行为；具体断言和适用环境以测试源码为准。

| 文件 | 用途 |
| --- | --- |
| `newton_effort_capture.py` | Newton 驱动 effort 数据采集辅助代码。 |
| `policy_contract_helpers.py` | 策略接口测试共用辅助代码。 |
| `test_absolute_goal_cli.py` | 针对 `absolute_goal_cli` 的自动化测试（源码中 15 个 test 函数或方法）。 |
| `test_absolute_goal_observation_schema.py` | 针对 `absolute_goal_observation_schema` 的自动化测试（源码中 7 个 test 函数或方法）。 |
| `test_absolute_goal_policy.py` | 针对 `absolute_goal_policy` 的自动化测试（源码中 8 个 test 函数或方法）。 |
| `test_absolute_goal_training_integration.py` | 针对 `absolute_goal_training_integration` 的自动化测试（源码中 9 个 test 函数或方法）。 |
| `test_absolute_goal_viewer_contract.py` | 针对 `absolute_goal_viewer_contract` 的自动化测试（源码中 3 个 test 函数或方法）。 |
| `test_action_target_persistence.py` | 针对 `action_target_persistence` 的自动化测试（源码中 1 个 test 函数或方法）。 |
| `test_activation_maps.py` | 针对 `activation_maps` 的自动化测试（源码中 7 个 test 函数或方法）。 |
| `test_body_stability.py` | 针对 `body_stability` 的自动化测试（源码中 5 个 test 函数或方法）。 |
| `test_body_stability_recording.py` | 针对 `body_stability_recording` 的自动化测试（源码中 4 个 test 函数或方法）。 |
| `test_continue_foundation.py` | 针对 `continue_foundation` 的自动化测试（源码中 6 个 test 函数或方法）。 |
| `test_curriculum_control.py` | 针对 `curriculum_control` 的自动化测试（源码中 11 个 test 函数或方法）。 |
| `test_drive_control.py` | 针对 `drive_control` 的自动化测试（源码中 14 个 test 函数或方法）。 |
| `test_env_integration.py` | 针对 `env_integration` 的自动化测试（源码中 16 个 test 函数或方法）。 |
| `test_evaluate_rsl_curriculum.py` | 针对 `evaluate_rsl_curriculum` 的自动化测试（源码中 12 个 test 函数或方法）。 |
| `test_fast_stop_mastery_acceptance.py` | 针对 `fast_stop_mastery_acceptance` 的自动化测试（源码中 4 个 test 函数或方法）。 |
| `test_fast_stop_resume.py` | 针对 `fast_stop_resume` 的自动化测试（源码中 10 个 test 函数或方法）。 |
| `test_fast_stop_resume_evaluation.py` | 针对 `fast_stop_resume_evaluation` 的自动化测试（源码中 6 个 test 函数或方法）。 |
| `test_fast_stop_v2_reward.py` | 针对 `fast_stop_v2_reward` 的自动化测试（源码中 7 个 test 函数或方法）。 |
| `test_feedback.py` | 针对 `feedback` 的自动化测试（源码中 9 个 test 函数或方法）。 |
| `test_finished_observation_cache.py` | 针对 `finished_observation_cache` 的自动化测试（源码中 4 个 test 函数或方法）。 |
| `test_fresh_foundation.py` | 针对 `fresh_foundation` 的自动化测试（源码中 29 个 test 函数或方法）。 |
| `test_front_guide_cli.py` | 针对 `front_guide_cli` 的自动化测试（源码中 2 个 test 函数或方法）。 |
| `test_front_guide_native.py` | 针对 `front_guide_native` 的自动化测试（源码中 3 个 test 函数或方法）。 |
| `test_goal_map_viewer.py` | 针对 `goal_map_viewer` 的自动化测试（源码中 13 个 test 函数或方法）。 |
| `test_goal_map_viewer_identity_follow.py` | 针对 `goal_map_viewer_identity_follow` 的自动化测试（源码中 9 个 test 函数或方法）。 |
| `test_isaac_exploration.py` | 针对 `isaac_exploration` 的自动化测试（源码中 3 个 test 函数或方法）。 |
| `test_isaac_field_adversarial_training.py` | 针对 `isaac_field_adversarial_training` 的自动化测试（源码中 1 个 test 函数或方法）。 |
| `test_isaac_field_adversary.py` | 针对 `isaac_field_adversary` 的自动化测试（源码中 2 个 test 函数或方法）。 |
| `test_isaac_field_assets.py` | 针对 `isaac_field_assets` 的自动化测试（源码中 9 个 test 函数或方法）。 |
| `test_isaac_field_cli.py` | 针对 `isaac_field_cli` 的自动化测试（源码中 2 个 test 函数或方法）。 |
| `test_isaac_field_gpu_recovery_pipeline.py` | 针对 `isaac_field_gpu_recovery_pipeline` 的自动化测试（源码中 10 个 test 函数或方法）。 |
| `test_isaac_field_headless_recovery_pipeline.py` | 针对 `isaac_field_headless_recovery_pipeline` 的自动化测试（源码中 11 个 test 函数或方法）。 |
| `test_isaac_field_layouts.py` | 针对 `isaac_field_layouts` 的自动化测试（源码中 4 个 test 函数或方法）。 |
| `test_isaac_field_learning.py` | 针对 `isaac_field_learning` 的自动化测试（源码中 15 个 test 函数或方法）。 |
| `test_isaac_field_live_pipeline.py` | 针对 `isaac_field_live_pipeline` 的自动化测试（源码中 12 个 test 函数或方法）。 |
| `test_isaac_field_mirror_pipeline.py` | 针对 `isaac_field_mirror_pipeline` 的自动化测试（源码中 7 个 test 函数或方法）。 |
| `test_isaac_field_policy.py` | 针对 `isaac_field_policy` 的自动化测试（源码中 16 个 test 函数或方法）。 |
| `test_isaac_field_viewer.py` | 针对 `isaac_field_viewer` 的自动化测试（源码中 25 个 test 函数或方法）。 |
| `test_isaac_foundation_resume.py` | 针对 `isaac_foundation_resume` 的自动化测试（源码中 6 个 test 函数或方法）。 |
| `test_isaac_mastery_runner.py` | 针对 `isaac_mastery_runner` 的自动化测试（源码中 6 个 test 函数或方法）。 |
| `test_isaac_sim_runtime.py` | 针对 `isaac_sim_runtime` 的自动化测试（源码中 10 个 test 函数或方法）。 |
| `test_lassc_consumer.py` | 针对 `lassc_consumer` 的自动化测试（源码中 5 个 test 函数或方法）。 |
| `test_lassc_replay.py` | 针对 `lassc_replay` 的自动化测试（源码中 9 个 test 函数或方法）。 |
| `test_live_state.py` | 针对 `live_state` 的自动化测试（源码中 7 个 test 函数或方法）。 |
| `test_mastery_process_evaluation.py` | 针对 `mastery_process_evaluation` 的自动化测试（源码中 12 个 test 函数或方法）。 |
| `test_mastery_runner.py` | 针对 `mastery_runner` 的自动化测试（源码中 7 个 test 函数或方法）。 |
| `test_mastery_selection.py` | 针对 `mastery_selection` 的自动化测试（源码中 5 个 test 函数或方法）。 |
| `test_minimal_input_contract.py` | 针对 `minimal_input_contract` 的自动化测试（源码中 3 个 test 函数或方法）。 |
| `test_network_2d.py` | 针对 `network_2d` 的自动化测试（源码中 20 个 test 函数或方法）。 |
| `test_network_layout.py` | 针对 `network_layout` 的自动化测试（源码中 6 个 test 函数或方法）。 |
| `test_network_panel.py` | 针对 `network_panel` 的自动化测试（源码中 7 个 test 函数或方法）。 |
| `test_newton_drive.py` | 针对 `newton_drive` 的自动化测试（源码中 3 个 test 函数或方法）。 |
| `test_physx_coupling_contract.py` | 针对 `physx_coupling_contract` 的自动化测试（源码中 3 个 test 函数或方法）。 |
| `test_policy_inspection.py` | 针对 `policy_inspection` 的自动化测试（源码中 9 个 test 函数或方法）。 |
| `test_rsl_native_smoke_evaluation.py` | 针对 `rsl_native_smoke_evaluation` 的自动化测试（源码中 13 个 test 函数或方法）。 |
| `test_rsl_policy_inspection.py` | 针对 `rsl_policy_inspection` 的自动化测试（源码中 3 个 test 函数或方法）。 |
| `test_rsl_rl_config.py` | 针对 `rsl_rl_config` 的自动化测试（源码中 6 个 test 函数或方法）。 |
| `test_rsl_rl_export.py` | 针对 `rsl_rl_export` 的自动化测试（源码中 5 个 test 函数或方法）。 |
| `test_rsl_rl_inference.py` | 针对 `rsl_rl_inference` 的自动化测试（源码中 9 个 test 函数或方法）。 |
| `test_rsl_rl_policy.py` | 针对 `rsl_rl_policy` 的自动化测试（源码中 10 个 test 函数或方法）。 |
| `test_rsl_shared_encoder_boundary.py` | 针对 `rsl_shared_encoder_boundary` 的自动化测试（源码中 2 个 test 函数或方法）。 |
| `test_rsl_viewer_entrypoints.py` | 针对 `rsl_viewer_entrypoints` 的自动化测试（源码中 11 个 test 函数或方法）。 |
| `test_sampling_copy_cache.py` | 针对 `sampling_copy_cache` 的自动化测试（源码中 2 个 test 函数或方法）。 |
| `test_sampling_frozen_equivalence.py` | 针对 `sampling_frozen_equivalence` 的自动化测试（源码中 1 个 test 函数或方法）。 |
| `test_task.py` | 针对 `task` 的自动化测试（源码中 38 个 test 函数或方法）。 |
| `test_telemetry.py` | 针对 `telemetry` 的自动化测试（源码中 12 个 test 函数或方法）。 |
| `test_tensor_env_integration.py` | 针对 `tensor_env_integration` 的自动化测试（源码中 4 个 test 函数或方法）。 |
| `test_tensor_sensors.py` | 针对 `tensor_sensors` 的自动化测试（源码中 9 个 test 函数或方法）。 |
| `test_tensor_task.py` | 针对 `tensor_task` 的自动化测试（源码中 7 个 test 函数或方法）。 |
| `test_training_completion_watch.py` | 针对 `training_completion_watch` 的自动化测试（源码中 14 个 test 函数或方法）。 |
| `test_training_diagnostics.py` | 针对 `training_diagnostics` 的自动化测试（源码中 10 个 test 函数或方法）。 |
| `test_training_diagnostics_native.py` | 针对 `training_diagnostics_native` 的自动化测试（源码中 1 个 test 函数或方法）。 |
| `test_training_observability.py` | 针对 `training_observability` 的自动化测试（源码中 5 个 test 函数或方法）。 |
| `test_training_state_viewer.py` | 针对 `training_state_viewer` 的自动化测试（源码中 6 个 test 函数或方法）。 |
| `test_view_training_state_web.py` | 针对 `view_training_state_web` 的自动化测试（源码中 18 个 test 函数或方法）。 |
| `README.md` | 本目录文件索引。 |

训练或硬件测试可能需要对应依赖和设备；本索引不代表测试已执行。
