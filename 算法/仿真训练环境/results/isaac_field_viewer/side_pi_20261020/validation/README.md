# Fresh side PI v6 checkpoint validation

The validator loads only serialized `RecurrentPPO` artifacts on CPU. It does not create an environment, invoke simulation, run evaluation seeds, or train. It checks the zero-step baseline and, when supplied, the 8192-step checkpoint against the v6 observation and encoder contracts, PI v6 drive metadata, checkpoint manifest, protocol SHA, the 41 frozen-source hashes, finite policy/Adam state, and baseline-to-checkpoint branch changes. The detection branch is reported separately because this run's `plants=0` training may provide no positive detector inputs. A protected-hash reference for the prior `fresh_minimal_inputs_20261020` trial is stored here; its evaluation trajectory JSON files were intentionally excluded.

Run the baseline-only check with an explicit trial path and expected protocol digest:

```bash
cd "/mnt/e/Pioneer/先进通用履带底盘工程/算法/仿真训练环境"
bash results/isaac_field_viewer/side_pi_20261020/validation/run_checkpoint_validation.sh \
  "$PWD/results/isaac_field_improvement/fresh_side_pi_20261020" \
  "$PWD/results/isaac_field_improvement/fresh_side_pi_20261020/untrained_baseline.zip" \
  - \
  0a356055c2ff08e13133ebfe12c71e90c63612a8efce7ea0db8b1187aafd10b8 \
  bdb944d4dbba038536687810089448aecb132d07979dd5493c8a33ebbe9e0129 \
  baseline_only
```

After the 8192-step checkpoint is confirmed present, run the paired check:

```bash
bash results/isaac_field_viewer/side_pi_20261020/validation/run_checkpoint_validation.sh \
  "$PWD/results/isaac_field_improvement/fresh_side_pi_20261020" \
  "$PWD/results/isaac_field_improvement/fresh_side_pi_20261020/untrained_baseline.zip" \
  "$PWD/results/isaac_field_improvement/fresh_side_pi_20261020/stage_1_near/checkpoints/field_policy_near_8192_steps.zip" \
  0a356055c2ff08e13133ebfe12c71e90c63612a8efce7ea0db8b1187aafd10b8 \
  bdb944d4dbba038536687810089448aecb132d07979dd5493c8a33ebbe9e0129 \
  checkpoint_8192
```

Each invocation refuses to overwrite a prior JSON result, log, or exit record. The first result uses the stem `baseline_only`; the paired result uses `checkpoint_8192`.
