# Actual contact-probe follow-up (2026-10-20)

The formerly prepared-only probe was run once after the fresh training runner exited 0. See [the actual probe analysis](contact_probe_analysis.md), [metrics JSON](contact_probe_metrics.json), and [preserved raw output](contact_probe_runs/diff_left_plus_0.10_20261020/contact_transfer_probe_summary.json). It captured 180 policy samples and 1,800 effort substeps; actual exit was 0 and the 2 Nm cap/finite checks passed.

The last-second per-wheel aggregate upward normal force averaged **293.1 N**, close to the 30 kg vehicle weight of 294.3 N, while per-wheel readings were uneven and included zero samples. Side-mean wheel qd tracked the ±9.259 rad/s targets, but measured body yaw rate was only about **3.6%** of the nominal no-slip yaw command; the root cause remains undetermined. `net_friction_forces_w` is documented as world-frame, but the complete solver-to-sensor force-frame chain was not verified, so the forward values are aggregate API-field projections rather than definitive per-pair traction.

The original prepared-only report follows unchanged below and records the pre-run analysis state.
# Side-PI confirmation: wheel-to-body yaw gap

## Scope and evidence

This is a read-only review of the frozen local vehicle/environment source and the already completed `confirmation_p` / `confirmation_pi` raw files. It does not access 830xx data, launch the simulator, run an evaluation, or modify production/training sources. The pure-Python calculation helper completed with exit 0 and wrote [`evidence_metrics_v3.json`](evidence_metrics_v3.json). The paired confirmation protocol SHA is `1236e551140423afd21749fd31a9431436fbf810d47f1331a83d36e0960fc8d0`; the two runs have identical 41-source hash maps and identical v5 base-drive config. Every raw policy and substep-effort record is finite, and neither run exceeds the per-wheel 2 Nm native cap.

Primary raw inputs:

- `results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual/confirmation_protocol.json`
- `results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual/confirmation_p/{counterfactual_summary.json,policy_step_samples.jsonl,physics_substep_effort.jsonl}`
- `results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual/confirmation_pi/{counterfactual_summary.json,policy_step_samples.jsonl,physics_substep_effort.jsonl}`

These are runtime P-only and runtime PI counterfactuals over the same v5 environment; the PI run uses uncalibrated `Ki=0.1` and is not a trained policy or a hardware controller result.

## What the source models

`isaac_field/assets.py:32-103,186-198,283-306` defines a 30 kg assembly: 26.4 kg chassis, fourteen 0.25 kg rollers, and two 0.05 kg gimbal links. The track is represented by seven separate 45 mm radius, 60 mm wide Y-axis cylinders on each side, spaced over 595 mm, with revolute joints also on Y. The centers are 410 mm apart. There is no continuous belt; each proxy roller has its own driven joint.

`isaac_field/env.py:75-87,163-171` configures Newton MJWarp at 5 ms physics steps, ten physics steps per 50 ms policy step, explicit per-wheel IdealPD with `Kp=0.04`, and 2 Nm actuator and joint effort limits. The shared Newton shape configuration sets `mu=0.8`, `margin=0.002 m`, and `gap=0.005 m`; the local track/terrain meshes do not author separate calibrated friction materials. These are simulation assumptions, not measured soil or rubber properties.

`isaac_field/env.py:389-410` converts each side's track-speed target to roller speed as `qd_target = track_command / 0.045`. Its `command[:,1] = (right-left)/0.41` is the ideal no-slip kinematic yaw rate retained for diagnostics; comments explicitly say task state and reward use actual chassis velocity from physics. Thus the nominal `v,w` pair is not a body-yaw servo target.

During the confirmation reset, `root_z` is about 0.15396 m. With the roller center at −0.105 m and radius 0.045 m, the ideal roller bottom is about 3.96 mm above a zero-height flat plane. That is within the configured 5 mm contact gap but is not geometric penetration. This makes actual per-wheel contact force and separation useful measurements; the gap value alone does not prove missing contact.

## Confirmation results

The values below come from the final one-second raw drive samples. The no-slip wheel-derived yaw is `(mean(right_qd) - mean(left_qd)) * 0.045 / 0.41`.

| Case/controller | Nominal yaw (rad/s) | Actual body yaw (rad/s) | Body / nominal | Wheel-derived no-slip yaw (rad/s) | Body / wheel-derived | Mean side qd (left, right; rad/s) | Max substep effort |
|---|---:|---:|---:|---:|---:|---:|---:|
| diff 0.05, capped P | −1.0163 | −0.00144 | 0.14% | −0.00280 | 51.5% | 0.0128, −0.0127 | 0.185 Nm |
| diff 0.05, PI candidate | −1.0163 | −0.03901 | 3.84% | −0.8300 | 4.70% | 3.528, −4.034 | 0.670 Nm |
| diff 0.10, capped P | −2.0325 | −0.00363 | 0.18% | −0.00680 | 53.3% | 0.0308, −0.0311 | 0.370 Nm |
| diff 0.10, PI candidate | −2.0325 | −0.07475 | 3.68% | −2.0497 | 3.65% | 9.094, −9.581 | 0.884 Nm |
| diff 0.80 brake case, capped P | −16.2602 | −0.81795 | 5.03% | −15.0908 | 5.42% | 70.768, −66.726 | 2.000 Nm |
| diff 0.80 brake case, PI candidate | −16.2602 | −0.88155 | 5.42% | −15.1511 | 5.82% | 66.704, −71.339 | 2.000 Nm |

For the 0.05 / 0.10 commands, target roller speeds are ±4.63 / ±9.26 rad/s. P-only side means remain near zero despite nonzero unsaturated efforts. The PI candidate improves roller speed tracking substantially, bringing the 0.10 case close to its ±9.26 rad/s targets, but body yaw remains only 3.7% of the nominal rate and about 3.6% of the yaw implied by the measured roller speeds. The sign is correct in both cases. At the 0.80 command both controllers reach the 2 Nm cap and body yaw remains about 5% of the command; therefore low-command torque saturation alone does not explain the yaw gap.

The five-second braking portions ended with exact zero track commands and passed the predeclared low-speed tail gate in both runs. This is a stop-gate result only, not a navigation or goal-completion result.

## Torque and traction scale

The following are idealized scale calculations, not measured contact loads or motor output:

- 30 kg gives about 294.3 N weight. If all fourteen rollers shared the normal load equally, that is 21.0 N each. With the configured `mu=0.8`, the corresponding Coulomb limit is about 16.8 N per roller, or 235 N total. The equal-load assumption is not verified.
- A 2 Nm roller effort corresponds to `2 / 0.045 = 44.4 N` tangential force per roller if the entire effort becomes ground traction. That is 311 N per side, 622 N in both directions, about 2.64 times the simple `mu*m*g` bound. Under that idealized distribution, the cap allows wheel slip rather than enforcing a traction limit.
- At the low PI cases, the final one-second mean per-roller efforts are approximately +0.424/−0.494 Nm for the 0.05 command and +0.516/−0.409 Nm for 0.10. Their force equivalents are below the equal-load 0.757 Nm/roller friction-torque estimate, yet the body/wheel yaw ratio remains near four percent. Actual normal loads and friction forces were not captured in these confirmation files, so this is a clue, not proof of the cause.
- Summing seven 2 Nm proxy rollers gives 14 Nm per side at the roller shafts. With the declared 45 mm proxy radius, 30 mm active sprocket radius, and 4:1 reduction, the idealized motor-side equivalent is `14 * 0.03 / (0.045 * 4) = 2.33 Nm` per motor. The model runs fourteen per-wheel PD actuators, not two motor actuators driving continuous belts; the metadata correctly labels this as proxy-equivalent, not measured motor torque/current.

## Existing related diagnostics

The earlier `results/isaac_field_diagnosis/ground_contact/ground_contact_tail_pose_summary_20261017.json` and its referenced A/B/C/D raw files already exercised the flat mesh, an added plane, lower/higher PD gains, and the alternate contact flag. They recorded roughly 0.50–0.53 m/s forward motion. The contact bookkeeping differed across cases: the flat-mesh case A reported zero wheel-ground pairs while selected-wheel contact sensors reported up to about 92.9 N; cases B/C/D reported 28/46/28 pairs. Those runs sampled only two wheels and used different actions/controller era. They show why contact labels alone are insufficient, and they do not reproduce the exact low differential commands here.

The older `results/isaac_field_diagnosis/skid_response/wheel_response.json` also found weak in-place turn response (for example, a −1.05 rad/s nominal turn with about −0.00343 rad/s final body yaw and ±0.75 rad/s mean roller speeds). It used the older forward/turn action semantics and 0.25 m track separation, so it is corroboration that turning response deserves attention, not a paired comparison to the present 0.41 m v5 confirmation.

## Evidence versus hypotheses

**Established by source and raw data:** the nominal yaw is computed from track-speed commands; wheel targets use the 45 mm radius; the proxy consists of fourteen independently actuated cylinders; `mu=0.8` is configured globally; the low-command PI runs do not hit the 2 Nm cap; their measured roller speeds imply much more no-slip yaw than the chassis actually exhibits; and the raw P/PI source and base-drive snapshots match.

**Leading physical hypotheses, not yet proved:**

1. The proxy rollers are slipping longitudinally or failing to transmit their angular motion into chassis yaw. The large difference between wheel-derived and measured body yaw in PI cases is direct evidence of a kinematic/traction mismatch, but contact force was not recorded, so its mechanism is unresolved.
2. The continuous track is represented as seven separate, independently driven rollers per side. This omits belt tension, distributed belt contact, and one-motor-per-side torque coupling; it can make both yaw and saturation response unlike the real vehicle.
3. The wheel-bottom position is close to the configured contact gap. Actual contact may be sensitive to the mesh/gap/solver path or normal-load distribution. Earlier contact diagnostics show mixed pair-count reporting, so the current raw does not settle this.
4. A speed PI can improve roller qd without fixing body yaw. The confirmation demonstrates that distinction; changing the training curriculum before measuring contact-force transfer could teach the policy around an incorrect or weak action-to-body model.

The evidence does **not** establish that the detector, policy architecture, or training curriculum caused the yaw shortfall. The 0.1 Ki value is uncalibrated and the counterfactual is not a real-hardware controller.

## Smallest next falsifiable probe (prepared only; do not run without authorization)

Reuse the existing contact-sampling pattern from `scripts/diagnose_isaac_ground_contact.py`, but run only the confirmation `diff_left_plus_0.10` sequence in one environment, with the same flat plants-free start and current frozen drive/physics config. Add all fourteen wheel contacts (not only the earlier two selected wheels) and record per policy step: wheel-ground pair identity, each wheel's normal and longitudinal contact force, wheel-center-to-ground separation, wheel qd, body `v/w`, track command/target qd, and the existing post-actuator per-substep effort tape. Do not change `mu`, `Kp`, `Ki`, effort caps, solver, wheel height, task threshold, or training source.

Predeclare the decision:

- If the rollers do not carry a stable share of body weight or show few/unstable ground contacts while qd tracks target, investigate the wheel/ground gap and collider path first.
- If stable normal loads exist but longitudinal contact force is far below the torque-equivalent wheel force while qd spins near target, that supports slip or inadequate contact friction in the proxy.
- If measured contact forces approach the expected side force and their moment about the body still does not match measured angular acceleration, audit torque/joint-axis and force aggregation conventions; do not retune the policy.

Only if the instrumented baseline leaves the contact path ambiguous, make one paired, test-only contact-path variant (same initial state, action, and controller) using the already explored plane-vs-mesh or solver-contact switch. This keeps the first probe small and makes the next decision depend on measured contact transfer instead of another broad controller sweep.
