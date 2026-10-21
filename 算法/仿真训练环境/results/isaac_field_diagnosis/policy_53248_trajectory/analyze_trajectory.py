import json, math
from pathlib import Path

root = Path.cwd()
base = root / 'results/isaac_field_diagnosis/policy_53248_trajectory'
report = json.loads((base / 'evaluation.json').read_text(encoding='utf-8'))
protocol = json.loads((base / 'protocol.json').read_text(encoding='utf-8'))
tolerance = protocol['environment']['goal_tolerance']

def mean(xs):
    return sum(xs) / len(xs) if xs else None

def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

summaries = []
for row in report['rows']:
    tr = row['trajectory']
    first, last = tr[0], tr[-1]
    d0 = first['before']['goal_distance_m']
    b0 = first['before']['goal_bearing_rad']
    df = last['after']['goal_distance_m']
    bf = last['after']['goal_bearing_rad']
    samples = [first['before']] + [p['after'] for p in tr]
    min_point = min(samples, key=lambda p: p['goal_distance_m'])
    yaw0 = first['before']['pose_xy_yaw'][2]
    yaw_prev = yaw0
    yaw_delta = 0.0
    for p in tr:
        yaw = p['after']['pose_xy_yaw'][2]
        yaw_delta += wrap(yaw - yaw_prev)
        yaw_prev = yaw
    actions_f = [p['action_4'][0] for p in tr]
    actions_d = [p['action_4'][1] for p in tr]
    cmds_v = [p['command_v_w'][0] for p in tr]
    cmds_w = [p['command_v_w'][1] for p in tr]
    actual_v = [p['after']['actual_v_w'][0] for p in tr]
    actual_w = [p['after']['actual_v_w'][1] for p in tr]
    track_l = [p['track_command_lr_mps'][0] for p in tr]
    track_r = [p['track_command_lr_mps'][1] for p in tr]
    holds = [p['after']['hold_s'] for p in tr]
    inside = [p for p in samples if p['goal_distance_m'] <= tolerance]
    damage = [p['after']['damage'] for p in tr]
    collisions = [p['after']['collisions'] for p in tr]
    xyz0 = first['before']['pose_xy_yaw']
    xyzf = last['after']['pose_xy_yaw']
    summary = {
        'seed': row['seed'], 'steps': row['steps'], 'reason': row['reason'],
        'start_distance_m': d0, 'start_bearing_rad': b0,
        'min_distance_m': min_point['goal_distance_m'],
        'min_distance_time_s': min_point['time_s'],
        'final_distance_m': df, 'final_bearing_rad': bf,
        'distance_reduction_m': d0 - df,
        'start_yaw_rad': yaw0, 'final_yaw_rad_wrapped': last['after']['pose_xy_yaw'][2],
        'unwrapped_yaw_change_rad': yaw_delta,
        'mean_abs_yaw_rate_rad_s': mean([abs(x) for x in actual_w]),
        'mean_yaw_rate_rad_s': mean(actual_w),
        'peak_abs_yaw_rate_rad_s': max([abs(x) for x in actual_w]),
        'mean_actual_v_m_s': mean(actual_v), 'peak_abs_actual_v_m_s': max([abs(x) for x in actual_v]),
        'mean_forward_action': mean(actions_f),
        'forward_action_positive_steps': sum(x > 1e-3 for x in actions_f),
        'forward_action_negative_steps': sum(x < -1e-3 for x in actions_f),
        'forward_action_near_zero_steps': sum(abs(x) <= 1e-3 for x in actions_f),
        'mean_differential_action': mean(actions_d),
        'differential_positive_steps': sum(x > 1e-3 for x in actions_d),
        'differential_negative_steps': sum(x < -1e-3 for x in actions_d),
        'differential_near_zero_steps': sum(abs(x) <= 1e-3 for x in actions_d),
        'mean_command_v_m_s': mean(cmds_v), 'mean_command_w_rad_s': mean(cmds_w),
        'mean_track_command_left_m_s': mean(track_l), 'mean_track_command_right_m_s': mean(track_r),
        'mean_actual_w_rad_s': mean(actual_w),
        'goal_tolerance_m': tolerance,
        'inside_tolerance_samples': len(inside),
        'first_tolerance_time_s': inside[0]['time_s'] if inside else None,
        'max_hold_s': max(holds),
        'hold_threshold_s': protocol['environment']['goal_hold'],
        'max_damage': max(damage), 'final_damage': damage[-1],
        'max_collisions': max(collisions), 'final_collisions': collisions[-1],
        'net_xy_displacement_m': math.hypot(xyzf[0]-xyz0[0], xyzf[1]-xyz0[1]),
        'position_xy_start': xyz0[:2], 'position_xy_end': xyzf[:2],
    }
    summaries.append(summary)
analysis = {
    'source': 'evaluation.json',
    'method': 'Derived only from saved per-step task trajectories; unwrapped yaw sums wrapped incremental pose differences.',
    'goal_tolerance_m': tolerance,
    'episodes': summaries,
}
(base / 'analysis.json').write_text(json.dumps(analysis, indent=2), encoding='utf-8')
print(json.dumps(analysis, indent=2))
