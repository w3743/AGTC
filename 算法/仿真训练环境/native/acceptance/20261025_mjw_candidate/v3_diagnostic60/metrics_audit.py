import json, math
from pathlib import Path
run = Path('/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/acceptance/20261025_v3_diagnostic60/run')
def finite_tree(value):
    if isinstance(value, float): return math.isfinite(value)
    if isinstance(value, dict): return all(finite_tree(v) for v in value.values())
    if isinstance(value, list): return all(finite_tree(v) for v in value)
    return True
rows = [json.loads(line) for line in (run/'metrics.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
summary = json.loads((run/'summary.json').read_text(encoding='utf-8'))
contract = json.loads((run/'observation_contract.json').read_text(encoding='utf-8'))
identity = json.loads((run/'identity.json').read_text(encoding='utf-8'))
nonfinite_path = run/'nonfinite_events.jsonl'
nonfinite_count = sum(1 for line in nonfinite_path.read_text(encoding='utf-8').splitlines() if line.strip()) if nonfinite_path.exists() else 0
postrun = json.loads((run/'diagnostics'/'postrun_audit.json').read_text(encoding='utf-8'))
result = {
  'schema':'native-v3-diagnostic-metrics-audit-v1',
  'metrics_rows':len(rows), 'iterations': [row.get('iter') for row in rows],
  'iterations_complete_1_to_60': [row.get('iter') for row in rows] == list(range(1,61)),
  'samples_per_iteration_values': sorted(set(row.get('samples') for row in rows)),
  'all_2048_samples_per_iteration': all(row.get('samples') == 2048 for row in rows),
  'all_metric_json_finite': all(finite_tree(row) for row in rows),
  'max_nonfinite_env_resets': max((row.get('nonfinite_env_resets',0) for row in rows), default=None),
  'max_nonfinite_actions': max((row.get('nonfinite_actions',0) for row in rows), default=None),
  'nonfinite_event_rows': nonfinite_count,
  'failure_json_present': (run/'failure.json').exists(),
  'diagnostic_nonfinite_event_present': (run/'diagnostics'/'nonfinite_event.json').exists(),
  'summary_iterations': summary.get('iterations'), 'summary_total_wall_s_diagnostic_only': summary.get('total_wall_s'),
  'exit_code': int((run/'exit_code.txt').read_text().strip()),
  'wrapper_exit_code': json.loads((run/'diagnostics'/'wrapper_exit.json').read_text()).get('exit_code'),
  'contract_task_identity_sha256': contract.get('task_identity_sha256'),
  'identity_task_identity_sha256': identity.get('task_identity_sha256'),
  'all_metrics_contract_sha_matches': all(row.get('observation_contract_sha256') == contract.get('task_identity_sha256') for row in rows),
  'source_manifest_stable': postrun.get('source_manifest_stable'),
  'diagnostic_wrapper_stable': postrun.get('diagnostic_wrapper_stable'),
  'interpretation':'diagnostic reproduction attempt only; throughput and training outcome are not benchmark or capability acceptance',
}
result['complete_clean_diagnostic_run'] = bool(result['iterations_complete_1_to_60'] and result['all_2048_samples_per_iteration'] and result['all_metric_json_finite'] and result['max_nonfinite_env_resets']==0 and result['max_nonfinite_actions']==0 and nonfinite_count==0 and not result['failure_json_present'] and not result['diagnostic_nonfinite_event_present'] and result['summary_iterations']==60 and result['exit_code']==0 and result['wrapper_exit_code']==0 and result['all_metrics_contract_sha_matches'] and result['source_manifest_stable'] and result['diagnostic_wrapper_stable'])
(run/'diagnostics'/'metrics_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(result,ensure_ascii=False))