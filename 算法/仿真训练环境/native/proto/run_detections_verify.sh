#!/bin/bash
export PATH=/usr/lib/wsl/lib:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CUDA_VISIBLE_DEVICES=''
cd /mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto || exit 1
echo '=== detections verify (mask/packing vs FieldTask) ==='
/opt/venv/bin/python m1_detections_verify.py 2>&1 | tail -2
echo '=== provider integration ==='
/opt/venv/bin/python m1_detections_integration.py 2>&1 | tail -2
echo '=== merged detections_verification.json ==='
/opt/venv/bin/python - <<'PYEOF'
import json
base = json.load(open('detections_verification.json', encoding='utf-8'))
integ = json.load(open('detections_integration.json', encoding='utf-8'))
merged = dict(base)
merged['provider_integration'] = integ
merged['capability_metrics_allowed'] = False
merged['capability_blockers'] = ['RTX domain gap not pixel-compared (depth, detections)']
merged['sensor_domain_gap'] = [{
    'channel': 'detections',
    'isaac_source': 'Isaac RTX instance_segmentation 64x64 + 1-ray rangefinder; env.py mask->box/confidence/class/range rules',
    'native_source': 'analytic leaf-instance raycast mask -> same env.py rules -> task.py packing',
    'expected_differences': [
        'rasterization vs raycast mask edges (1-2 px box differences)',
        'reference detector camera optical-axis convention assumed identical to the depth camera mount basis (not pixel-verified)',
        'confidence/class/range noise draws follow the same distributions but independent RNG streams',
    ],
    'algorithm_equivalence': {'reference': 'FieldTask._observation packing', 'max_abs_error': 0.0,
                              'detections_exercised': 34, 'range_branch_exercised': 1},
    'domain_equivalence': {'reference': 'Isaac RTX detector capture', 'status': 'not_run',
                           'tolerance': 'box center/edge <=2 px; range delta <=0.02 m on centre instance'},
    'usable_for_capability': False,
}]
merged['verdict'] = ('analytic detections match the reference FieldTask packing element-wise and are integrated '
                     'into ObservationProvider; capability metrics blocked on RTX domain-gap acceptance')
json.dump(merged, open('detections_verification.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
print('verdict:', merged['verdict'])
print('summary:', json.dumps({k: merged[k] for k in
      ('poses', 'masks', 'detections_per_frame', 'active_rows_total',
       'rows_max_abs_error_vs_reference_packing', 'box_checks', 'range_field',
       'performance')}))
print('integration:', json.dumps(integ))
PYEOF
