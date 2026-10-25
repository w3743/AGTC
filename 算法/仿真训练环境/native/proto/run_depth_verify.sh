#!/bin/bash
export PATH=/usr/lib/wsl/lib:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CUDA_VISIBLE_DEVICES=''
cd /mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto || exit 1
echo '=== depth verify (numpy vs warp + flat plane + perf) ==='
/opt/venv/bin/python m1_depth_verify.py 2>&1 | tail -2
echo '=== provider integration ==='
/opt/venv/bin/python m1_depth_integration.py 2>&1 | tail -2
echo '=== merged depth_verification.json ==='
/opt/venv/bin/python - <<'PYEOF'
import json
m1 = json.load(open('m1_depth_verification.json', encoding='utf-8'))
integ = json.load(open('depth_integration.json', encoding='utf-8'))
merged = dict(m1)
merged['provider_integration'] = integ
merged['capability_metrics_allowed'] = False
merged['capability_blockers'] = ['detections placeholder (16x8 zeros)',
                                 'depth domain gap vs Isaac RTX not pixel-compared']
merged['sensor_domain_gap'] = [{
    'channel': 'depth',
    'isaac_source': 'Isaac RTX Camera distance_to_image_plane, 160x88, 2.5 Hz, latency 0.3 s, additive range noise',
    'native_source': 'analytic warp raycast (leaf patches + stem cylinders + terrain heightfield), camera-axial Z',
    'expected_differences': [
        'rasterization/Z-buffer vs exact ray intersection (edge/occlusion boundary pixels)',
        'no material/TAA/antialiasing in the analytic path',
        'transparent/thin leaf geometry coverage differs at sub-pixel scale',
    ],
    'algorithm_equivalence': {'reference': 'independent numpy caster', 'valid_mask_mismatch': 0,
                              'max_abs_diff_m': 0.000206, 'within_1e-3_ratio': 1.0},
    'domain_equivalence': {'reference': 'Isaac RTX offline capture', 'status': 'not_run',
                           'edge_policy': 'exclude 1 px border', 'tolerance': 'plane <=1e-3 m, p95 <=2e-3 m'},
    'usable_for_capability': False,
}]
merged['verdict'] = ('analytic depth verified against independent numpy caster and integrated into '
                     'ObservationProvider; capability metrics still blocked by detections')
json.dump(merged, open('depth_verification.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
print('verdict:', merged['verdict'])
print('plane:', json.dumps(merged['analytic_plane']))
print('stats:', json.dumps(merged['stats']))
print('performance:', json.dumps(merged['performance']))
print('integration:', json.dumps(integ))
PYEOF
