"""Board-only inference benchmark; never opens cameras or modifies the live service."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from rknnlite.api import RKNNLite
import rknnlite.api.rknn_lite as lite
from rknnlite.api.rknn_runtime import RKNNRuntime


class IsolatedRuntime(RKNNRuntime):
    def _get_rknn_api_lib_path(self):
        return '/opt/lightstereo/runtime/librknnrt.so'


lite.RKNNRuntime = IsolatedRuntime

p = argparse.ArgumentParser()
p.add_argument('--model', default='/opt/lightstereo/lightstereo_256_512.rknn')
p.add_argument('--reference', default='/opt/lightstereo/reference.npz')
p.add_argument('--output', default='/opt/lightstereo/board-report.json')
p.add_argument('--runs', type=int, default=12)
p.add_argument('--cores', type=int, choices=[1, 7], default=7)
a = p.parse_args()
data = np.load(a.reference, allow_pickle=False)
r = RKNNLite(verbose=False)
try:
    assert r.load_rknn(a.model) == 0, 'model load failed'
    assert r.init_runtime(core_mask=a.cores) == 0, 'NPU init failed'
    maps = Path('/proc/self/maps').read_text()
    assert '/opt/lightstereo/runtime/librknnrt.so' in maps, 'isolated runtime not loaded'
    assert '/usr/lib/librknnrt.so' not in maps, 'unexpected system runtime loaded'
    images = [np.ascontiguousarray(data[k]) for k in ('left', 'right')]
    elapsed = []
    for i in range(a.runs+2):
        started = time.perf_counter()
        out = r.inference(inputs=images, data_format=['nhwc', 'nhwc'])
        assert out is not None and np.isfinite(out[0]).all(), 'invalid output'
        if i >= 2:
            elapsed.append((time.perf_counter()-started)*1000)
    prediction = np.asarray(out[0]).squeeze()
    expected = data['disparity'].squeeze()
    assert prediction.shape == expected.shape
    delta = np.abs(prediction-expected)
    width = int(data['content_width']) if 'content_width' in data else prediction.shape[1]
    top = int(data['pad_top']) if 'pad_top' in data else 0
    region = delta[top:, :width]
    report = dict(model=Path(a.model).name, runtime='2.3.2', driver='0.9.6',
                  runtime_path='/opt/lightstereo/runtime/librknnrt.so',
                  shape=list(prediction.shape), content_width=width, core_mask=a.cores,
                  reference=Path(a.reference).name,
                  runs=a.runs, inference_ms=dict(median=float(np.median(elapsed)),
                  p95=float(np.percentile(elapsed, 95))),
                  onnx_delta_px=dict(mean=float(region.mean()),
                  p95=float(np.percentile(region, 95)), max=float(region.max())),
                  scope='NPU forward only; excludes camera, rectification, reverse consistency and UI')
    Path(a.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    np.savez_compressed(Path(a.output).with_suffix('.npz'), disparity=prediction)
    print(json.dumps(report), flush=True)
finally:
    r.release()
