"""Convert the pinned upstream LightStereo ONNX to isolated RK3588 FP16 model."""
import argparse
from pathlib import Path
from rknn.api import RKNN

p = argparse.ArgumentParser()
p.add_argument('--size', choices=['256_512', '576_960', '640_352'], default='640_352')
a = p.parse_args()
root = Path(__file__).resolve().parent
source = root/'upstream/models'/f'lightstereo_s_sceneflow_general_opt_{a.size}.onnx'
if a.size == '640_352':
    source = root/'models/lightstereo_640_352.onnx'
output = root/'models'/f'lightstereo_{a.size}.rknn'
output.parent.mkdir(exist_ok=True)
r = RKNN(verbose=False)
try:
    r.config(mean_values=[[123.675, 116.28, 103.53]]*2,
             std_values=[[58.395, 57.12, 57.375]]*2,
             target_platform='rk3588', optimization_level=2)
    assert r.load_onnx(model=str(source)) == 0, 'ONNX load failed'
    assert r.build(do_quantization=False) == 0, 'RKNN build failed'
    assert r.export_rknn(str(output)) == 0, 'RKNN export failed'
    print('SAVED', output, flush=True)
finally:
    r.release()
