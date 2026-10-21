"""Retarget the pinned fully convolutional ONNX without changing learned weights."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper as nh

ROOT = Path(__file__).resolve().parent


def spatial_constants(height, width):
    if height % 32 or width % 32 or width < 192:
        raise ValueError('Require stride-32 sizes and enough horizontal disparity support')
    h, w = height//4, width//4
    values = {
        '/Add_2_output_0': np.arange(h)[None, :] + np.arange(3)[:, None],
        '/Add_4_output_0': np.arange(w)[None, :] + np.arange(3)[:, None],
        '/Concat_4_output_0': [1, -1, h, w],
        '/Unsqueeze_106_output_0': [height],
        '/Unsqueeze_107_output_0': [width],
    }
    for i in range(48):
        name = '/Unsqueeze'+('' if i == 0 else f'_{i}')+'_output_0'
        values[name] = [w+48-i]
    return {k: np.asarray(v, np.int64) for k, v in values.items()}


def retarget(model, height, width):
    old_h, old_w = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim][-2:]
    old = spatial_constants(old_h, old_w)
    new = spatial_constants(height, width)
    found = set()
    for tensor in model.graph.initializer:
        if tensor.name in old:
            assert np.array_equal(nh.to_array(tensor), old[tensor.name]), tensor.name
            found.add(tensor.name)
            tensor.CopyFrom(nh.from_array(new[tensor.name], tensor.name))
    assert found == set(old)
    for value in list(model.graph.input)+list(model.graph.output):
        dims = value.type.tensor_type.shape.dim
        dims[-2].dim_value, dims[-1].dim_value = height, width
    # Old inferred intermediate shapes belong to the old image size.
    del model.graph.value_info[:]
    onnx.checker.check_model(model)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    return model


def main():
    base = ROOT/'upstream/models/lightstereo_s_sceneflow_general_opt_576_960.onnx'
    small = ROOT/'upstream/models/lightstereo_s_sceneflow_general_opt_256_512.onnx'
    a, b = onnx.load(base), onnx.load(small)
    # Independent exports establish the exact integer/index topology at two sizes.
    assert [n.SerializeToString() for n in a.graph.node] == [n.SerializeToString() for n in b.graph.node]
    resized = retarget(onnx.load(base), 256, 512)
    expected = {t.name: nh.to_array(t) for t in b.graph.initializer}
    verified = 0
    for t in resized.graph.initializer:
        v = nh.to_array(t)
        if v.dtype.kind in 'iu':
            assert np.array_equal(v, expected[t.name]), t.name
            verified += 1
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.log_severity_level = 3
    rebuilt = ort.InferenceSession(resized.SerializeToString(), options, providers=['CPUExecutionProvider'])
    independent = ort.InferenceSession(str(small), options, providers=['CPUExecutionProvider'])
    inputs = {}
    for eye in ('left', 'right'):
        im = cv2.imdecode(np.frombuffer((ROOT/f'upstream/test_data/{eye}.png').read_bytes(), np.uint8), 1)
        im = cv2.cvtColor(cv2.resize(im, (512, 256)), cv2.COLOR_BGR2RGB).astype(np.float32)
        im = (im-np.array([123.675, 116.28, 103.53], np.float32))/np.array([58.395, 57.12, 57.375], np.float32)
        inputs[eye+'_img'] = np.ascontiguousarray(im.transpose(2, 0, 1)[None])
    delta = np.abs(rebuilt.run(None, inputs)[0]-independent.run(None, inputs)[0])
    # The independently exported model has tiny float differences from BN folding.
    assert float(np.percentile(delta, 99)) < .02 and float(delta.max()) < .2
    original_weights = {t.name: t.SerializeToString() for t in a.graph.initializer
                        if nh.to_array(t).dtype.kind == 'f'}
    portrait = retarget(a, 640, 352)
    assert all(t.SerializeToString() == original_weights[t.name]
               for t in portrait.graph.initializer if t.name in original_weights)
    target = ROOT/'models/lightstereo_640_352.onnx'
    onnx.save(portrait, target)
    report = dict(source=base.name, source_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
                  output=target.name, output_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                  input_nchw=[1, 3, 640, 352], unchanged_float_tensors=len(original_weights),
                  changed_spatial_constants=len(spatial_constants(640, 352)),
                  independently_checked_integer_tensors=verified,
                  independent_export_delta_px=dict(mean=float(delta.mean()), p99=float(np.percentile(delta, 99)), max=float(delta.max())),
                  external_image_padding=0, input_crop_left_right_px=[4, 4],
                  note='Spatial/index constants retargeted; learned weights unchanged. Network internal convolution padding is retained.')
    (ROOT/'results/portrait-export.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
