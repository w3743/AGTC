#!/usr/bin/env python3
"""Evaluate the existing LightStereo model on synchronized Gazebo stereo views."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

from common import ROOT, RUNTIME, lock_run

EXPERIMENT = ROOT.parent / '视觉算法' / '软件' / 'LightStereo实验'
CALIBRATION = EXPERIMENT.parent / 'stereo/calibration/current/parameters.npz'
POSES = [(0, 0, .55, 0), (1, 0, .55, 0), (1, -1.5, .55, 0),
         (0, 1, .55, math.pi/2), (0, 1, .55, math.pi)]
VALIDATION_POSES = [(0, -1, .55, .3), (-1, 1, .65, -.25),
                    (1.4, -1, .5, .65), (0, 1, .6, 2.8), (-1.2, -.5, .55, -1)]
REPLACEMENT_POSES = [(.7, .5, .55, .4), (-.4, 0, .65, -.5), (1.5, 1.2, .45, 2),
                     (-1.8, -1.2, .65, -2.6), (.2, -.8, .8, -1.8)]
REFINEMENT_POSES = [(-.8, -.6, .45, .2), (.8, 1.4, .7, -.6), (1.7, -.3, .6, 1.7),
                    (-1.5, .7, .8, -2.1), (.3, .4, .5, 2.6)]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding='utf-8')


def geometry():
    sys.path.insert(0, str(EXPERIMENT))
    from portrait import projection_geometry, verify_geometry
    verify_geometry(CALIBRATION)
    return projection_geometry(CALIBRATION)


def configure_appearance(world, folder, mode):
    """Change albedo only; use a uniform texture as the PBR shader control."""
    import cv2
    import numpy as np
    geometry = ET.fromstring(ET.tostring(world))
    for visual in geometry.findall('.//visual'):
        for material in visual.findall('material'):
            visual.remove(material)
    geometry_hash = hashlib.sha256(ET.tostring(geometry)).hexdigest()
    if mode == 'original':
        return dict(mode=mode, geometry_sha256=geometry_hash)
    textures = folder/'textures'
    textures.mkdir(exist_ok=True)
    manifest = hashlib.sha256()
    count = 0
    for model in world.findall('model'):
        name = model.get('name', '')
        if name != 'ground_plane' and not name.startswith(('wall_', 'marker_', 'crate_')):
            continue
        for index, visual in enumerate(model.findall('.//visual')):
            size = visual.findtext('geometry/plane/size') or visual.findtext('geometry/box/size')
            if size is None:
                continue
            extent = max(map(float, size.split()))
            cells = max(6, round(extent*16))
            pixels = 1024 if extent >= 3 else 128
            textured = mode == 'textured_all' or (mode == 'textured_floor' and name == 'ground_plane')
            if textured:
                seed = int.from_bytes(hashlib.sha256((name+str(index)).encode()).digest()[:4], 'little')
                rng = np.random.default_rng(seed)
                field = rng.uniform(-1, 1, (cells, cells)).astype(np.float32)
                field = cv2.resize(field, (pixels, pixels), interpolation=cv2.INTER_LINEAR)
                field -= field.mean()
                image = np.rint(153+70*field).clip(0, 255).astype(np.uint8)
            else:
                image = np.full((pixels, pixels), 153, np.uint8)
            path = textures/f'{name}_{index}.png'
            ok, encoded = cv2.imencode('.png', image)
            if not ok:
                raise RuntimeError('Texture encoding failed')
            path.write_bytes(encoded.tobytes())
            manifest.update(path.name.encode())
            manifest.update(encoded.tobytes())
            count += 1
            material = visual.find('material')
            if material is None:
                material = ET.SubElement(visual, 'material')
            for previous in material.findall('pbr'):
                material.remove(previous)
            metal = ET.SubElement(ET.SubElement(material, 'pbr'), 'metal')
            ET.SubElement(metal, 'albedo_map').text = str(path.resolve())
            ET.SubElement(metal, 'metalness').text = '0'
            ET.SubElement(metal, 'roughness').text = '1'
    return dict(mode=mode, geometry_sha256=geometry_hash, textures_sha256=manifest.hexdigest(),
                texture_count=count, mean_texture_u8=153, amplitude_u8=70,
                seed='sha256(model_name + visual_index) first four bytes little-endian',
                pattern='bilinear random cells, approximately 16 cells per metre of longest primitive dimension',
                changed='albedo maps only; same PBR parameters, geometry, lights and camera')


def build_world(folder, projection, appearance='original'):
    """Reuse the room geometry and add an ideal, already rectified stereo rig."""
    document = ET.parse(ROOT / 'simulation/lab.sdf')
    world = document.getroot().find('world')
    world.set('name', 'lightstereo_lab')
    for element in list(world):
        if ((element.tag == 'plugin' and 'UWB' in element.get('filename', '')) or
                (element.tag == 'model' and element.get('name') == 'x500_lab_rover')):
            world.remove(element)
    appearance_info = configure_appearance(world, folder, appearance)
    rig = ET.SubElement(world, 'model', name='lightstereo_rig')
    ET.SubElement(rig, 'static').text = 'true'
    ET.SubElement(rig, 'pose').text = '0 0 .55 0 0 0'
    link = ET.SubElement(rig, 'link', name='rig')
    baseline = -float(projection['P2'][0, 3]/projection['P2'][0, 0])
    for side, sign, kind in [('left', 1, 'rgbd_camera'), ('right', -1, 'camera')]:
        p = projection['P1' if side == 'left' else 'P2']
        sensor = ET.SubElement(link, 'sensor', name=side, type=kind)
        ET.SubElement(sensor, 'pose').text = f'0 {sign*baseline/2:.12f} 0 0 0 0'
        topic = f'/lightstereo/{side}'
        ET.SubElement(sensor, 'topic').text = topic if side == 'left' else topic+'/image'
        ET.SubElement(sensor, 'always_on').text = 'true'
        ET.SubElement(sensor, 'update_rate').text = '2'
        ET.SubElement(sensor, 'gz_frame_id').text = side+'_optical_frame'
        camera = ET.SubElement(sensor, 'camera')
        ET.SubElement(camera, 'camera_info_topic').text = topic+'/camera_info'
        ET.SubElement(camera, 'optical_frame_id').text = side+'_optical_frame'
        ET.SubElement(camera, 'horizontal_fov').text = str(2*math.atan(352/(2*p[0, 0])))
        image = ET.SubElement(camera, 'image')
        for key, value in [('width', 352), ('height', 640), ('format', 'R8G8B8')]:
            ET.SubElement(image, key).text = str(value)
        clip = ET.SubElement(camera, 'clip')
        ET.SubElement(clip, 'near').text = '.05'
        ET.SubElement(clip, 'far').text = '12'
        lens = ET.SubElement(camera, 'lens')
        ET.SubElement(lens, 'type').text = 'gnomonical'
        intrinsics = ET.SubElement(lens, 'intrinsics')
        for key, value in [('fx', p[0, 0]), ('fy', p[1, 1]), ('cx', p[0, 2]),
                           ('cy', p[1, 2]), ('s', 0)]:
            ET.SubElement(intrinsics, key).text = str(float(value))
    world_path = folder/'world.sdf'
    document.write(world_path, encoding='utf-8', xml_declaration=True)
    bridges = []
    for side in ('left', 'right'):
        for stream, message in [('image', 'Image'), ('camera_info', 'CameraInfo')]:
            topic = f'/lightstereo/{side}/{stream}'
            bridges.append(dict(gz_topic_name=topic, ros_topic_name=topic,
                                gz_type_name='gz.msgs.'+message,
                                ros_type_name='sensor_msgs/msg/'+message,
                                direction='GZ_TO_ROS'))
    bridges.append(dict(gz_topic_name='/lightstereo/left/depth_image',
                        ros_topic_name='/lightstereo/truth', gz_type_name='gz.msgs.Image',
                        ros_type_name='sensor_msgs/msg/Image', direction='GZ_TO_ROS'))
    bridge_path = folder/'bridge.yaml'
    write_json(bridge_path, bridges)
    return world_path, bridge_path, baseline, appearance_info


def capture(out, timeout, study=False, poses=None, appearance='original'):
    import cv2
    import numpy as np
    import rclpy
    from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image, CameraInfo
    from cv_bridge import CvBridge

    projection = geometry()
    runtime = RUNTIME/'lightstereo'
    runtime.mkdir(exist_ok=True)
    world, bridge, baseline, appearance_info = build_world(runtime, projection, appearance)
    os.environ['GZ_PARTITION'] = f'robotics_lightstereo_{os.getuid()}_{os.getpid()}_{time.time_ns()}'
    children, handles = [], []
    node = None

    def start(name, command):
        handle = (out/(name+'.log')).open('w')
        handles.append(handle)
        child = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        children.append(child)

    def alive():
        if any(child.poll() is not None for child in children):
            raise RuntimeError('Gazebo or bridge exited; inspect gazebo.log / bridge.log')

    rclpy.init()
    try:
        node = rclpy.create_node('lightstereo_capture')
        cvbridge = CvBridge()
        pending, complete, infos = {}, {}, {}

        def on_image(message, kind):
            stamp = message.header.stamp.sec*10**9 + message.header.stamp.nanosec
            encoding = '32FC1' if kind == 'truth' else 'rgb8'
            array = np.ascontiguousarray(cvbridge.imgmsg_to_cv2(message, encoding))
            expected_shape = (640, 352) if kind == 'truth' else (640, 352, 3)
            if array.shape != expected_shape:
                raise ValueError(f'{kind} shape is {array.shape}, expected {expected_shape}')
            group = pending.setdefault(stamp, {})
            if kind == 'truth':
                group[kind] = array
            else:
                group[kind+'_rgb'] = array
                group[kind] = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
            if set(group) == {'left', 'right', 'left_rgb', 'right_rgb', 'truth'}:
                complete[stamp] = group
            for cache in (pending, complete):
                while len(cache) > 12:
                    cache.pop(min(cache))

        def on_info(message, side):
            actual = np.array(message.k).reshape(3, 3)
            expected = projection['P1' if side == 'left' else 'P2'][:, :3]
            if (message.width, message.height) != (352, 640) or not np.allclose(actual, expected, atol=1e-5, rtol=0):
                raise ValueError(f'{side} CameraInfo does not match calibration: {actual}')
            infos[side] = actual.tolist()

        subscriptions = []
        # Raw RGB/depth messages span many DDS fragments. Reliable delivery
        # requests retransmission instead of silently losing whole frames.
        image_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        for kind in ('left', 'right', 'truth'):
            topic = '/lightstereo/truth' if kind == 'truth' else f'/lightstereo/{kind}/image'
            subscriptions.append(node.create_subscription(
                Image, topic, lambda msg, k=kind: on_image(msg, k), image_qos))
        for side in ('left', 'right'):
            subscriptions.append(node.create_subscription(
                CameraInfo, f'/lightstereo/{side}/camera_info',
                lambda msg, s=side: on_info(msg, s), qos_profile_sensor_data))
        start('gazebo', ['gz', 'sim', '-s', '-r', '-v', '3', str(world)])
        start('bridge', ['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
                         '--ros-args', '-p', f'config_file:={bridge}'])
        deadline = time.monotonic()+timeout
        while not complete or len(infos) != 2:
            alive()
            if time.monotonic() > deadline:
                raise TimeoutError('No synchronized stereo + truth / CameraInfo before deadline')
            rclpy.spin_once(node, timeout_sec=.1)
        initial_truth = complete[max(complete)]['truth']
        k = projection['K']
        cx, cy = round(float(k[0, 2])), round(float(k[1, 2]))
        wall = [float(np.median(initial_truth[yy-2:yy+3, xx-2:xx+3]))
                for xx, yy in [(cx, cy), (cx+70, cy-80)]]
        if not all(4.88 <= depth <= 4.95 for depth in wall):
            raise ValueError(f'Gazebo optical-Z wall sanity check failed: {wall}')
        frames, stamps = [], []
        poses = poses if poses is not None else (POSES + VALIDATION_POSES if study else POSES)
        for index, (x, y, z, yaw) in enumerate(poses):
            request = (f'name: "lightstereo_rig", position: {{x: {x}, y: {y}, z: {z}}}, '
                       f'orientation: {{z: {math.sin(yaw/2)}, w: {math.cos(yaw/2)}}}')
            response = subprocess.run(['gz', 'service', '-s', '/world/lightstereo_lab/set_pose',
                                       '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
                                       '--timeout', '5000', '--req', request],
                                      capture_output=True, text=True, timeout=8, check=True)
            if 'data: true' not in response.stdout:
                raise RuntimeError('Set-pose request was not accepted: '+response.stdout)
            settle = time.monotonic()+1
            while time.monotonic() < settle:
                rclpy.spin_once(node, timeout_sec=.1)
            pending.clear()
            complete.clear()
            while not complete:
                alive()
                if time.monotonic() > deadline:
                    recent = {stamp: sorted(value) for stamp, value in sorted(pending.items())[-4:]}
                    raise TimeoutError(f'Stereo capture {index+1} timed out; latest streams: {recent}')
                rclpy.spin_once(node, timeout_sec=.1)
            stamp = max(complete)
            frames.append(complete[stamp])
            stamps.append(stamp)
            print(f'Captured synchronized Gazebo view {index+1}/{len(poses)}', flush=True)
        np.savez_compressed(out/'frames.npz',
                            **{key: np.stack([frame[key] for frame in frames])
                               for key in ('left', 'right', 'left_rgb', 'right_rgb', 'truth')},
                            poses=np.array(poses), sim_stamp_ns=np.array(stamps, dtype=np.int64))
        renderer = subprocess.run(['glxinfo', '-B'], capture_output=True, text=True, timeout=10)
        return dict(views=len(frames), poses_xyz_yaw=poses, appearance=appearance_info,
                    timestamps_ns=stamps, stereo_timestamp_difference_ns=0, image_delivery_qos='reliable',
                    camera_info_K=infos, baseline_m=baseline,
                    analytic_wall_optical_z_m=wall, expected_wall_interval_m=[4.90, 4.925],
                    world_sha256=sha256(world),
                    original_world_sha256=sha256(ROOT/'simulation/lab.sdf'),
                    renderer=[line.strip() for line in renderer.stdout.splitlines()
                              if 'renderer string:' in line],
                    camera_model='ideal rectified pinhole; no noise or lens distortion')
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
        for child in reversed(children):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGINT)
        for child in children:
            try:
                child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        for handle in handles:
            handle.close()


def inference(out):
    import cv2
    import numpy as np
    import onnxruntime as ort
    sys.path.insert(0, str(EXPERIMENT))
    import evaluate as audit
    from evaluate_portrait import _run, _distance_legend
    from portrait import CroppedMetric

    cv2.setNumThreads(4)
    metric = CroppedMetric(CALIBRATION)
    model = EXPERIMENT/'models/lightstereo_640_352.onnx'
    model_hash = sha256(model)
    expected = json.loads((EXPERIMENT/'manifest.json').read_text())['files']['models\\lightstereo_640_352.onnx']['sha256']
    if model_hash != expected:
        raise ValueError('ONNX model differs from the existing manifest')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    with np.load(out/'frames.npz', allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    if arrays['left'].shape != (len(POSES), 640, 352):
        raise ValueError('Expected five Gazebo portrait stereo views')
    _run(session, arrays['left'][0], arrays['right'][0], native=True)
    stats, predictions, valids, errors = [], [], [], {'LightStereo': [], 'SGBM': []}

    def score(depth, valid, truth):
        eligible = np.isfinite(truth) & (truth >= .3) & (truth <= 5)
        checked = valid & eligible & np.isfinite(depth)
        delta = depth[checked].astype(np.float64)-truth[checked]
        absolute = np.abs(delta)
        result = dict(truth_pixels=int(eligible.sum()), checked_pixels=int(checked.sum()),
                      image_coverage=float(checked.mean()),
                      truth_coverage=float(checked.sum()/max(eligible.sum(), 1)))
        result.update(mae_m=float(absolute.mean()) if delta.size else None,
                      rmse_m=float(np.sqrt(np.mean(delta**2))) if delta.size else None,
                      abs_rel=float(np.mean(absolute/truth[checked])) if delta.size else None,
                      median_absolute_error_m=float(np.median(absolute)) if delta.size else None,
                      fraction_error_over_0_1m=float(np.mean(absolute > .1)) if delta.size else None)
        return result, absolute

    for index, (left, right, truth) in enumerate(zip(arrays['left'], arrays['right'], arrays['truth'])):
        disp, reverse, forward_ms, reverse_ms, _, _ = _run(session, left, right, native=True)
        if not np.isfinite(disp).all() or not np.isfinite(reverse).all():
            raise ValueError('Nonfinite ONNX disparity prediction')
        depth, _, _, valid, _ = audit._depth_from_disparity(metric, disp, reverse)
        learned, error = score(depth, valid, truth)
        learned.update(forward_ms=forward_ms, reverse_ms=reverse_ms)
        errors['LightStereo'].append(error)
        before = time.perf_counter()
        dl, dr = metric.disparities(left, right)
        baseline_depth, baseline_valid = metric.metric_values(dl, dr, left)
        baseline, error = score(baseline_depth, baseline_valid, truth)
        baseline['compute_ms'] = (time.perf_counter()-before)*1000
        errors['SGBM'].append(error)
        stats.append(dict(view=index+1, LightStereo=learned, SGBM=baseline))
        predictions.append(depth)
        valids.append(valid)
        print(f'View {index+1}: coverage={learned["truth_coverage"]:.1%}, MAE={learned["mae_m"]} m', flush=True)
    aggregate = {}
    for name in ('LightStereo', 'SGBM'):
        absolute = np.concatenate(errors[name])
        items = [item[name] for item in stats]
        aggregate[name] = dict(mean_truth_coverage=float(np.mean([item['truth_coverage'] for item in items])),
                               mean_image_coverage=float(np.mean([item['image_coverage'] for item in items])),
                               evaluated_pixels=int(absolute.size),
                               pooled_mae_m=float(absolute.mean()) if absolute.size else None,
                               pooled_median_error_m=float(np.median(absolute)) if absolute.size else None,
                               pooled_p95_error_m=float(np.percentile(absolute, 95)) if absolute.size else None)
    aggregate['LightStereo']['mean_stereo_inference_ms'] = float(np.mean([
        item['LightStereo']['forward_ms']+item['LightStereo']['reverse_ms'] for item in stats]))
    arrays.update(prediction=np.stack(predictions), prediction_valid=np.stack(valids))
    np.savez_compressed(out/'frames.npz', **arrays)
    index = 2
    truth = arrays['truth'][index].copy()
    truth[(truth < .3) | (truth > 5)] = np.nan
    predicted = predictions[index].copy()
    predicted[~valids[index]] = np.nan
    error = np.abs(predicted-truth)
    error_valid = np.isfinite(error)
    error_view = cv2.applyColorMap(np.nan_to_num(np.clip(error/.5, 0, 1)*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    error_view[~error_valid] = 0
    panes = [audit._pane(cv2.cvtColor(arrays['left'][index], cv2.COLOR_GRAY2BGR), 'left gray'),
             audit._pane(cv2.cvtColor(arrays['right'][index], cv2.COLOR_GRAY2BGR), 'right gray'),
             audit._pane(audit._color_depth(truth, (.3, 5)), 'Gazebo ground truth'),
             audit._pane(audit._color_depth(predicted, (.3, 5)), 'LightStereo checked'),
             audit._pane(error_view, 'absolute error 0-0.5m')]
    view = np.hstack(panes)
    audit._write_png(out/'comparison.png', np.vstack([view, _distance_legend(view.shape[1], (.3, 5))]))
    report = json.loads((out/'acceptance.json').read_text(encoding='utf-8'))
    report.update(status='COMPLETED', finished_at_utc=datetime.now(timezone.utc).isoformat(),
                  backend='ONNX Runtime CPUExecutionProvider',
                  versions=dict(python=sys.version.split()[0], numpy=np.__version__, opencv=cv2.__version__, onnxruntime=ort.__version__),
                  intra_op_threads=4, warmup_stereo_pairs=1,
                  model_sha256=model_hash, calibration_sha256=sha256(CALIBRATION),
                  filter='actual crop geometry; both-eye image support; LR <=1px; optical Z 0.3-5m; no hole filling',
                  aggregate=aggregate, views=stats,
                  interpretation='Simulator optical-Z truth; scores on valid predicted pixels with coverage reported separately. Five static views are an experiment, not real-world mapping/avoidance acceptance.',
                  output_sha256=dict(frames=sha256(out/'frames.npz'), comparison=sha256(out/'comparison.png')))
    write_json(out/'acceptance.json', report)
    print(json.dumps(aggregate, indent=2), flush=True)


def study_inference(out):
    """Choose preprocessing on views 1-5, then evaluate it unchanged on views 6-10."""
    import cv2
    import numpy as np
    import onnxruntime as ort
    sys.path.insert(0, str(EXPERIMENT))
    import evaluate as audit
    from evaluate_portrait import _run, _distance_legend
    from portrait import CroppedMetric

    cv2.setNumThreads(4)
    metric = CroppedMetric(CALIBRATION)
    with np.load(out/'frames.npz', allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    if arrays['left'].shape != (10, 640, 352) or arrays['left_rgb'].shape != (10, 640, 352, 3):
        raise ValueError('Study needs ten synchronized RGB stereo + truth views; rerun with --study')
    manifest = json.loads((EXPERIMENT/'manifest.json').read_text())['files']
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
    sessions, model_hashes = {}, {}
    paths = {'native': 'models/lightstereo_640_352.onnx',
             'padded': 'upstream/models/lightstereo_s_sceneflow_general_opt_576_960.onnx',
             'tiled': 'upstream/models/lightstereo_s_sceneflow_general_opt_256_512.onnx'}
    for name, relative in paths.items():
        path = EXPERIMENT/relative
        digest = sha256(path)
        if digest != manifest[relative.replace('/', '\\')]['sha256']:
            raise ValueError(f'Model differs from existing manifest: {relative}')
        model_hashes[name] = digest
        sessions[name] = ort.InferenceSession(str(path), sess_options=options,
                                             providers=['CPUExecutionProvider'])
    yy, xx = np.indices((640, 352), dtype=np.float32)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    methods = {'gray': ('gray', 1.), 'gray_gamma045': ('gray', .45),
               'gray_gamma065': ('gray', .65), 'rgb': ('rgb', 1.),
               'rgb_gamma045': ('rgb', .45), 'clahe': ('clahe', 1.),
               'padded': ('padded', 1.), 'tiled': ('tiled', 1.)}

    def native(left, right):
        def prep(image):
            rgb = np.repeat(image[..., None], 3, axis=2) if image.ndim == 2 else image
            tensor = np.transpose(rgb, (2, 0, 1))[None].astype(np.float32)/255
            return (tensor-audit.RGB_MEAN)/audit.RGB_STD
        session = sessions['native']
        names = [item.name for item in session.get_inputs()]
        output = session.get_outputs()[0].name
        start = time.perf_counter()
        d = session.run([output], {names[0]: prep(left), names[1]: prep(right)})[0][0, 0]
        r = session.run([output], {names[0]: prep(np.flip(right, 1)),
                                   names[1]: prep(np.flip(left, 1))})[0][0, 0]
        return d.astype(np.float32), np.flip(r, 1).copy(), (time.perf_counter()-start)*1000

    def predict(name, index):
        mode, gamma = methods[name]
        left, right = [arrays[side+('_rgb' if mode == 'rgb' else '')][index]
                       for side in ('left', 'right')]
        if gamma != 1:
            lut = np.round((np.arange(256)/255)**gamma*255).clip(0, 255).astype(np.uint8)
            left, right = cv2.LUT(left, lut), cv2.LUT(right, lut)
        if mode == 'clahe':
            left, right = clahe.apply(left), clahe.apply(right)
        if mode == 'padded':
            d, r, fwd, rev, _, _ = _run(sessions['padded'], left, right)
            return d, r, fwd+rev
        if mode == 'tiled':
            d, r, weight = [np.zeros(left.shape, np.float32) for _ in range(3)]
            ms = 0.
            taper = np.maximum(.1, 1-np.abs(np.arange(256)-127.5)/128).astype(np.float32)[:, None]
            for row in (0, 192, 384):
                dd, rr, fwd, rev, _, _ = _run(sessions['tiled'], left[row:row+256], right[row:row+256])
                d[row:row+256] += dd*taper
                r[row:row+256] += rr*taper
                weight[row:row+256] += taper
                ms += fwd+rev
            return d/weight, r/weight, ms
        return native(left, right)

    def evaluate(d, r, ms, index, lr_threshold=1., interpolate=False):
        if not np.isfinite(d).all() or not np.isfinite(r).all():
            raise ValueError('Nonfinite model disparity')
        depth, dense, support, valid, lr = audit._depth_from_disparity(metric, d, r)
        if interpolate:
            xr = xx-d
            sampled = cv2.remap(r, xr, yy, cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=float('nan'))
            support = dense & (xr >= 0) & (xr < 351)
            lr = np.abs(d-sampled)
        valid = support & np.isfinite(lr) & (lr <= lr_threshold)
        return dict(d=d, reverse=r, depth=depth, valid=valid, ms=ms,
                    stages={'range': dense, 'image_support': support, 'LR': valid})

    def summary(items, indices, baseline=None):
        errors, disparity_errors, common_old, common_new, rows = [], [], [], [], []
        stages = {key: [] for key in ('range', 'image_support', 'LR')}
        for item, index in zip(items, indices):
            t, gray = arrays['truth'][index], arrays['left'][index]
            eligible = np.isfinite(t) & (t >= .3) & (t <= 5)
            valid = item['valid'] & eligible & np.isfinite(item['depth'])
            absolute = np.abs(item['depth']-t)
            errors.append(absolute[valid].astype(np.float64))
            true_d = np.divide(metric.depth_factor, t, out=np.zeros_like(t), where=t > 0)+metric.disparity_offset
            disparity_errors.append(np.abs(item['d'][valid]-true_d[valid]).astype(np.float64))
            mean = cv2.boxFilter(gray.astype(np.float32), -1, (7, 7))
            variance = cv2.boxFilter(gray.astype(np.float32)**2, -1, (7, 7))-mean**2
            flat = eligible & (variance <= 4)
            row = dict(view=index+1, truth_pixels=int(eligible.sum()), valid_pixels=int(valid.sum()),
                       truth_coverage=float(valid.sum()/max(1, eligible.sum())),
                       output_image_coverage=float(item['valid'].mean()),
                       eligible_image_coverage=float(valid.mean()),
                       correct_0_1m_truth_coverage=float((valid & (absolute <= .1)).sum()/max(1, eligible.sum())),
                       mae_m=float(absolute[valid].mean()) if valid.any() else None,
                       flat_truth_fraction=float(flat.sum()/max(1, eligible.sum())),
                       flat_coverage=float((flat & valid).sum()/max(1, flat.sum())),
                       textured_coverage=float((eligible & ~flat & valid).sum()/max(1, (eligible & ~flat).sum())),
                       compute_ms=item['ms'])
            rows.append(row)
            for key, mask in item.get('stages', {}).items():
                stages[key].append(float((eligible & mask).sum()/max(1, eligible.sum())))
            if baseline is not None:
                old = baseline[index]
                common = valid & old['valid'] & np.isfinite(old['depth'])
                common_old.append(np.abs(old['depth'][common]-t[common]).astype(np.float64))
                common_new.append(absolute[common].astype(np.float64))
        error, disp_error = np.concatenate(errors), np.concatenate(disparity_errors)
        result = dict(mean_truth_coverage=float(np.mean([row['truth_coverage'] for row in rows])),
                      mean_output_image_coverage=float(np.mean([row['output_image_coverage'] for row in rows])),
                      mean_eligible_image_coverage=float(np.mean([row['eligible_image_coverage'] for row in rows])),
                      correct_0_1m_truth_coverage=float(np.mean([row['correct_0_1m_truth_coverage'] for row in rows])),
                      evaluated_pixels=int(error.size), pooled_mae_m=float(error.mean()) if error.size else None,
                      pooled_p95_error_m=float(np.percentile(error, 95)) if error.size else None,
                      pooled_disparity_epe_px=float(disp_error.mean()) if disp_error.size else None,
                      bad_disparity_over_3px_fraction=float(np.mean(disp_error > 3)) if disp_error.size else None,
                      mean_compute_ms=float(np.mean([row['compute_ms'] for row in rows])),
                      mean_flat_truth_fraction=float(np.mean([row['flat_truth_fraction'] for row in rows])),
                      stages_truth_coverage={key: float(np.mean(values)) for key, values in stages.items() if values},
                      views=rows)
        if baseline is not None:
            old, new = np.concatenate(common_old), np.concatenate(common_new)
            result['common_pixels'] = dict(count=int(new.size), baseline_mae_m=float(old.mean()) if new.size else None,
                                           candidate_mae_m=float(new.mean()) if new.size else None)
        return result

    native(arrays['left'][0], arrays['right'][0])  # One unscored warm-up pair.
    development, validation = list(range(5)), list(range(5, 10))
    baseline = {index: evaluate(*predict('gray', index), index) for index in range(10)}
    trials, predictions = {}, {'gray': [baseline[index] for index in development]}
    for name in methods:
        if name != 'gray':
            predictions[name] = [evaluate(*predict(name, index), index) for index in development]
        trials[name] = summary(predictions[name], development, baseline)
        print(f'Development {name}: coverage={trials[name]["mean_truth_coverage"]:.1%}, MAE={trials[name]["pooled_mae_m"]:.3f}m', flush=True)
    for name, threshold, interpolate in [('LR_2px', 2., False), ('LR_subpixel', 1., True)]:
        items = [evaluate(baseline[index]['d'], baseline[index]['reverse'], baseline[index]['ms'],
                          index, threshold, interpolate) for index in development]
        trials[name] = summary(items, development, baseline)
    reference = trials['gray']
    candidates = []
    for name in ('gray_gamma045', 'gray_gamma065', 'rgb', 'rgb_gamma045'):
        item, common = trials[name], trials[name]['common_pixels']
        if (item['mean_truth_coverage'] > reference['mean_truth_coverage'] and
                item['pooled_mae_m'] < reference['pooled_mae_m'] and
                item['pooled_p95_error_m'] < reference['pooled_p95_error_m'] and
                common['count'] > 0 and common['candidate_mae_m'] < common['baseline_mae_m']):
            candidates.append(name)
    selected = max(candidates, key=lambda name: trials[name]['correct_0_1m_truth_coverage']) if candidates else 'gray'
    report = json.loads((out/'acceptance.json').read_text(encoding='utf-8'))
    report['study'] = dict(status='DEVELOPMENT_COMPLETE', development_views=[1, 2, 3, 4, 5],
                          validation_views=[6, 7, 8, 9, 10], selected_candidate=selected,
                          selection='Native model only; require increased coverage and lower MAE/p95/common-pixel MAE, then maximize development truth coverage within 0.1m. Validation truth is not used for selection.',
                          development_trials=trials)
    write_json(out/'acceptance.json', report)
    chosen = {index: item for index, item in zip(development, predictions[selected])}
    for index in validation:
        chosen[index] = baseline[index] if selected == 'gray' else evaluate(*predict(selected, index), index)
    sgbm = {}
    for index in range(10):
        started = time.perf_counter()
        dl, dr = metric.disparities(arrays['left'][index], arrays['right'][index])
        z, valid = metric.metric_values(dl, dr, arrays['left'][index])
        sgbm[index] = dict(d=dl, depth=z, valid=valid, ms=(time.perf_counter()-started)*1000)
    aggregate = {}
    for split, indices in [('development', development), ('validation', validation)]:
        aggregate[split] = {name: summary([values[index] for index in indices], indices,
                                           baseline if name != 'baseline' else None)
                            for name, values in [('baseline', baseline), ('candidate', chosen), ('SGBM', sgbm)]}
    before, after = aggregate['validation']['baseline'], aggregate['validation']['candidate']
    common = after['common_pixels']
    improved = (after['mean_truth_coverage'] > before['mean_truth_coverage'] and
                after['pooled_mae_m'] < before['pooled_mae_m'] and
                after['correct_0_1m_truth_coverage'] > before['correct_0_1m_truth_coverage'] and
                common['count'] > 0 and common['candidate_mae_m'] < common['baseline_mae_m'])
    arrays.update(baseline_prediction=np.stack([baseline[i]['depth'] for i in range(10)]),
                  baseline_valid=np.stack([baseline[i]['valid'] for i in range(10)]),
                  prediction=np.stack([chosen[i]['depth'] for i in range(10)]),
                  prediction_valid=np.stack([chosen[i]['valid'] for i in range(10)]),
                  disparity=np.stack([chosen[i]['d'] for i in range(10)]),
                  reverse_disparity=np.stack([chosen[i]['reverse'] for i in range(10)]))
    np.savez_compressed(out/'frames.npz', **arrays)
    panels = []
    for index in (2, 7):  # Fixed representative views, independent of results.
        truth = arrays['truth'][index].copy()
        truth[(truth < .3) | (truth > 5)] = np.nan
        views = [audit._pane(cv2.cvtColor(arrays['left_rgb'][index], cv2.COLOR_RGB2BGR), f'view {index+1} RGB'),
                 audit._pane(audit._color_depth(truth, (.3, 5)), 'Gazebo truth')]
        for label, values in [('baseline gray', baseline), (selected, chosen)]:
            z = values[index]['depth'].copy()
            z[~values[index]['valid']] = np.nan
            views.append(audit._pane(audit._color_depth(z, (.3, 5)), label))
        for label, values in [('baseline error', baseline), ('candidate error', chosen)]:
            error = np.abs(values[index]['depth']-truth)
            color = cv2.applyColorMap(np.nan_to_num(np.clip(error/.5, 0, 1)*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            color[~(np.isfinite(error) & values[index]['valid'])] = 0
            views.append(audit._pane(color, label+' 0-0.5m'))
        panels.append(np.hstack(views))
    width = panels[0].shape[1]
    audit._write_png(out/'comparison.png', np.vstack(panels+[_distance_legend(width, (.3, 5))]))
    report['study'].update(status='IMPROVED_ON_NEW_VIEWS' if improved else 'NO_GENERALIZATION_CONFIRMED',
                           selected_preprocess=dict(input=methods[selected][0], gamma=methods[selected][1],
                                                    lut='round(255*(uint8/255)**gamma)', same_lut_both_eyes=True),
                           validation_aggregate=aggregate['validation'],
                           validation_views_with_both_coverage_and_mae_improved=sum(
                               new['truth_coverage'] > old['truth_coverage'] and new['mae_m'] < old['mae_m']
                               for old, new in zip(before['views'], after['views'])))
    # Existing real photos provide a separate compatibility check. Their PnP
    # plane shares the calibration and is not independent metric ground truth.
    calibration_dir = CALIBRATION.parent
    calibration_report = audit._json(calibration_dir/'report.json')
    real_ids = calibration_report['heldout_ids']
    detections = audit._json(calibration_dir/'detections.json')
    original_metric = audit.MetricDepth(CALIBRATION)
    real_rows, real_errors = [], {'baseline': [], 'candidate': []}
    for image_id in real_ids:
        folder = EXPERIMENT.parent/'stereo/captures'/image_id
        raw_left = cv2.imread(str(folder/'left.png'), cv2.IMREAD_GRAYSCALE)
        raw_right = cv2.imread(str(folder/'right.png'), cv2.IMREAD_GRAYSCALE)
        if raw_left is None or raw_right is None:
            raise FileNotFoundError(f'Missing existing real stereo photos: {image_id}')
        full_left, full_right = original_metric.rectify(raw_left, raw_right)
        left, right = full_left[:, 4:-4], full_right[:, 4:-4]
        expected_plane, _ = audit._plane_reference(original_metric, detections, image_id)
        if expected_plane is None:
            raise ValueError(f'No checkerboard plane for {image_id}')
        expected_plane = expected_plane[:, 4:-4]
        real_methods = {}
        for name, gamma in [('baseline', 1.), ('candidate', methods[selected][1])]:
            # The selected gray method can run on existing monochrome captures.
            # RGB candidates cannot be evaluated from these monochrome files.
            if name == 'candidate' and methods[selected][0] != 'gray':
                continue
            lut = np.round((np.arange(256)/255)**gamma*255).clip(0, 255).astype(np.uint8)
            d, r, ms = native(cv2.LUT(left, lut), cv2.LUT(right, lut))
            item = evaluate(d, r, ms, 0)
            board, _ = audit._board_stats(item['depth'], item['valid'], expected_plane)
            real_methods[name] = (item, board)
        row = dict(id=image_id, mean_gray=float(left.mean()),
                   methods={name: dict(image_coverage=float(item['valid'].mean()), plane_consistency_only=board)
                            for name, (item, board) in real_methods.items()})
        if 'candidate' in real_methods:
            common = np.isfinite(expected_plane)
            for item, _ in real_methods.values():
                common &= item['valid'] & np.isfinite(item['depth'])
            for name, (item, _) in real_methods.items():
                real_errors[name].append(np.abs(item['depth'][common]-expected_plane[common]).astype(np.float64))
        real_rows.append(row)
    real_aggregate = {}
    for name in ('baseline', 'candidate'):
        rows = [row['methods'][name] for row in real_rows if name in row['methods']]
        error = np.concatenate(real_errors[name]) if real_errors[name] else np.array([])
        if rows:
            real_aggregate[name] = dict(mean_image_coverage=float(np.mean([row['image_coverage'] for row in rows])),
                                       common_board_pixels=int(error.size),
                                       common_plane_mae_m=float(error.mean()) if error.size else None,
                                       common_plane_p90_error_m=float(np.percentile(error, 90)) if error.size else None)
    report['study']['real_camera_sanity'] = dict(
        interpretation='Five existing monochrome checkerboard photos; coverage and calibration-dependent plane consistency only, not independent distance accuracy or unseen obstacle acceptance.',
        aggregate=real_aggregate, views=real_rows, deployed=False)

    dev_before, dev_after = aggregate['development']['baseline'], aggregate['development']['candidate']
    factors = [
        dict(factor='低纹理的大面积平面', status='已测量关联；未做增加纹理的因果对照',
             evidence=dict(development_flat_fraction=dev_before['mean_flat_truth_fraction'],
                           validation_flat_fraction=before['mean_flat_truth_fraction'],
                           validation_flat_coverage=float(np.mean([v['flat_coverage'] for v in before['views']])),
                           validation_textured_coverage=float(np.mean([v['textured_coverage'] for v in before['views']])),
                           almost_textureless_view=before['views'][1]),
             explanation='7×7灰度方差≤4的区域缺少局部匹配证据；黑色区域集中在这些区域。重复色块可能进一步产生匹配歧义，但该项未单独隔离。'),
        dict(factor='输入亮度与对比度分布影响模型', status='同帧对照及新视角验证支持',
             evidence=dict(gamma=methods[selected][1], development_before=dev_before['mean_truth_coverage'],
                           development_after=dev_after['mean_truth_coverage'], validation_before=before['mean_truth_coverage'],
                           validation_after=after['mean_truth_coverage'], validation_mae_before_m=before['pooled_mae_m'],
                           validation_mae_after_m=after['pooled_mae_m'], real_camera_sanity=real_aggregate),
             explanation='两眼使用相同Gamma查表，保持几何和LR阈值不变。仿真候选不能自动推广到实拍图像。'),
        dict(factor='左右视差不一致导致筛除', status='已确认直接原因；放宽阈值的副作用已对照',
             evidence=dict(development_stages=dev_before['stages_truth_coverage'], validation_stages=before['stages_truth_coverage'],
                           LR_1px_mae_m=reference['pooled_mae_m'], LR_2px_coverage=trials['LR_2px']['mean_truth_coverage'],
                           LR_2px_mae_m=trials['LR_2px']['pooled_mae_m'], subpixel_coverage=trials['LR_subpixel']['mean_truth_coverage']),
             explanation='对应坐标采样的LR筛选拒绝不可靠视差。放宽阈值减少黑色区域，同时使有效预测平均误差升高；亚像素LR采样未带来明显改善。'),
        dict(factor='量程限制和双目视野边界', status='已确认筛选规则',
             evidence=dict(depth_range_m=[.3, 5], validation_truth_outside_range_fraction=float(np.mean([
                 1-v['truth_pixels']/(640*352) for v in before['views']])),
                 development_stages=dev_before['stages_truth_coverage']),
             explanation='真实深度超出0.3～5m的黑色区域属于当前工作量程；没有右眼对应位置的像素无法通过双目支持筛选。真实量程与预测越界分别统计。'),
        dict(factor='远距离视差小，误差放大', status='由现有标定几何推算；非单独因果试验',
             evidence=dict(depth_factor_px_m=metric.depth_factor, disparity_at_5m_px=metric.depth_factor/5+metric.disparity_offset,
                           error_at_5m_plus_1px_m=5-metric.depth_factor/(metric.depth_factor/5+1),
                           error_at_5m_minus_1px_m=metric.depth_factor/(metric.depth_factor/5-1)-5),
             explanation='约96.3mm基线和393.3px焦距使5m目标仅约7.57px视差。1px误差可造成约0.58～0.76m的距离误差。'),
        dict(factor='输入缩放补边及不合适的局部增强', status='同帧对照支持；非当前原生输入的残留原因',
             evidence={name: dict(coverage=trials[name]['mean_truth_coverage'], mae_m=trials[name]['pooled_mae_m'])
                       for name in ('gray', 'padded', 'tiled', 'clahe')},
             explanation='本场景中576×960大面积补边、分块推理和独立CLAHE均未同时改善覆盖与误差；当前保留原生352×640输入。'),
        dict(factor='灰度复制丢失颜色是否是主因', status='本次RGB对照不支持该解释',
             evidence={name: dict(coverage=trials[name]['mean_truth_coverage'], mae_m=trials[name]['pooled_mae_m'])
                       for name in ('gray', 'rgb', 'rgb_gamma045')},
             explanation='官方模型支持RGB；该Gazebo房间使用RGB反而变差，不能把缺色彩直接判作本次低覆盖的主因。'),
        dict(factor='训练域差异和真实相机误差', status='待验证因素',
             evidence=dict(model='existing SceneFlow pretrained LightStereo-S; weights unchanged',
                           simulator='ideal rectified cameras; no noise/distortion',
                           real_calibration_heldout=calibration_report['heldout']),
             explanation='当前模型未针对本场景训练；材质、照明和纹理分布的域差异尚未隔离。仿真几何检查通过，但实拍还可能受标定残差、同步、曝光、模糊影响，需独立实测。')]
    boundaries = []
    for index in range(10):
        truth = arrays['truth'][index]
        eligible = np.isfinite(truth) & (truth >= .3) & (truth <= 5)
        safe_truth = np.nan_to_num(truth, nan=12., posinf=12., neginf=0.)
        edge = eligible & (cv2.dilate(safe_truth, np.ones((3, 3), np.uint8))-
                           cv2.erode(safe_truth, np.ones((3, 3), np.uint8)) > .25)
        item = chosen[index]
        regions = {}
        for name, mask in [('depth_boundary', edge), ('interior', eligible & ~edge)]:
            valid = mask & item['valid'] & np.isfinite(item['depth'])
            error = np.abs(item['depth'][valid]-truth[valid])
            regions[name] = dict(truth_pixels=int(mask.sum()), valid_pixels=int(valid.sum()),
                                 coverage=float(valid.sum()/mask.sum()) if mask.any() else None,
                                 mae_m=float(error.mean()) if error.size else None)
        boundaries.append(dict(view=index+1, regions=regions))
    factors.append(dict(factor='深度边界和遮挡附近', status='边界区域已分组测量；遮挡机制未独立隔离',
                        evidence=boundaries, explanation='以3×3真值深度跨度>0.25m定义边界；边界与内部的有效覆盖及误差分别记录，不能把插值跨越边界产生的深度当作观测。'))
    report['study']['low_coverage_factors'] = factors
    report['study']['primary_sources'] = [
        'https://github.com/XiandaGuo/OpenStereo/blob/v2/cfgs/lightstereo/lightstereo_s_sceneflow.yaml',
        'https://arxiv.org/abs/2406.19833']
    report.update(status='COMPLETED', finished_at_utc=datetime.now(timezone.utc).isoformat(),
                  backend='ONNX Runtime CPUExecutionProvider', intra_op_threads=4, warmup_stereo_pairs=1,
                  versions=dict(python=sys.version.split()[0], numpy=np.__version__, opencv=cv2.__version__, onnxruntime=ort.__version__),
                  model_sha256=model_hashes, calibration_sha256=sha256(CALIBRATION),
                  filter='Both-eye support; LR<=1px for selected candidate; optical Z 0.3-5m; no filling or GT in prediction.',
                  aggregate=aggregate,
                  interpretation='Five development and five new static validation views in the same simulator room. Optical-Z truth is evaluation-only. Coverage is not classification accuracy; valid-pixel MAE and common-pixel MAE are reported. Real-camera and RK3588 performance are unverified.',
                  output_sha256=dict(frames=sha256(out/'frames.npz'), comparison=sha256(out/'comparison.png')))
    write_json(out/'acceptance.json', report)
    print(json.dumps(dict(selected=selected, status=report['study']['status'], validation=aggregate['validation']), indent=2), flush=True)


def extend_capture(out, timeout):
    """Keep the ten original views and append five new views for replacement validation."""
    import numpy as np
    frames = out/'frames.npz'
    original_hash = sha256(frames)
    keys = ('left', 'right', 'left_rgb', 'right_rgb', 'truth', 'poses', 'sim_stamp_ns')
    with np.load(frames, allow_pickle=False) as saved:
        original = {key: saved[key][:10].copy() for key in keys}
    if original['left'].shape != (10, 640, 352):
        raise ValueError('Run menu 11 before collecting replacement validation views')
    report = json.loads((out/'acceptance.json').read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory(prefix='stereo-validation-', dir=RUNTIME) as temporary:
        folder = Path(temporary)
        try:
            new_capture = capture(folder, timeout, poses=REPLACEMENT_POSES)
            with np.load(folder/'frames.npz', allow_pickle=False) as saved:
                arrays = {key: np.concatenate([original[key], saved[key]]) for key in keys}
            if sha256(frames) != original_hash:
                raise RuntimeError('Source frames changed during capture; refusing to overwrite')
            np.savez_compressed(frames, **arrays)
        finally:
            for name in ('gazebo.log', 'bridge.log'):
                if (folder/name).is_file():
                    (out/name).write_bytes((folder/name).read_bytes())
    report['capture'].update(views=15, poses_xyz_yaw=arrays['poses'].tolist(),
                             timestamps_ns=arrays['sim_stamp_ns'].tolist(),
                             session_per_view=[0]*10+[1]*5,
                             replacement_validation_capture=new_capture)
    report.update(status='CAPTURED_FOR_REPLACEMENT')
    report['output_sha256']['frames'] = sha256(frames)
    write_json(out/'acceptance.json', report)


def refinement_capture(out, timeout):
    """Capture new validation views while retaining the earlier experiment."""
    import numpy as np
    keys = ('left', 'right', 'left_rgb', 'right_rgb', 'truth', 'poses', 'sim_stamp_ns')
    with np.load(out/'frames.npz', allow_pickle=False) as saved:
        original = {key: saved[key].copy() for key in keys}
    if original['left'].shape != (15, 640, 352):
        raise ValueError('Run menu 12 before the refinement experiment')
    report = json.loads((out/'acceptance.json').read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory(prefix='stereo-refinement-', dir=RUNTIME) as temporary:
        folder = Path(temporary)
        info = capture(folder, timeout, poses=REFINEMENT_POSES)
        with np.load(folder/'frames.npz', allow_pickle=False) as saved:
            arrays = {key: np.concatenate([original[key], saved[key]]) for key in keys}
        np.savez_compressed(out/'refinement.npz', **arrays)
    report['refinement_study'] = dict(status='CAPTURED', capture=info,
        development_views=list(range(1, 16)), new_validation_views=list(range(16, 21)),
        original_frames_sha256=sha256(out/'frames.npz'))
    write_json(out/'acceptance.json', report)


def appearance_capture(out, timeout):
    import numpy as np
    report = json.loads((out/'acceptance.json').read_text(encoding='utf-8'))
    arrays, captures = {}, {}
    for condition in ('uniform', 'textured_floor', 'textured_all'):
        with tempfile.TemporaryDirectory(prefix='stereo-appearance-', dir=RUNTIME) as temporary:
            folder = Path(temporary)
            try:
                # A fresh ROS context/process per condition avoids carrying DDS
                # subscriptions between sequential Gazebo instances.
                subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--appearance-single', condition,
                                '--seconds', str(timeout), '--output', str(folder)], check=True)
                captures[condition] = json.loads((folder/'capture.json').read_text(encoding='utf-8'))
                with np.load(folder/'frames.npz', allow_pickle=False) as saved:
                    arrays.update({condition+'_'+key: saved[key].copy() for key in saved.files})
            finally:
                for name in ('gazebo.log', 'bridge.log'):
                    if (folder/name).is_file():
                        (out/name).write_bytes((folder/name).read_bytes())
        print('Captured appearance condition: '+condition, flush=True)
        np.savez_compressed(out/'appearance.npz', **arrays)
    for condition in ('textured_floor', 'textured_all'):
        if not np.array_equal(arrays['uniform_truth'], arrays[condition+'_truth'], equal_nan=True):
            raise ValueError('Albedo comparison changed depth truth')
        if captures[condition]['appearance']['geometry_sha256'] != captures['uniform']['appearance']['geometry_sha256']:
            raise ValueError('Albedo comparison changed geometry')
    np.savez_compressed(out/'appearance.npz', **arrays)
    report['appearance_study'] = dict(status='CAPTURED', captures=captures, depth_truth_identical=True,
        interpretation='Controlled albedo experiment on fixed poses16-20; uniform PBR control isolates the texture change from switching shaders. No algorithm retuning.')
    write_json(out/'acceptance.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--infer', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--study', action='store_true', help='Compare preprocessing on five views and validate the chosen method on five new views')
    parser.add_argument('--extend', action='store_true', help='Append five new views for replacement-algorithm validation')
    parser.add_argument('--refine-capture', action='store_true', help='Capture five new views for cross-model refinement')
    parser.add_argument('--appearance-capture', action='store_true', help='Capture uniform/floor-textured/all-textured controls')
    parser.add_argument('--appearance-single', choices=('uniform', 'textured_floor', 'textured_all'), help=argparse.SUPPRESS)
    parser.add_argument('--python', type=Path, default=Path(os.environ.get('LIGHTSTEREO_PYTHON', str(Path.home()/'.venvs/lightstereo/bin/python'))))
    parser.add_argument('--seconds', type=float, default=90, help='Capture deadline in wall seconds')
    parser.add_argument('--output', type=Path, default=ROOT/'results/lightstereo')
    args = parser.parse_args()
    out = args.output.resolve()
    if args.infer:
        (study_inference if args.study else inference)(out)
        return
    if args.appearance_single:
        out.mkdir(parents=True, exist_ok=True)
        info = capture(out, args.seconds, poses=REFINEMENT_POSES, appearance=args.appearance_single)
        write_json(out/'capture.json', info)
        return
    lock = lock_run('lightstereo')
    if args.appearance_capture:
        appearance_capture(out, args.seconds)
        return
    if args.refine_capture:
        refinement_capture(out, args.seconds)
        return
    if args.extend:
        extend_capture(out, args.seconds)
        return
    if args.output.is_symlink():
        raise RuntimeError('Result directory must not be a symlink')
    out.mkdir(parents=True, exist_ok=True)
    for name in ('frames.npz', 'comparison.png', 'acceptance.json', 'gazebo.log', 'bridge.log'):
        path = out/name
        if path.is_symlink():
            raise RuntimeError('Generated output must not be a symlink')
        if path.is_file():
            path.unlink()
    if not args.python.is_file():
        raise FileNotFoundError(f'Provide the existing LightStereo environment with --python: {args.python}')
    try:
        capture_info = capture(out, args.seconds, args.study)
        write_json(out/'acceptance.json', dict(status='CAPTURED', capture=capture_info))
        environment = os.environ.copy()
        environment.pop('PYTHONPATH', None)
        environment.pop('PYTHONHOME', None)
        command = [str(args.python), '-B', str(Path(__file__).resolve()), '--infer', '--output', str(out)]
        if args.study:
            command.append('--study')
        subprocess.run(command,
                       env=environment, check=True)
    except Exception as error:
        report = json.loads((out/'acceptance.json').read_text()) if (out/'acceptance.json').is_file() else {}
        report.update(status='FAILED', error=str(error))
        write_json(out/'acceptance.json', report)
        raise


if __name__ == '__main__':
    main()
