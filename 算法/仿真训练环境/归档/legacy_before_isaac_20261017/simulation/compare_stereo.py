#!/usr/bin/env python3
"""Compare replacement stereo models using the existing synchronized frames."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import warnings
from types import SimpleNamespace, ModuleType

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT.parent/'视觉算法/软件/LightStereo实验'
CALIBRATION = EXPERIMENT.parent/'stereo/calibration/current/parameters.npz'
CACHE = Path(os.environ.get('STEREO_CACHE', str(Path(os.environ.get('LOCALAPPDATA', str(Path.home()/'.cache')))/'stereo-lab')))


def sha256(path):
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256')
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def raw_capture_sha256(arrays):
    """Hash captured inputs independently of the derived prediction arrays."""
    digest = hashlib.sha256()
    for key in ('left', 'right', 'left_rgb', 'right_rgb', 'truth', 'poses', 'sim_stamp_ns'):
        value = arrays[key]
        digest.update(f'{key}:{value.dtype}:{value.shape}\n'.encode('ascii'))
        digest.update(value.tobytes(order='C'))
    return digest.hexdigest()


def low_texture_evidence(arrays):
    import cv2
    import numpy as np
    rows = []
    for index in range(5, 15):
        gray = arrays['left'][index].astype(np.float32)
        mean = cv2.boxFilter(gray, cv2.CV_32F, (7, 7))
        variance = cv2.boxFilter(gray*gray, cv2.CV_32F, (7, 7))-mean*mean
        truth = arrays['truth'][index]
        eligible = np.isfinite(truth) & (truth >= .3) & (truth <= 5)
        rows.append(dict(view=index+1, flat_truth_fraction=float((eligible & (variance <= 4)).sum()/max(1, eligible.sum()))))
    return rows


def select_source(repo):
    # The upstream projects both name their implementation package "core".
    for name in list(sys.modules):
        if name == 'core' or name.startswith('core.') or name == 'Utils':
            del sys.modules[name]
    sys.path[:] = [path for path in sys.path if Path(path).name not in ('RAFT-Stereo', 'FoundationStereo')]
    sys.path.insert(0, str(repo))


class StereoModel:
    def __init__(self, name, cache, iterations=32):
        import numpy as np
        import torch
        self.torch, self.iterations, self.name = torch, iterations, name
        self.foundation = name.startswith('foundation')
        repo = cache/('FoundationStereo' if self.foundation else 'RAFT-Stereo')
        select_source(repo)
        if self.foundation:
            from omegaconf import OmegaConf
            os.environ['XFORMERS_DISABLED'] = '1'
            # Load the exact two upstream utility functions used by the model,
            # without importing unrelated point-cloud/visualization packages.
            utilities = ModuleType('Utils')
            utilities.__dict__.update(np=np, torch=torch)
            tree = ast.parse((repo/'Utils.py').read_text(encoding='utf-8'))
            functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                         and node.name in ('get_resize_keep_aspect_ratio', 'freeze_model')]
            if len(functions) != 2:
                raise ValueError('Unexpected FoundationStereo utility contract')
            exec(compile(ast.Module(body=functions, type_ignores=[]), str(repo/'Utils.py'), 'exec'), utilities.__dict__)
            sys.modules['Utils'] = utilities
            from core.foundation_stereo import FoundationStereo
            from core.utils.utils import InputPadder
            folder = repo/'pretrained_models/11-33-40'
            checkpoint = folder/'model_best_bp2.pth'
            cfg = OmegaConf.load(folder/'cfg.yaml')
            cfg.valid_iters, cfg.mixed_precision = iterations, True
            import timm
            create_model, hub_load = timm.create_model, torch.hub.load
            # Full strict checkpoint loading supplies these backbone weights.
            # Use the upstream bundled DINO source and avoid redundant downloads.
            def local_backbone(*args, **kwargs):
                kwargs['pretrained'] = False
                return create_model(*args, **kwargs)
            def local_dino(repo_or_dir, model, *args, **kwargs):
                if repo_or_dir == 'facebookresearch/dinov2':
                    repo_or_dir, kwargs['source'] = str(repo/'dinov2'), 'local'
                return hub_load(repo_or_dir, model, *args, **kwargs)
            try:
                timm.create_model, torch.hub.load = local_backbone, local_dino
                self.model = FoundationStereo(cfg)
            finally:
                timm.create_model, torch.hub.load = create_model, hub_load
            from collections import defaultdict
            from typing import Any
            from omegaconf.nodes import AnyNode
            from omegaconf.base import Metadata, ContainerMetadata
            from omegaconf import ListConfig, DictConfig
            allowed = [defaultdict, dict, int, list, Any, AnyNode, Metadata, ContainerMetadata,
                       ListConfig, DictConfig, np.dtype, type(np.dtype(np.float64)),
                       (np._core.multiarray.scalar, 'numpy.core.multiarray.scalar')]
            with torch.serialization.safe_globals(allowed):
                state = torch.load(checkpoint, map_location='cpu', weights_only=True)['model']
            self.model.load_state_dict(state, strict=True)
            self.padder, self.sign = InputPadder, 1.
        else:
            from core.raft_stereo import RAFTStereo
            from core.utils.utils import InputPadder
            files = {'raft_middlebury': 'raftstereo-middlebury.pth',
                     'raft_rvc': 'iraftstereo_rvc.pth', 'raft_sceneflow': 'raftstereo-sceneflow.pth'}
            checkpoint = repo/'models'/files[name]
            args = SimpleNamespace(hidden_dims=[128]*3, corr_implementation='reg',
                                   shared_backbone=False, corr_levels=4, corr_radius=4,
                                   n_downsample=2, context_norm='instance' if name == 'raft_rvc' else 'batch',
                                   slow_fast_gru=False, n_gru_layers=3, mixed_precision=False)
            self.model = RAFTStereo(args)
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            self.model.load_state_dict({key.removeprefix('module.'): value for key, value in state.items()}, strict=True)
            self.padder, self.sign = InputPadder, -1.
        self.model.to('cuda').eval()
        self.metadata = dict(source=str(repo.resolve()), commit=subprocess.check_output(
            ['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
            checkpoint=str(checkpoint.resolve()), checkpoint_sha256=sha256(checkpoint), iterations=iterations,
            correlation='upstream PyTorch correlation' if self.foundation else 'pure PyTorch reg',
            input=('grayscale repeated into RGB' if name.endswith('_gray') else 'RGB')+' uint8 0-255; upstream internal normalization',
            precision='mixed FP16' if self.foundation else 'FP32', forward_sign=self.sign)
        self.warmed = False

    def predict(self, left, right):
        import cv2
        import numpy as np
        torch = self.torch
        if self.name.endswith('_gray'):
            left, right = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in (left, right)]
        def tensor(image):
            if image.ndim == 2:
                image = np.repeat(image[..., None], 3, axis=2)
            return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None].float().cuda()
        def single(a, b):
            a, b = tensor(a), tensor(b)
            padder = self.padder(a.shape, divis_by=32)
            a, b = padder.pad(a, b)
            with torch.inference_mode(), torch.autocast('cuda', enabled=self.foundation):
                raw = self.model(a, b, iters=self.iterations, test_mode=True)
                if not self.foundation:
                    raw = raw[1]
            return padder.unpad(raw.float())[0, 0].cpu().numpy()*self.sign
        if not self.warmed:
            single(left, right)
            self.warmed = True
        torch.cuda.synchronize()
        start = time.perf_counter()
        disparity = single(left, right)
        reverse = np.flip(single(np.flip(right, 1), np.flip(left, 1)), 1).copy()
        torch.cuda.synchronize()
        return disparity, reverse, (time.perf_counter()-start)*1000


def checked_prediction(metric, disparity, reverse, ms, policy='pixel'):
    import numpy as np
    import evaluate as audit
    if not np.isfinite(disparity).all() or not np.isfinite(reverse).all():
        raise ValueError('Nonfinite disparity from replacement model')
    depth, dense, support, valid, lr = audit._depth_from_disparity(metric, disparity, reverse)
    if policy == 'metric':
        yy, xx = np.indices(disparity.shape)
        xr = np.rint(xx-disparity).astype(np.int32).clip(0, metric.width-1)
        denominator = reverse[yy, xr]-metric.disparity_offset
        right_depth = np.divide(metric.depth_factor, denominator,
                                out=np.full_like(depth, np.nan), where=denominator > 0)
        # Apply the same distance confidence requirement at near and far range.
        tolerance = np.minimum(.1, .05*depth)
        valid = support & np.isfinite(right_depth) & (np.abs(depth-right_depth) <= tolerance)
    return dict(d=disparity, reverse=reverse, depth=depth, valid=valid, ms=ms,
                dense=dense, support=support, lr_error=lr)


def score(items, truths, indices, baseline=None):
    import numpy as np
    absolute, common_before, common_after, rows = [], [], [], []
    for index in indices:
        item, truth = items[index], truths[index]
        eligible = np.isfinite(truth) & (truth >= .3) & (truth <= 5)
        valid = item['valid'] & eligible & np.isfinite(item['depth'])
        error = np.abs(item['depth']-truth)
        absolute.append(error[valid].astype(np.float64))
        row = dict(view=index+1, truth_pixels=int(eligible.sum()), valid_pixels=int(valid.sum()),
                   coverage=float(valid.sum()/max(1, eligible.sum())), output_image_coverage=float(item['valid'].mean()),
                   accurate_0_1m_coverage=float((valid & (error <= .1)).sum()/max(1, eligible.sum())),
                   accurate_0_25m_coverage=float((valid & (error <= .25)).sum()/max(1, eligible.sum())),
                   false_in_range_pixels=int((item['valid'] & ~eligible).sum()),
                   mae_m=float(error[valid].mean()) if valid.any() else None, compute_ms=item['ms'])
        rows.append(row)
        if baseline is not None:
            common = valid & baseline[index]['valid'] & np.isfinite(baseline[index]['depth'])
            common_before.append(np.abs(baseline[index]['depth'][common]-truth[common]).astype(np.float64))
            common_after.append(error[common].astype(np.float64))
    errors = np.concatenate(absolute)
    result = dict(mean_truth_coverage=float(np.mean([row['coverage'] for row in rows])),
                  mean_output_image_coverage=float(np.mean([row['output_image_coverage'] for row in rows])),
                  accurate_0_1m_truth_coverage=float(np.mean([row['accurate_0_1m_coverage'] for row in rows])),
                  accurate_0_25m_truth_coverage=float(np.mean([row['accurate_0_25m_coverage'] for row in rows])),
                  evaluated_pixels=int(errors.size), pooled_mae_m=float(errors.mean()) if errors.size else None,
                  false_in_range_pixels=sum(row['false_in_range_pixels'] for row in rows),
                  pooled_p95_error_m=float(np.percentile(errors, 95)) if errors.size else None,
                  mean_stereo_ms=float(np.mean([row['compute_ms'] for row in rows])), views=rows)
    if baseline is not None:
        old, new = np.concatenate(common_before), np.concatenate(common_after)
        result['common_pixels'] = dict(count=int(new.size), baseline_mae_m=float(old.mean()) if new.size else None,
                                       candidate_mae_m=float(new.mean()) if new.size else None)
    return result


def fuse(primary, fallback):
    """Use the accepted primary prediction first; retain accepted fallback elsewhere."""
    import numpy as np
    use = primary['valid']
    return dict(d=np.where(use, primary['d'], fallback['d']),
                depth=np.where(use, primary['depth'], fallback['depth']),
                valid=primary['valid'] | fallback['valid'],
                source=use.astype(np.uint8), ms=primary['ms']+fallback['ms'])


def validation_limitations(aggregate):
    checks = {}
    for split in ('previous_validation', 'new_validation'):
        before, after = aggregate[split]['gamma'], aggregate[split]['replacement']
        false_increase = after['false_in_range_pixels']-before['false_in_range_pixels']
        checks[split] = dict(
            p95_before_m=before['pooled_p95_error_m'], p95_after_m=after['pooled_p95_error_m'],
            p95_not_worse=after['pooled_p95_error_m'] <= before['pooled_p95_error_m'],
            out_of_range_false_predictions_before=before['false_in_range_pixels'],
            out_of_range_false_predictions_after=after['false_in_range_pixels'],
            false_prediction_increase_within_0_5_percent_image=false_increase <= .005*len(after['views'])*640*352,
            views_with_worse_mae=[new['view'] for old, new in zip(before['views'], after['views'])
                                  if old['mae_m'] is not None and new['mae_m'] is not None and new['mae_m'] > old['mae_m']])
    return dict(validation_guardrails_passed=all(value['p95_not_worse'] and
                value['false_prediction_increase_within_0_5_percent_image'] for value in checks.values()),
                checks=checks,
                confidence_limit='Left/right agreement is a consistency check, not proof of a correct distance; correlated errors can pass both models.',
                deployment_acceptance=False,
                real_distance_limit='Checkerboard PnP uses the same calibration; independent real distance ground truth remains required.')


def baseline_predictions(arrays, metric, include_sgbm=True):
    import cv2
    import numpy as np
    import onnxruntime as ort
    from evaluate_portrait import _run
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
    path = EXPERIMENT/'models/lightstereo_640_352.onnx'
    manifest = json.loads((EXPERIMENT/'manifest.json').read_text(encoding='utf-8'))['files']
    if sha256(path) != manifest['models\\lightstereo_640_352.onnx']['sha256']:
        raise ValueError('LightStereo model differs from the existing manifest')
    session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    _run(session, arrays['left'][0], arrays['right'][0], native=True)
    result = {'original': {}, 'gamma_pixel': {}, 'gamma_metric': {}, 'SGBM': {}}
    lut = np.round((np.arange(256)/255)**.45*255).clip(0, 255).astype(np.uint8)
    for index, (left, right) in enumerate(zip(arrays['left'], arrays['right'])):
        for name, l, r in [('original', left, right), ('gamma', cv2.LUT(left, lut), cv2.LUT(right, lut))]:
            d, rev, fwd_ms, rev_ms, _, _ = _run(session, l, r, native=True)
            if name == 'original':
                result[name][index] = checked_prediction(metric, d, rev, fwd_ms+rev_ms)
            else:
                for policy in ('pixel', 'metric'):
                    result['gamma_'+policy][index] = checked_prediction(metric, d, rev, fwd_ms+rev_ms, policy)
        if include_sgbm:
            start = time.perf_counter()
            dl, dr = metric.disparities(left, right)
            z, valid = metric.metric_values(dl, dr, left)
            result['SGBM'][index] = dict(d=dl, depth=z, valid=valid, ms=(time.perf_counter()-start)*1000)
    return result


def real_photo_check(model, selected_policy, hybrid, metric):
    import cv2
    import numpy as np
    import onnxruntime as ort
    import evaluate as audit
    from evaluate_portrait import _run
    report = audit._json(CALIBRATION.parent/'report.json')
    detections = audit._json(CALIBRATION.parent/'detections.json')
    original_metric = audit.MetricDepth(CALIBRATION)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads, opts.inter_op_num_threads = 4, 1
    session = ort.InferenceSession(str(EXPERIMENT/'models/lightstereo_640_352.onnx'),
                                   sess_options=opts, providers=['CPUExecutionProvider'])
    lut = np.round((np.arange(256)/255)**.45*255).clip(0, 255).astype(np.uint8)
    rows, errors = [], {'gamma': [], 'replacement': []}
    for image_id in report['heldout_ids']:
        folder = EXPERIMENT.parent/'stereo/captures'/image_id
        left, right = [cv2.imdecode(np.frombuffer((folder/(side+'.png')).read_bytes(), np.uint8),
                                   cv2.IMREAD_GRAYSCALE) for side in ('left', 'right')]
        if left is None or right is None:
            raise FileNotFoundError(f'Missing real stereo photos: {image_id}')
        left, right = [image[:, 4:-4] for image in original_metric.rectify(left, right)]
        truth, _ = audit._plane_reference(original_metric, detections, image_id)
        truth = truth[:, 4:-4]
        d, rev, fwd_ms, rev_ms, _, _ = _run(session, cv2.LUT(left, lut), cv2.LUT(right, lut), native=True)
        gamma = checked_prediction(metric, d, rev, fwd_ms+rev_ms)
        primary = checked_prediction(metric, *model.predict(cv2.cvtColor(left, cv2.COLOR_GRAY2RGB),
                                                            cv2.cvtColor(right, cv2.COLOR_GRAY2RGB)), policy=selected_policy)
        replacement = fuse(primary, gamma) if hybrid else primary
        common = np.isfinite(truth) & gamma['valid'] & replacement['valid']
        row = dict(id=image_id, common_board_pixels=int(common.sum()))
        for name, item in [('gamma', gamma), ('replacement', replacement)]:
            error = np.abs(item['depth'][common]-truth[common])
            errors[name].append(error.astype(np.float64))
            row[name] = dict(image_coverage=float(item['valid'].mean()),
                             common_plane_mae_m=float(error.mean()) if error.size else None)
        rows.append(row)
    aggregate = {}
    for name in errors:
        error = np.concatenate(errors[name])
        aggregate[name] = dict(mean_image_coverage=float(np.mean([row[name]['image_coverage'] for row in rows])),
                               common_board_pixels=int(error.size),
                               common_plane_mae_m=float(error.mean()) if error.size else None,
                               common_plane_p90_error_m=float(np.percentile(error, 90)) if error.size else None)
    return dict(interpretation='Existing five checkerboard pairs: coverage and calibration-dependent PnP plane consistency only; not independent real-world distance ground truth.',
                aggregate=aggregate, views=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=CACHE)
    parser.add_argument('--methods', default='raft_middlebury,raft_rvc,foundation_rgb,foundation_gray')
    parser.add_argument('--iterations', type=int, default=32)
    parser.add_argument('--probe', action='store_true', help='Evaluate one view without changing saved results')
    parser.add_argument('--capture', action='store_true', help='Capture five new validation views with the existing WSL simulator first')
    parser.add_argument('--distribution', help='Override the WSL distribution in .local.json')
    parser.add_argument('--output', type=Path, default=ROOT/'results/lightstereo')
    args = parser.parse_args()
    import cv2
    import numpy as np
    import torch
    warnings.filterwarnings('ignore', message='.*torch.cuda.amp.autocast.*', category=FutureWarning)
    warnings.filterwarnings('ignore', message='torch.meshgrid:.*', category=UserWarning)
    warnings.filterwarnings('ignore', message='xFormers is .*', category=UserWarning)
    sys.path.insert(0, str(EXPERIMENT))
    from portrait import CroppedMetric
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('This comparison requires the existing CUDA-capable PyTorch environment')
    metric = CroppedMetric(CALIBRATION)
    out = args.output.resolve()
    if args.capture:
        settings = ROOT/'.local.json'
        distribution = args.distribution or (json.loads(settings.read_text(encoding='utf-8-sig')).get('distribution', 'Ubuntu-24.04') if settings.exists() else 'Ubuntu-24.04')
        path = subprocess.check_output(['wsl.exe', '-d', distribution, '--exec', 'wslpath', '-a', str(ROOT).replace('\\', '/')],
                                       text=True, encoding='utf-8').strip()
        subprocess.run(['wsl.exe', '-d', distribution, '--exec', 'bash', path+'/scripts/launch.sh', 'lightstereo-extend'], check=True)
    input_hash = sha256(out/'frames.npz')
    with np.load(out/'frames.npz', allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    development = [0] if args.probe else list(range(5))
    if args.probe:
        gamma = {i: dict(depth=arrays['prediction'][i], valid=arrays['prediction_valid'][i], ms=0.) for i in development}
        baselines = None
    else:
        if len(arrays['left']) != 15:
            raise ValueError('Use --capture to collect the five new replacement validation views')
        baselines = baseline_predictions(arrays, metric)
        gamma = baselines['gamma_pixel']
    results, predictions, metadata = {}, {}, {}
    for name in args.methods.split(','):
        torch.cuda.reset_peak_memory_stats()
        model = StereoModel(name, args.cache, args.iterations)
        raw = {}
        for index in development:
            raw[index] = model.predict(arrays['left_rgb'][index], arrays['right_rgb'][index])
        policies = ('pixel',) if args.probe else ('pixel', 'metric')
        for policy in policies:
            label = name if args.probe else name+':'+policy
            predictions[label] = {i: checked_prediction(metric, *raw[i], policy=policy) for i in development}
            results[label] = score(predictions[label], arrays['truth'], development, gamma)
            print(json.dumps(dict(method=label, coverage=results[label]['mean_truth_coverage'],
                                  accurate_0_1m_coverage=results[label]['accurate_0_1m_truth_coverage'],
                                  mae_m=results[label]['pooled_mae_m'], false_in_range_pixels=results[label]['false_in_range_pixels'])), flush=True)
            if not args.probe:
                fused = label+'+gamma'
                predictions[fused] = {i: fuse(predictions[label][i], gamma[i]) for i in development}
                results[fused] = score(predictions[fused], arrays['truth'], development, gamma)
                print(json.dumps(dict(method=fused, coverage=results[fused]['mean_truth_coverage'],
                                      accurate_0_1m_coverage=results[fused]['accurate_0_1m_truth_coverage'],
                                      mae_m=results[fused]['pooled_mae_m'], false_in_range_pixels=results[fused]['false_in_range_pixels'])), flush=True)
        metadata[name] = model.metadata
        metadata[name]['peak_gpu_allocated_bytes'] = torch.cuda.max_memory_allocated()
        del model
        torch.cuda.empty_cache()
    if args.probe:
        return
    reference = score(gamma, arrays['truth'], development)
    for name in ('original', 'gamma_pixel', 'gamma_metric', 'SGBM'):
        results[name] = score(baselines[name], arrays['truth'], development, gamma)
        predictions[name] = {i: baselines[name][i] for i in development}
    development_report = json.loads((out/'acceptance.json').read_text(encoding='utf-8'))
    development_report['replacement_study'] = dict(status='DEVELOPMENT_COMPLETE',
        input_frames_sha256=input_hash, raw_capture_sha256=raw_capture_sha256(arrays), development_views=[1, 2, 3, 4, 5],
        development_trials=results, metadata=metadata)
    write_json(out/'acceptance.json', development_report)
    candidates = []
    for name, result in results.items():
        common = result['common_pixels']
        if (result['mean_truth_coverage'] > reference['mean_truth_coverage'] and
                result['accurate_0_1m_truth_coverage'] > reference['accurate_0_1m_truth_coverage'] and
                result['pooled_mae_m'] < reference['pooled_mae_m'] and
                result['pooled_p95_error_m'] < reference['pooled_p95_error_m'] and
                result['false_in_range_pixels'] <= reference['false_in_range_pixels']+.005*len(development)*640*352 and
                common['count'] > 0 and common['candidate_mae_m'] < common['baseline_mae_m']):
            candidates.append(name)
    if not candidates:
        raise RuntimeError('No candidate jointly improved coverage and accuracy under the development checks')
    selected = max(candidates, key=lambda name: results[name]['accurate_0_1m_truth_coverage'])
    development_report['replacement_study'].update(selected=selected, status='SELECTION_FROZEN',
                                                  frozen_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(out/'acceptance.json', development_report)
    print('Frozen development selection: '+selected, flush=True)
    chosen = predictions[selected].copy()
    validation = list(range(5, 10))
    new_validation = list(range(10, 15))
    if selected in baselines:
        chosen = baselines[selected]
        model = None
        real_check = dict(status='BASELINE_METHOD_SELECTED')
    else:
        primary_name, policy = selected.removesuffix('+gamma').split(':')
        hybrid = selected.endswith('+gamma')
        model = StereoModel(primary_name, args.cache, args.iterations)
        for index in validation+new_validation:
            item = checked_prediction(metric, *model.predict(arrays['left_rgb'][index], arrays['right_rgb'][index]), policy=policy)
            chosen[index] = fuse(item, gamma[index]) if hybrid else item
        real_check = real_photo_check(model, policy, hybrid, metric)
    aggregate = {}
    for split, indices in [('development', development), ('previous_validation', validation), ('new_validation', new_validation)]:
        aggregate[split] = {name: score(values, arrays['truth'], indices, gamma if name != 'gamma' else None)
                            for name, values in [('original', baselines['original']), ('gamma', gamma), ('replacement', chosen), ('SGBM', baselines['SGBM'])]}
    before, after = aggregate['new_validation']['gamma'], aggregate['new_validation']['replacement']
    improved = (after['mean_truth_coverage'] > before['mean_truth_coverage'] and
                after['accurate_0_1m_truth_coverage'] > before['accurate_0_1m_truth_coverage'] and
                after['pooled_mae_m'] < before['pooled_mae_m'] and
                after['common_pixels']['candidate_mae_m'] < after['common_pixels']['baseline_mae_m'])
    report = json.loads((out/'acceptance.json').read_text(encoding='utf-8'))
    report['replacement_study'] = dict(status='IMPROVED_ON_NEW_VIEWS' if improved else 'NO_JOINT_IMPROVEMENT_ON_NEW_VIEWS',
        input_frames_sha256=input_hash, raw_capture_sha256=raw_capture_sha256(arrays),
        low_texture_validation_evidence=low_texture_evidence(arrays),
        frozen_at_utc=development_report['replacement_study']['frozen_at_utc'],
        development_selection_frozen_before_validation=True,
        selected=selected, development_views=[1, 2, 3, 4, 5], previous_validation_views=[6, 7, 8, 9, 10],
        new_validation_views=[11, 12, 13, 14, 15], development_trials=results, aggregate=aggregate,
        selection='Freeze on views 1-5: improve coverage, <=0.1m coverage, MAE/p95/common MAE; limit extra out-of-range false predictions to 0.5% of image pixels; then maximize <=0.1m truth coverage.',
        confidence=dict(pixel='LR disparity <=1px', metric='LR optical-Z difference <=min(0.1m, 5% of predicted Z)',
                        fusion='Accepted primary prediction takes priority; accepted gamma fallback otherwise; keep source identity.'),
        metadata=metadata, real_camera_sanity=real_check,
        versions=dict(python=sys.version.split()[0], numpy=np.__version__, opencv=cv2.__version__, torch=torch.__version__,
                      cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                      onnxruntime=version('onnxruntime'), timm=version('timm'), omegaconf=version('omegaconf')),
        runtime=dict(python=str(Path(sys.executable).resolve()), cache=str(args.cache.resolve())),
        interpretation='Simulator optical-Z truth is evaluation only. The five new validation poses were fixed before replacement inference. Real checkerboard plane consistency is not independent real distance accuracy. GPU inference is not RK3588 NPU performance.')
    report['replacement_study']['limitations'] = validation_limitations(aggregate)
    arrays.update(baseline_prediction=np.stack([baselines['original'][i]['depth'] for i in range(15)]),
                  baseline_valid=np.stack([baselines['original'][i]['valid'] for i in range(15)]),
                  prediction=np.stack([gamma[i]['depth'] for i in range(15)]),
                  prediction_valid=np.stack([gamma[i]['valid'] for i in range(15)]),
                  disparity=np.stack([gamma[i]['d'] for i in range(15)]),
                  reverse_disparity=np.stack([gamma[i]['reverse'] for i in range(15)]),
                  replacement_prediction=np.stack([chosen[i]['depth'] for i in range(15)]),
                  replacement_valid=np.stack([chosen[i]['valid'] for i in range(15)]),
                  replacement_disparity=np.stack([chosen[i]['d'] for i in range(15)]),
                  replacement_source=np.stack([chosen[i].get('source', chosen[i]['valid'].astype(np.uint8)) for i in range(15)]))
    if sha256(out/'frames.npz') != input_hash:
        raise RuntimeError('Source frames changed during comparison; refusing to overwrite')
    from evaluate_portrait import _distance_legend
    import evaluate as audit
    panels = []
    for index in (2, 12):
        truth = arrays['truth'][index].copy()
        truth[(truth < .3) | (truth > 5)] = np.nan
        views = [audit._pane(cv2.cvtColor(arrays['left_rgb'][index], cv2.COLOR_RGB2BGR), f'view {index+1} RGB'),
                 audit._pane(audit._color_depth(truth, (.3, 5)), 'Gazebo truth')]
        for label, values in [('LightStereo gamma', gamma), (selected, chosen)]:
            depth = values[index]['depth'].copy()
            depth[~values[index]['valid']] = np.nan
            views.append(audit._pane(audit._color_depth(depth, (.3, 5)), label))
        for label, values in [('gamma error', gamma), ('replacement error', chosen)]:
            error = np.abs(values[index]['depth']-truth)
            color = cv2.applyColorMap(np.nan_to_num(np.clip(error/.5, 0, 1)*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            color[~(np.isfinite(error) & values[index]['valid'])] = 0
            views.append(audit._pane(color, label+' 0-0.5m'))
        panels.append(np.hstack(views))
    np.savez_compressed(out/'frames.npz', **arrays)
    audit._write_png(out/'comparison.png', np.vstack(panels+[_distance_legend(panels[0].shape[1], (.3, 5))]))
    report.update(status='COMPLETED', current_experiment='replacement_study', legacy_metrics_views=list(range(1, 11)),
                  finished_at_utc=datetime.now(timezone.utc).isoformat(),
                  output_sha256=dict(frames=sha256(out/'frames.npz'), comparison=sha256(out/'comparison.png')))
    write_json(out/'acceptance.json', report)
    print(json.dumps(dict(selected=selected, status=report['replacement_study']['status'], new_validation=aggregate['new_validation'], real_camera_sanity=real_check), indent=2), flush=True)


if __name__ == '__main__':
    main()
