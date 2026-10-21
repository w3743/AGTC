#!/usr/bin/env python3
"""Cross-model and resolution experiments with a separately held-out capture."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import warnings

import cv2
import numpy as np
import torch

import compare_stereo as stereo

SOURCES = {'raft1': ('raft_middlebury', 1), 'raft2': ('raft_middlebury', 2),
           'foundation1': ('foundation_gray', 1), 'foundation2': ('foundation_gray', 2)}


def scaled_prediction(model, left, right, scale):
    height, width = left.shape[:2]
    if scale != 1:
        left, right = [cv2.resize(x, (width*scale, height*scale), interpolation=cv2.INTER_LINEAR)
                       for x in (left, right)]
    d, reverse, milliseconds = model.predict(left, right)
    if scale != 1:
        d, reverse = [cv2.resize(x, (width, height), interpolation=cv2.INTER_AREA)/scale
                      for x in (d, reverse)]
    return d, reverse, milliseconds


def predictions(arrays, names, indices, metric, cache, metadata):
    result = {}
    for name in names:
        method, scale = SOURCES[name]
        prefix = 'development_'+name
        cached = all(prefix+'_'+key in arrays for key in ('d', 'reverse', 'ms')) and indices == list(range(15))
        info = metadata.get(name, {})
        if cached:
            cached = (info.get('inference_scale') == scale and info.get('iterations') == 32 and
                      Path(info.get('checkpoint', '')).is_file() and
                      stereo.sha256(Path(info['checkpoint'])) == info.get('checkpoint_sha256') and
                      subprocess.check_output(['git', '-C', info['source'], 'rev-parse', 'HEAD'], text=True).strip() == info['commit'])
        values = {}
        if cached:
            for index in indices:
                values[index] = stereo.checked_prediction(metric, arrays[prefix+'_d'][index],
                    arrays[prefix+'_reverse'][index], float(arrays[prefix+'_ms'][index]), 'metric')
        else:
            model = stereo.StereoModel(method, cache)
            torch.cuda.reset_peak_memory_stats()
            for index in indices:
                raw = scaled_prediction(model, arrays['left_rgb'][index], arrays['right_rgb'][index], scale)
                values[index] = stereo.checked_prediction(metric, *raw, policy='metric')
            metadata[name] = dict(**model.metadata, inference_scale=scale,
                scale_interpolation='bilinear RGB upsample; area disparity downsample divided by scale',
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
            del model
            torch.cuda.empty_cache()
        result[name] = values
        if indices == list(range(15)):
            for key in ('d', 'reverse', 'ms'):
                arrays[prefix+'_'+key] = np.asarray([values[index][key] for index in indices])
        print(f'Finished {name}: {len(indices)} views'+(' (cached)' if cached else ''), flush=True)
    return result


def previous_predictions(arrays, indices, raft, metric):
    """Reproduce the previous frozen candidate, including its Gamma fallback."""
    import onnxruntime as ort
    from evaluate_portrait import _run
    opts = ort.SessionOptions()
    opts.intra_op_num_threads, opts.inter_op_num_threads = 4, 1
    model_path = stereo.EXPERIMENT/'models/lightstereo_640_352.onnx'
    manifest = json.loads((stereo.EXPERIMENT/'manifest.json').read_text(encoding='utf-8'))['files']
    if stereo.sha256(model_path) != manifest['models\\lightstereo_640_352.onnx']['sha256']:
        raise ValueError('LightStereo model differs from its manifest')
    session = ort.InferenceSession(str(model_path),
        sess_options=opts, providers=['CPUExecutionProvider'])
    lut = np.round(255*(np.arange(256)/255)**.45).astype(np.uint8)
    result = {}
    for index in indices:
        d, reverse, forward_ms, reverse_ms, _, _ = _run(session,
            cv2.LUT(arrays['left'][index], lut), cv2.LUT(arrays['right'][index], lut), native=True)
        gamma = stereo.checked_prediction(metric, d, reverse, forward_ms+reverse_ms)
        result[index] = stereo.fuse(raft[index], gamma)
    return result


def candidate(spec, sources, previous, indices, metric):
    output = {}
    for index in indices:
        if spec['kind'] == 'single':
            value = sources[spec['source']][index]
            if spec['policy'] == 'pixel':
                value = stereo.checked_prediction(metric, value['d'], value['reverse'], value['ms'])
            output[index] = value
            continue
        a, b = sources[spec['a']][index], sources[spec['b']][index]
        if spec.get('confidence') == 'pixel':
            a, b = [stereo.checked_prediction(metric, value['d'], value['reverse'], value['ms']) for value in (a, b)]
        tolerance = np.minimum(spec['absolute_m'], spec['relative']*np.minimum(a['depth'], b['depth']))
        agreement = a['support'] & b['support'] & (np.abs(a['depth']-b['depth']) <= tolerance)
        accepted = agreement & (a['valid'] | b['valid'])
        d = (a['d']+b['d'])/2 if spec['estimate'] == 'mean_disparity' else b['d']
        depth = np.divide(metric.depth_factor, d-metric.disparity_offset,
                          out=np.full_like(d, np.nan), where=(d-metric.disparity_offset) > 0)
        source = accepted.astype(np.uint8)
        if spec['retain_agreeing_previous']:
            old = previous[index]
            old_tolerance = np.minimum(spec['absolute_m'], spec['relative']*old['depth'])
            old_agreement = np.abs(old['depth']-b['depth']) <= old_tolerance
            if spec.get('fallback') == 'foundation':
                retain = old['valid'] & b['support'] & old_agreement
            elif spec.get('fallback') == 'reject_conflict':
                retain = old['valid'] & (~b['valid'] | old_agreement)
            else:
                retain = old['valid'] & agreement & (np.abs(old['depth']-a['depth']) <= tolerance)
                retain &= np.abs(old['depth']-b['depth']) <= tolerance
            d = np.where(accepted, d, old['d'])
            depth = np.where(accepted, depth, old['depth'])
            accepted |= retain
        milliseconds = a['ms']+b['ms']
        if spec['retain_agreeing_previous']:
            milliseconds += previous[index]['ms']
            if spec['a'] == 'raft1':
                milliseconds -= a['ms']  # Shared RAFT inference is performed once.
        output[index] = dict(d=d, depth=depth, valid=accepted, source=source, ms=milliseconds)
    return output


def specs():
    result = {}
    for source in SOURCES:
        for policy in ('metric', 'pixel'):
            result[f'{source}_{policy}'] = dict(kind='single', source=source, policy=policy)
    for a in ('raft1', 'raft2'):
        for b in ('foundation1', 'foundation2'):
            for absolute, relative in ((.1, .05), (.25, .1)):
                for estimate in ('mean_disparity', 'foundation'):
                    for retain in (False, True):
                        name = f'{a}_{b}_{absolute}_{estimate}'+('_retain' if retain else '')
                        result[name] = dict(kind='consensus', a=a, b=b, absolute_m=absolute, relative=relative,
                            estimate=estimate, retain_agreeing_previous=retain)
    # The native models disagree by more than 0.25m on some otherwise accurate
    # Foundation predictions. Screen wider cross-model bounds on development only.
    for absolute, relative in ((.5, .15), (.75, .2), (1., .25)):
        for estimate in ('mean_disparity', 'foundation'):
            for retain in (False, True):
                for confidence in ('metric', 'pixel'):
                    name = f'raft1_foundation1_{absolute}_{estimate}_{confidence}'+('_retain' if retain else '')
                    result[name] = dict(kind='consensus', a='raft1', b='foundation1', absolute_m=absolute,
                        relative=relative, estimate=estimate, retain_agreeing_previous=retain, confidence=confidence)
    for absolute, relative in ((.25, .1), (.5, .15), (.75, .2)):
        for fallback in ('foundation', 'reject_conflict'):
            name = f'raft1_foundation1_{absolute}_mean_disparity_{fallback}'
            result[name] = dict(kind='consensus', a='raft1', b='foundation1', absolute_m=absolute, relative=relative,
                estimate='mean_disparity', retain_agreeing_previous=True, confidence='pixel', fallback=fallback)
    return result


def checks(before, after):
    common = after['common_pixels']
    return dict(coverage=after['mean_truth_coverage'] > before['mean_truth_coverage'],
        accurate_0_1m_coverage=after['accurate_0_1m_truth_coverage'] > before['accurate_0_1m_truth_coverage'],
        mae=after['pooled_mae_m'] < before['pooled_mae_m'],
        p95=after['pooled_p95_error_m'] <= before['pooled_p95_error_m'],
        out_of_range_false_predictions=after['false_in_range_pixels'] <= before['false_in_range_pixels'],
        common_pixel_mae=common['count'] > 0 and common['candidate_mae_m'] < common['baseline_mae_m'])


def compact(value):
    return {key: value[key] for key in ('mean_truth_coverage', 'accurate_0_1m_truth_coverage',
        'pooled_mae_m', 'pooled_p95_error_m', 'false_in_range_pixels', 'mean_stereo_ms')}


def distance_bands(depth, valid, truth):
    rows = []
    for low, high in ((.3, 1), (1, 2), (2, 3), (3, 5)):
        eligible = np.isfinite(truth) & (truth >= low) & ((truth <= high) if high == 5 else (truth < high))
        selected = eligible & valid
        errors = np.abs(depth[selected]-truth[selected])
        rows.append(dict(range_m=[low, high], truth_pixels=int(eligible.sum()), valid_pixels=int(selected.sum()),
            coverage=float(selected.sum()/max(1, eligible.sum())), mae_m=float(errors.mean()) if errors.size else None,
            p95_m=float(np.percentile(errors, 95)) if errors.size else None))
    return rows


def draw_comparison(arrays, previous, refined, path):
    import evaluate as audit
    from evaluate_portrait import _distance_legend
    panels = []
    for index in (2, 11, 17):
        truth = arrays['truth'][index].copy()
        eligible = np.isfinite(truth) & (truth >= .3) & (truth <= 5)
        truth[~eligible] = np.nan
        row = [audit._pane(cv2.cvtColor(arrays['left_rgb'][index], cv2.COLOR_RGB2BGR), f'view {index+1} RGB'),
               audit._pane(audit._color_depth(truth, (.3, 5)), 'Gazebo truth')]
        for label, values in [('Previous fusion', previous), ('Refined candidate', refined)]:
            depth = values[index]['depth'].copy()
            depth[~values[index]['valid']] = np.nan
            row.append(audit._pane(audit._color_depth(depth, (.3, 5)), label))
        for label, values in [('Previous error', previous), ('Refined error', refined)]:
            error = np.abs(values[index]['depth']-truth)
            color = cv2.applyColorMap(np.nan_to_num(np.clip(error/.5, 0, 1)*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            color[~(eligible & values[index]['valid'])] = 0
            color[~eligible & values[index]['valid']] = (255, 0, 255)
            row.append(audit._pane(color, label+' 0-0.5m'))
        panels.append(np.hstack(row))
    footer = np.zeros((30, panels[0].shape[1], 3), np.uint8)
    cv2.putText(footer, 'Error panes: magenta = predicted in range but truth is outside 0.3-5m',
                (15, 21), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 0, 255), 1, cv2.LINE_AA)
    audit._write_png(path, np.vstack(panels+[_distance_legend(panels[0].shape[1], (.3, 5)), footer]))


def real_photo_check(spec, metric, cache):
    import evaluate as audit
    calibration_report = audit._json(stereo.CALIBRATION.parent/'report.json')
    detections = audit._json(stereo.CALIBRATION.parent/'detections.json')
    original_metric = audit.MetricDepth(stereo.CALIBRATION)
    inputs = {'left': [], 'right': [], 'left_rgb': [], 'right_rgb': [], 'truth': []}
    for image_id in calibration_report['heldout_ids']:
        folder = stereo.EXPERIMENT.parent/'stereo/captures'/image_id
        raw = [cv2.imdecode(np.frombuffer((folder/(side+'.png')).read_bytes(), np.uint8),
                           cv2.IMREAD_GRAYSCALE) for side in ('left', 'right')]
        if any(image is None for image in raw):
            raise ValueError(f'Cannot decode real photos: {image_id}')
        left, right = metric.rectify(*raw)
        truth, _ = audit._plane_reference(original_metric, detections, image_id)
        for key, value in [('left', left), ('right', right), ('left_rgb', cv2.cvtColor(left, cv2.COLOR_GRAY2RGB)),
                           ('right_rgb', cv2.cvtColor(right, cv2.COLOR_GRAY2RGB)), ('truth', truth[:, 4:-4])]:
            inputs[key].append(value)
    arrays = {key: np.stack(value) for key, value in inputs.items()}
    indices = list(range(len(arrays['left'])))
    names = {'raft1', spec['source']} if spec['kind'] == 'single' else {'raft1', spec['a'], spec['b']}
    sources = predictions(arrays, sorted(names), indices, metric, cache, {})
    previous = previous_predictions(arrays, indices, sources['raft1'], metric)
    refined = candidate(spec, sources, previous, indices, metric)
    rows, errors = [], {'previous': [], 'refined': []}
    for index in indices:
        truth = arrays['truth'][index]
        common = np.isfinite(truth) & previous[index]['valid'] & refined[index]['valid']
        row = dict(id=calibration_report['heldout_ids'][index], common_board_pixels=int(common.sum()))
        for name, items in [('previous', previous), ('refined', refined)]:
            error = np.abs(items[index]['depth'][common]-truth[common])
            errors[name].append(error.astype(np.float64))
            row[name] = dict(image_coverage=float(items[index]['valid'].mean()),
                             common_plane_mae_m=float(error.mean()) if error.size else None)
        rows.append(row)
    aggregate = {}
    for name in errors:
        error = np.concatenate(errors[name])
        aggregate[name] = dict(mean_image_coverage=float(np.mean([row[name]['image_coverage'] for row in rows])),
            common_board_pixels=int(error.size), common_plane_mae_m=float(error.mean()) if error.size else None,
            common_plane_p90_error_m=float(np.percentile(error, 90)) if error.size else None)
    return dict(aggregate=aggregate, views=rows,
        interpretation='Previously used five real checkerboard pairs; same-calibration PnP consistency, not independent distance ground truth.')


def appearance_experiment(metric, cache, out):
    import evaluate as audit
    from evaluate_portrait import _distance_legend
    report = json.loads((out/'acceptance.json').read_text(encoding='utf-8'))
    with np.load(out/'appearance.npz', allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    indices = list(range(5))
    results, texture, metadata, panels, previous_control, sgbm_control = {}, {}, {}, [], None, None
    bands = {}
    for condition in ('uniform', 'textured_floor', 'textured_all'):
        inputs = {key: arrays[condition+'_'+key] for key in ('left', 'right', 'left_rgb', 'right_rgb', 'truth')}
        if not np.array_equal(inputs['truth'], arrays['uniform_truth'], equal_nan=True):
            raise ValueError('Appearance experiment has different depth truth')
        sources = predictions(inputs, ['raft1'], indices, metric, cache, metadata)
        previous = previous_predictions(inputs, indices, sources['raft1'], metric)
        sgbm = {}
        import time
        for index in indices:
            start = time.perf_counter()
            d, reverse = metric.disparities(inputs['left'][index], inputs['right'][index])
            depth, valid = metric.metric_values(d, reverse, inputs['left'][index])
            sgbm[index] = dict(d=d, depth=depth, valid=valid, ms=(time.perf_counter()-start)*1000)
        results[condition] = dict(previous=stereo.score(previous, inputs['truth'], indices, previous_control),
                                  SGBM=stereo.score(sgbm, inputs['truth'], indices, sgbm_control))
        for value in results[condition].values():
            for row in value['views']:
                row['view'] += 15
        if condition == 'uniform':
            previous_control, sgbm_control = previous, sgbm
        rows = []
        for index in indices:
            gray = inputs['left'][index].astype(np.float32)
            mean = cv2.boxFilter(gray, cv2.CV_32F, (7, 7))
            variance = cv2.boxFilter(gray*gray, cv2.CV_32F, (7, 7))-mean*mean
            truth = inputs['truth'][index]
            eligible = np.isfinite(truth) & (truth >= .3) & (truth <= 5)
            rows.append(dict(view=index+16, flat_truth_fraction=float((eligible & (variance <= 4)).sum()/eligible.sum()),
                             mean_gray_u8=float(gray.mean())))
        texture[condition] = rows
        for name, values in [('previous', previous), ('SGBM', sgbm)]:
            for key in ('depth', 'valid'):
                arrays[condition+'_'+name+'_'+key] = np.stack([values[index][key] for index in indices])
        bands[condition] = distance_bands(arrays[condition+'_previous_depth'], arrays[condition+'_previous_valid'], inputs['truth'])
        index = 2  # Fixed view18, chosen before seeing the appearance results.
        truth = inputs['truth'][index].copy()
        eligible = np.isfinite(truth) & (truth >= .3) & (truth <= 5)
        truth[~eligible] = np.nan
        row = [audit._pane(cv2.cvtColor(inputs['left_rgb'][index], cv2.COLOR_RGB2BGR), condition+' RGB'),
               audit._pane(audit._color_depth(truth, (.3, 5)), 'Same geometry truth')]
        for label, values in [('RAFT + Gamma', previous), ('SGBM', sgbm)]:
            depth = values[index]['depth'].copy()
            depth[~values[index]['valid']] = np.nan
            row.append(audit._pane(audit._color_depth(depth, (.3, 5)), label))
        error = np.abs(previous[index]['depth']-truth)
        color = cv2.applyColorMap(np.nan_to_num(np.clip(error/.5, 0, 1)*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        color[~(eligible & previous[index]['valid'])] = 0
        color[~eligible & previous[index]['valid']] = (255, 0, 255)
        row.append(audit._pane(color, 'RAFT+Gamma error 0-0.5m'))
        panels.append(np.hstack(row))
        print(condition+' '+json.dumps({name: compact(value) for name, value in results[condition].items()}), flush=True)
    np.savez_compressed(out/'appearance.npz', **arrays)
    audit._write_png(out/'appearance.png', np.vstack(panels+[_distance_legend(panels[0].shape[1], (.3, 5))]))
    report['appearance_study'].update(status='COMPLETED', results=results, texture_measurements=texture, metadata=metadata,
        distance_bands=bands, calibration_sha256=stereo.sha256(stereo.CALIBRATION),
        lightstereo_sha256=stereo.sha256(stereo.EXPERIMENT/'models/lightstereo_640_352.onnx'),
        algorithm='Previous frozen RAFT Middlebury metric confidence + LightStereo gamma fallback; SGBM separately reported',
        output_sha256=stereo.sha256(out/'appearance.npz'), comparison_sha256=stereo.sha256(out/'appearance.png'),
        finished_at_utc=datetime.now(timezone.utc).isoformat(),
        limitation='This changes the simulated scene appearance; it is not a software-only improvement on unchanged images or a real active-projector validation.')
    controls = {}
    for condition in ('textured_floor', 'textured_all'):
        controls[condition] = {name: checks(results['uniform'][name], results[condition][name])
                               for name in ('previous', 'SGBM')}
    report['appearance_study']['paired_checks'] = controls
    factors = report.setdefault('study', {}).setdefault('low_coverage_factors', [])
    factor = next((item for item in factors if item['factor'] == '低纹理的大面积平面'), None)
    if factor is None:
        factor = dict(factor='低纹理的大面积平面', evidence={})
        factors.append(factor)
    factor['evidence']['appearance_control'] = dict(
        results_section='appearance_study', depth_truth_identical=True,
        unchanged_algorithm='RAFT Middlebury metric + LightStereo gamma',
        uniform=compact(results['uniform']['previous']), textured_floor=compact(results['textured_floor']['previous']),
        textured_all=compact(results['textured_all']['previous']))
    factor['status'] = '已做相同几何与PBR渲染方式下的表面纹理对照；适用范围为本仿真'
    factor['explanation'] = '低纹理区域缺少局部匹配依据。新增均匀材质、仅地面纹理、全场景纹理对照；未用真值补洞，物理深度数值完全相同。效果与限制见appearance_study。'
    report['current_experiment'] = 'appearance_study'
    stereo.write_json(out/'acceptance.json', report)


def lightstereo_experiment(metric, out):
    """Frozen raw/Gamma comparison on the exact previously captured images."""
    import evaluate as audit
    import onnxruntime as ort
    from evaluate_portrait import _distance_legend
    report = json.loads((out/'acceptance.json').read_text(encoding='utf-8'))
    calibration_hash = stereo.sha256(stereo.CALIBRATION)
    model_path = stereo.EXPERIMENT/'models/lightstereo_640_352.onnx'
    model_hash = stereo.sha256(model_path)
    inputs_hashes = {}
    for name, section in [('refinement', 'refinement_study'), ('appearance', 'appearance_study')]:
        inputs_hashes[name+'.npz'] = stereo.sha256(out/(name+'.npz'))
        if inputs_hashes[name+'.npz'] != report[section]['output_sha256']:
            raise ValueError(f'{name} inputs differ from the preceding experiment')
        if calibration_hash != report[section]['calibration_sha256']:
            raise ValueError('Calibration changed since the reference fusion was evaluated')
    if model_hash != report['appearance_study']['lightstereo_sha256']:
        raise ValueError('LightStereo differs from the preceding appearance experiment')
    study = dict(status='RUNNING', started_at_utc=datetime.now(timezone.utc).isoformat(),
        protocol='Fixed original grayscale versus identical Gamma=0.45 LUT in both eyes; no tuning or training.',
        data_scope='20 reused original views, three paired appearance conditions of five views each, five reused real checkerboards; not independent new validation.',
        model_sha256=model_hash, calibration_sha256=calibration_hash, input_sha256=inputs_hashes,
        runtime=dict(python=sys.version, onnxruntime=ort.__version__, provider='CPUExecutionProvider',
                     intra_op_threads=4, inter_op_threads=1, warmup_pairs_per_group=1),
        prediction=dict(input_hw=[640, 352], range_m=[.3, 5], lr_px=1., gamma=.45,
                        depth_factor=metric.depth_factor, disparity_offset=metric.disparity_offset),
        timing_scope='Sum of two ONNX session.run calls, excluding preprocessing, filtering, capture and display; desktop CPU, not RK3588.',
        fusion_reference='Previously saved frozen RAFT Middlebury metric confidence + LightStereo Gamma fallback; prediction files hash-checked, no inference rerun or timing comparison.',
        limitations=['Coverage is not distance accuracy.', 'Depth is camera optical-axis Z.',
                     'Appearance conditions change surface albedo, not the underlying geometry.',
                     'Real checkerboard PnP uses the same calibration and is not independent distance ground truth.',
                     'No RK3588 or physical camera run in this experiment.'])
    report['lightstereo_study'] = study
    stereo.write_json(out/'acceptance.json', report)
    output, panels = {}, []

    def run_group(inputs, prefix):
        all_values = stereo.baseline_predictions(inputs, metric, include_sgbm=False)
        values = {'original': all_values['original'], 'gamma': all_values['gamma_pixel']}
        for name, items in values.items():
            for key in ('d', 'reverse', 'valid', 'ms'):
                output[prefix+'_'+name+'_'+key] = np.asarray([items[i][key] for i in range(len(items))])
        print(f'LightStereo: {prefix}, {len(inputs["left"])} pairs, raw + Gamma finished', flush=True)
        return values

    def cached_fusion(arrays, prefix, count):
        return {i: dict(depth=arrays[prefix+'depth'][i], valid=arrays[prefix+'valid'][i], ms=0.)
                for i in range(count)}

    def summarize(items, inputs, indices, baseline=None, cached=False):
        value = stereo.score(items, inputs['truth'], indices, baseline)
        value['mean_black_image_fraction'] = 1-value['mean_output_image_coverage']
        value['false_in_range_image_fraction'] = value['false_in_range_pixels']/(len(indices)*640*352)
        value['distance_bands'] = distance_bands(np.stack([items[i]['depth'] for i in indices]),
            np.stack([items[i]['valid'] for i in indices]), inputs['truth'][indices])
        if cached:
            del value['mean_stereo_ms']
            for row in value['views']:
                del row['compute_ms']
            return value
        counts = dict(truth_pixels=0, depth_range_or_left_mask_rejected=0,
                      right_correspondence_rejected=0, lr_rejected=0, accepted=0)
        for i in indices:
            item, truth = items[i], inputs['truth'][i]
            eligible = np.isfinite(truth) & (truth >= .3) & (truth <= 5)
            counts['truth_pixels'] += int(eligible.sum())
            for key, mask in [('depth_range_or_left_mask_rejected', ~item['dense']),
                              ('right_correspondence_rejected', item['dense'] & ~item['support']),
                              ('lr_rejected', item['support'] & ~item['valid']),
                              ('accepted', item['valid'])]:
                counts[key] += int((eligible & mask).sum())
        if sum(counts[key] for key in counts if key != 'truth_pixels') != counts['truth_pixels']:
            raise ValueError('Rejection breakdown does not partition eligible truth')
        value['rejection_counts'] = counts
        return value

    def summarize_group(values, inputs, indices, fusion):
        return dict(original=summarize(values['original'], inputs, indices),
                    gamma=summarize(values['gamma'], inputs, indices, values['original']),
                    fusion=summarize(fusion, inputs, indices, values['gamma'], cached=True))

    def panel_row(label, inputs, values, fusion, index):
        truth = inputs['truth'][index].copy()
        truth[~(np.isfinite(truth) & (truth >= .3) & (truth <= 5))] = np.nan
        row = [audit._pane(cv2.cvtColor(inputs['left_rgb'][index], cv2.COLOR_RGB2BGR), label+' RGB'),
               audit._pane(audit._color_depth(truth, (.3, 5)), 'Depth truth')]
        for name, items in [('LightStereo raw', values['original']), ('LightStereo gamma .45', values['gamma']),
                            ('Saved RAFT + Gamma', fusion)]:
            depth = items[index]['depth'].copy()
            depth[~items[index]['valid']] = np.nan
            row.append(audit._pane(audit._color_depth(depth, (.3, 5)), name))
        panels.append(np.hstack(row))

    with np.load(out/'refinement.npz', allow_pickle=False) as saved:
        inputs = {key: saved[key] for key in ('left', 'right', 'left_rgb', 'truth')}
        fusion = cached_fusion(saved, 'previous_', 20)
    if inputs['left'].shape != (20, 640, 352):
        raise ValueError('Expected the fixed twenty original views')
    values = run_group(inputs, 'original_scene')
    study['original_scene'] = {}
    for label, indices in [('views_1_5', list(range(5))), ('views_6_10', list(range(5, 10))),
                           ('views_11_15', list(range(10, 15))), ('views_16_20', list(range(15, 20))),
                           ('all_20', list(range(20)))]:
        study['original_scene'][label] = summarize_group(values, inputs, indices, fusion)
    panel_row('Original view18', inputs, values, fusion, 17)
    study['gamma_checks'] = {label: checks(value['original'], value['gamma'])
                             for label, value in study['original_scene'].items()}

    study['appearance'], study['texture_effect'] = {}, {}
    control, control_truth, control_poses = None, None, None
    with np.load(out/'appearance.npz', allow_pickle=False) as saved:
        for condition in ('uniform', 'textured_floor', 'textured_all'):
            inputs = {key: saved[condition+'_'+key] for key in ('left', 'right', 'left_rgb', 'truth')}
            poses = saved[condition+'_poses']
            if control is None:
                control_truth, control_poses = inputs['truth'], poses
            elif not (np.array_equal(control_truth, inputs['truth'], equal_nan=True) and
                      np.array_equal(control_poses, poses)):
                raise ValueError('Appearance conditions do not share truth and poses')
            fusion = cached_fusion(saved, condition+'_previous_', 5)
            values = run_group(inputs, condition)
            indices = list(range(5))
            study['appearance'][condition] = summarize_group(values, inputs, indices, fusion)
            for value in study['appearance'][condition].values():
                for row in value['views']:
                    row['view'] += 15
            study['gamma_checks'][condition] = checks(study['appearance'][condition]['original'],
                                                       study['appearance'][condition]['gamma'])
            if control is None:
                control = values
            else:
                study['texture_effect'][condition] = {}
                for name in values:
                    result = summarize(values[name], inputs, indices, control[name])
                    for row in result['views']:
                        row['view'] += 15
                    result['checks'] = checks(study['appearance']['uniform'][name], result)
                    study['texture_effect'][condition][name] = result
            panel_row(condition+' view18', inputs, values, fusion, 2)

    calibration_report_path = stereo.CALIBRATION.parent/'report.json'
    detections_path = stereo.CALIBRATION.parent/'detections.json'
    calibration_report, detections = audit._json(calibration_report_path), audit._json(detections_path)
    original_metric = audit.MetricDepth(stereo.CALIBRATION)
    real_inputs = {'left': [], 'right': [], 'truth': []}
    real_hashes = {}
    for image_id in calibration_report['heldout_ids']:
        folder = stereo.EXPERIMENT.parent/'stereo/captures'/image_id
        raw = []
        for side in ('left', 'right'):
            path = folder/(side+'.png')
            real_hashes[image_id+'/'+side+'.png'] = stereo.sha256(path)
            raw.append(cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE))
        if any(image is None for image in raw):
            raise ValueError(f'Cannot decode real photos: {image_id}')
        left, right = metric.rectify(*raw)
        truth, _ = audit._plane_reference(original_metric, detections, image_id)
        for key, value in [('left', left), ('right', right), ('truth', truth[:, 4:-4])]:
            real_inputs[key].append(value)
    real_inputs = {key: np.stack(value) for key, value in real_inputs.items()}
    values = run_group(real_inputs, 'real_checkerboard')
    output['real_checkerboard_truth'] = real_inputs['truth']
    rows, errors = [], {name: [] for name in values}
    for index, image_id in enumerate(calibration_report['heldout_ids']):
        truth = real_inputs['truth'][index]
        board = np.isfinite(truth)
        common = board & values['original'][index]['valid'] & values['gamma'][index]['valid']
        row = dict(id=image_id, common_board_pixels=int(common.sum()))
        for name, items in values.items():
            error = np.abs(items[index]['depth'][common]-truth[common]).astype(np.float64)
            errors[name].append(error)
            row[name] = dict(image_coverage=float(items[index]['valid'].mean()),
                board_coverage=float((board & items[index]['valid']).sum()/max(1, board.sum())),
                common_plane_mae_m=float(error.mean()) if error.size else None)
        rows.append(row)
    aggregate = {}
    for name, chunks in errors.items():
        error = np.concatenate(chunks)
        aggregate[name] = dict(mean_image_coverage=float(np.mean([row[name]['image_coverage'] for row in rows])),
            mean_board_coverage=float(np.mean([row[name]['board_coverage'] for row in rows])),
            common_board_pixels=int(error.size), common_plane_mae_m=float(error.mean()) if error.size else None,
            common_plane_p90_m=float(np.percentile(error, 90)) if error.size else None)
    study['real_camera_sanity'] = dict(aggregate=aggregate, views=rows, input_sha256=real_hashes,
        calibration_report_sha256=stereo.sha256(calibration_report_path), detections_sha256=stereo.sha256(detections_path),
        interpretation='Reused same-calibration PnP plane consistency only; no truth outside the board, so no out-of-range false prediction score.')
    np.savez_compressed(out/'lightstereo.npz', **output)
    audit._write_png(out/'lightstereo.png', np.vstack(panels+[_distance_legend(panels[0].shape[1], (.3, 5))]))
    study.update(status='COMPLETED', finished_at_utc=datetime.now(timezone.utc).isoformat(),
        output_sha256=stereo.sha256(out/'lightstereo.npz'), comparison_sha256=stereo.sha256(out/'lightstereo.png'),
        code_sha256={path.name: stereo.sha256(path) for path in (Path(__file__), Path(stereo.__file__),
                       stereo.EXPERIMENT/'evaluate.py', stereo.EXPERIMENT/'evaluate_portrait.py')})
    for factor in report.get('study', {}).get('low_coverage_factors', []):
        if factor['factor'] in ('低纹理的大面积平面', '输入亮度与对比度分布影响模型',
                                '左右视差不一致导致筛除', '量程限制和双目视野边界'):
            factor.setdefault('evidence', {})['standalone_lightstereo'] = 'lightstereo_study: fixed raw/Gamma, paired texture and rejection_counts; gamma_checks explicitly retain negative results'
    report['current_experiment'] = 'lightstereo_study'
    stereo.write_json(out/'acceptance.json', report)
    for section in ('original_scene', 'appearance'):
        print(section+' '+json.dumps({label: {name: {key: value[key] for key in
            ('mean_truth_coverage', 'accurate_0_1m_truth_coverage', 'pooled_mae_m', 'pooled_p95_error_m',
             'mean_black_image_fraction', 'false_in_range_pixels')} for name, value in result.items()}
             for label, result in study[section].items()}), flush=True)
    print('real_camera_sanity '+json.dumps(aggregate), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen', action='store_true', help='Only screen the 15 development views')
    parser.add_argument('--candidate', help='Evaluate a development-selected candidate; never tune on new validation')
    parser.add_argument('--capture', action='store_true', help='Capture the five fixed validation poses before comparison')
    parser.add_argument('--distribution', help='Override WSL distribution')
    parser.add_argument('--cache', type=Path, help='Override model cache in .local.json')
    parser.add_argument('--appearance', action='store_true', help='Evaluate the fixed-algorithm albedo control experiment')
    parser.add_argument('--lightstereo', action='store_true', help='Compare standalone raw/Gamma LightStereo on saved original and appearance inputs')
    args = parser.parse_args()
    if args.lightstereo and (args.capture or args.appearance or args.screen or args.candidate):
        parser.error('--lightstereo uses saved fixed inputs and cannot be combined with capture or other experiments')
    warnings.filterwarnings('ignore', message='.*torch.cuda.amp.autocast.*', category=FutureWarning)
    warnings.filterwarnings('ignore', message='torch.meshgrid:.*', category=UserWarning)
    warnings.filterwarnings('ignore', message='xFormers is .*', category=UserWarning)
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    sys.path.insert(0, str(stereo.EXPERIMENT))
    from portrait import CroppedMetric
    metric = CroppedMetric(stereo.CALIBRATION)
    settings = json.loads((stereo.ROOT/'.local.json').read_text(encoding='utf-8-sig'))
    cache = args.cache or Path(settings['stereo_cache'])
    out = stereo.ROOT/'results/lightstereo'
    if args.lightstereo:
        lightstereo_experiment(metric, out)
        return
    if args.capture:
        distribution = args.distribution or settings.get('distribution', 'Ubuntu-24.04')
        linux_root = subprocess.check_output(['wsl.exe', '-d', distribution, '--exec', 'wslpath', '-a',
            str(stereo.ROOT).replace('\\', '/')], text=True, encoding='utf-8').strip()
        task = 'lightstereo-appearance-capture' if args.appearance else 'lightstereo-refine-capture'
        subprocess.run(['wsl.exe', '-d', distribution, '--exec', 'bash', linux_root+'/scripts/launch.sh', task], check=True)
    if args.appearance:
        appearance_experiment(metric, cache, out)
        return
    report = json.loads((out/'acceptance.json').read_text(encoding='utf-8'))
    study = report['refinement_study']
    if stereo.sha256(out/'frames.npz') != study['original_frames_sha256']:
        raise ValueError('The preceding experiment changed; rerun with --capture')
    with np.load(out/'refinement.npz', allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    if len(arrays['left']) != 20:
        raise ValueError('Capture five new refinement views first')
    raw_hash = stereo.raw_capture_sha256(arrays)
    if study.get('raw_capture_sha256', raw_hash) != raw_hash:
        raise ValueError('Refinement inputs changed')
    study['raw_capture_sha256'] = raw_hash
    study['calibration_sha256'] = stereo.sha256(stereo.CALIBRATION)
    development, validation = list(range(15)), list(range(15, 20))
    metadata = study.get('metadata', {})
    source = predictions(arrays, SOURCES, development, metric, cache, metadata)
    previous = previous_predictions(arrays, development, source['raft1'], metric)
    reference = stereo.score(previous, arrays['truth'], development)
    trials, configs, outputs = {}, specs(), {}
    for name, spec in configs.items():
        items = candidate(spec, source, previous, development, metric)
        trials[name] = stereo.score(items, arrays['truth'], development, previous)
        trials[name]['checks'] = checks(reference, trials[name])
        outputs[name] = items
    study.update(status='DEVELOPMENT_SCREENED', metadata=metadata, development_reference=reference,
        development_trials=trials, selection_rule='All development checks must pass, then maximize <=0.1m truth coverage; freeze before views16-20.',
        interpretation='All previous 15 views are development data now. Bilinear enlargement adds no new camera detail. No truth is used for prediction or validity.')
    np.savez_compressed(out/'refinement.npz', **arrays)
    stereo.write_json(out/'acceptance.json', report)
    print('REFERENCE '+json.dumps(compact(reference)), flush=True)
    for name in sorted(trials, key=lambda n: trials[n]['accurate_0_1m_truth_coverage'], reverse=True)[:12]:
        print(json.dumps(dict(candidate=name, **compact(trials[name]), checks=trials[name]['checks'])), flush=True)
    if args.screen:
        return
    eligible = [name for name in trials if all(trials[name]['checks'].values())]
    if args.candidate:
        selected = args.candidate
        if selected not in eligible:
            raise ValueError('Requested candidate did not pass all development checks')
    elif eligible:
        selected = max(eligible, key=lambda name: trials[name]['accurate_0_1m_truth_coverage'])
    else:
        study['status'] = 'NO_JOINT_DEVELOPMENT_IMPROVEMENT'
        stereo.write_json(out/'acceptance.json', report)
        return
    spec = configs[selected]
    study.update(selected=selected, selected_config=spec, status='SELECTION_FROZEN',
                 frozen_at_utc=datetime.now(timezone.utc).isoformat())
    stereo.write_json(out/'acceptance.json', report)
    print('FROZEN '+selected, flush=True)
    names = {'raft1', spec['source']} if spec['kind'] == 'single' else {'raft1', spec['a'], spec['b']}
    fresh_sources = predictions(arrays, sorted(names), validation, metric, cache, metadata)
    fresh_previous = previous_predictions(arrays, validation, fresh_sources['raft1'], metric)
    fresh = candidate(spec, fresh_sources, fresh_previous, validation, metric)
    before = stereo.score(fresh_previous, arrays['truth'], validation)
    after = stereo.score(fresh, arrays['truth'], validation, fresh_previous)
    validation_checks = checks(before, after)
    real_check = real_photo_check(spec, metric, cache)
    chosen = outputs[selected].copy()
    chosen.update(fresh)
    previous.update(fresh_previous)
    for name, values in [('previous', previous), ('refined', chosen)]:
        for key in ('d', 'depth', 'valid'):
            arrays[name+'_'+key] = np.stack([values[index][key] for index in range(20)])
    arrays['refined_source'] = np.stack([chosen[index].get('source', chosen[index]['valid'].astype(np.uint8)) for index in range(20)])
    study.update(status='IMPROVED_ON_NEW_VIEWS' if all(validation_checks.values()) else 'MIXED_NEW_VIEW_RESULTS',
        promoted=all(validation_checks.values()),
        new_validation=dict(previous=before, refined=after, checks=validation_checks),
        real_camera_sanity=real_check,
        finished_at_utc=datetime.now(timezone.utc).isoformat(), metadata=metadata,
        real_distance_limit='This stage uses ideal Gazebo cameras; independently measured real distances and RK3588 performance remain unverified.')
    np.savez_compressed(out/'refinement.npz', **arrays)
    draw_comparison(arrays, previous, chosen, out/'refinement.png')
    study['output_sha256'] = stereo.sha256(out/'refinement.npz')
    study['comparison_sha256'] = stereo.sha256(out/'refinement.png')
    report['current_experiment'] = 'refinement_study'
    stereo.write_json(out/'acceptance.json', report)
    print(json.dumps(dict(selected=selected, status=study['status'], previous=compact(before), refined=compact(after), checks=validation_checks), indent=2), flush=True)


if __name__ == '__main__':
    main()
