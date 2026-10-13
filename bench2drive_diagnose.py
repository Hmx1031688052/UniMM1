"""Diagnose rollout drift, static targets, anchor coverage and data-generation mismatch.

Uses validation data and existing checkpoints; never trains or modifies data.
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time

import numpy as np

from bench2drive_evaluate import choose_windows, displacement_errors, plot_rollout, select_checkpoint, summarize
from bench2drive_preprocess import read_frame


HORIZONS = [0.5, 1, 2, 4, 8]


def motion_groups(target, origin, velocity, valid):
    distance = np.linalg.norm(target - origin[:, None], axis=-1)
    maximum = np.where(valid, distance, 0).max(axis=1)
    stationary = (maximum < 0.5) & (valid.sum(axis=1) >= 40)
    return {'all': np.ones(len(target), dtype=bool),
            'stationary_gt': stationary,
            'moving_gt': maximum >= 0.5,
            'ego': np.arange(len(target)) == 0}


def anchor_coverage(anchors, types, target, origin, heading, valid):
    coverage = {str(horizon): [] for horizon in [0.5, 4, 8]}
    for category, trajectory, start, angle, mask in zip(types, target, origin, heading, valid):
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        local = (trajectory - start) @ rotation
        for horizon in [0.5, 4, 8]:
            steps = int(horizon * 10)
            count = mask[:steps].sum()
            if not count:
                coverage[str(horizon)].append(np.nan)
                continue
            difference = np.linalg.norm(anchors[int(category), :, :steps, :2] - local[None, :steps], axis=-1)
            error = np.where(mask[None, :steps], difference, 0).sum(axis=-1) / count
            coverage[str(horizon)].append(float(error.min()))
    return coverage


def current_raw(data, raw_root):
    clip, start = data['scenario_id'].rsplit('__', 1)
    root = Path(raw_root)
    if (root / 'v1').is_dir():
        root = root / 'v1'
    frame = read_frame(root / clip / 'anno' / f'{int(start) + 10:05d}.json.gz')
    actors = {str(actor['id']): actor for actor in frame['bounding_boxes'] if 'id' in actor}
    selected = [actors.get(str(actor_id), {}) for actor_id in data['agent']['id']]
    return selected, sum(actor.get('class') == 'traffic_light' for actor in actors.values())


def main(args):
    import torch
    from torch_geometric.data import Batch
    from scipy.spatial import cKDTree
    from datasets import WaymoDataset
    from models import UniMM

    if args.max_scenes < 0 or args.rollouts <= 0:
        raise ValueError('Scene count must be nonnegative and rollout count positive')
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    checkpoint = select_checkpoint(args.run_dir, args.ckpt)
    model = UniMM.load_from_checkpoint(str(checkpoint), map_location='cpu').eval().to(device)
    if model.anchor_free or model.num_historical_steps != 11:
        raise ValueError('This diagnostic currently targets the Anchor-Based Bench2Drive adapter')
    anchors = model.anchor_trajs.detach().cpu().numpy()
    anchor_stats = {}
    for category, name in enumerate(['Unset', 'Vehicle', 'Pedestrian', 'Cyclist', 'Other']):
        unique = len(np.unique(anchors[category].reshape(len(anchors[category]), -1), axis=0))
        # Rounded prefixes expose modes that differ mainly in their distant futures.
        prefix = np.round(anchors[category, :, :5].reshape(len(anchors[category]), -1), 2)
        anchor_stats[name] = {'modes': len(prefix), 'unique_full': unique,
                              'unique_05s_prefix_rounded_001': len(np.unique(prefix, axis=0)),
                              'near_stationary_8s_modes': int((np.linalg.norm(anchors[category, :, :, :2], axis=-1).max(axis=1) < 0.5).sum())}
    dataset = WaymoDataset(data_root=args.root, split_dir='processed_scenarios_validation',
                           info_file='processed_scenarios_validation_infos.pkl')
    if not len(dataset):
        raise ValueError('Validation split is empty')
    chosen = choose_windows(dataset.infos, args.max_scenes, args.seed)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    totals = defaultdict(list)
    data_audit = defaultdict(list)
    coverage = defaultdict(list)
    candidates = accepted = 0
    scenes = []
    started = time.monotonic()
    worst_error = -1
    for number, index in enumerate(chosen, 1):
        data = dataset[index]
        positions = data['agent']['position'].numpy()
        velocity = data['agent']['velocity'][:, 10, :2].numpy()
        heading = data['agent']['heading'][:, 10].numpy()
        target = positions[:, 11:91, :2].copy()
        origin = positions[:, 10, :2].copy()
        valid = data['agent']['valid_mask'][:, 11:91].numpy().copy()
        groups = motion_groups(target, origin, velocity, valid)
        groups.update({name: data['agent']['type'].numpy() == category
                       for category, name in [(1, 'vehicle'), (2, 'pedestrian'), (3, 'cyclist')]})
        raw_actors, light_count = current_raw(data, args.raw_root)
        groups['raw_static'] = np.array([actor.get('state') == 'static' for actor in raw_actors])
        groups['raw_dynamic'] = np.array([actor.get('state') == 'dynamic' for actor in raw_actors])
        data_audit['raw_traffic_light_count'].append(light_count)
        future_distance = np.linalg.norm(target - positions[0, 10, :2], axis=-1)
        data_audit['gt_outside_initial_100m_map_fraction'].extend((future_distance[valid] > 100.1).astype(float).tolist())
        points = data['map_point']['position'].numpy()[:, :2]
        if valid.any() and len(points):
            nearest = cKDTree(points).query(target[valid])[0]
            data_audit['gt_farther_than_10m_from_retained_map_fraction'].extend((nearest > 10).astype(float).tolist())
        consecutive = data['agent']['valid_mask'][:, :-1].numpy() & data['agent']['valid_mask'][:, 1:].numpy()
        jumps = np.linalg.norm(np.diff(positions[:, :, :2], axis=1), axis=-1)
        data_audit['gt_step_jump_gt_15m_fraction'].extend((jumps[consecutive] > 15).astype(float).tolist())
        data_audit['complete_11_frame_history_fraction'].extend(data['agent']['valid_mask'][:, :11].numpy().all(axis=1).astype(float).tolist())
        delta = np.linalg.norm(velocity, axis=-1)
        for i, actor in enumerate(raw_actors):
            raw_speed = actor.get('speed')
            if actor.get('class') in ('vehicle', 'ego_vehicle') and actor.get('state') != 'static' and raw_speed is not None:
                data_audit['vehicle_derived_vs_raw_speed_abs_error_mps'].append(abs(float(delta[i]) - abs(float(raw_speed))))
        sample_coverage = anchor_coverage(anchors, data['agent']['type'].numpy(), target, origin, heading, valid)
        for horizon, errors in sample_coverage.items():
            coverage[horizon].extend(errors)
        batch = Batch.from_data_list([data.clone()]).to(device)
        with torch.inference_mode():
            matching = model.match_rollout(batch, sim_steps=model.train_sim_steps,
                                           open_loop=model.open_loop_train,
                                           use_model=not model.approx_posterior,
                                           match_execution=model.match_execution)
            candidates += int(matching['replacement_candidates'])
            accepted += int(matching['replacement_accepted'])
            del matching
            strategies = [('full_sampling', -1, args.rollouts), ('top1', 1, 1),
                          ('top6_sampling', min(6, model.num_modes), args.rollouts)]
            per_scene = {}
            for strategy, top_k, repeats in strategies:
                torch.manual_seed(args.seed + int(index))
                predictions = []
                for _ in range(repeats):
                    result = model.inference(batch, sim_steps=math.ceil(80 / model.num_execution_steps), top_k=top_k)
                    predictions.append(result['sim_positions'][:, :80, :2].cpu().numpy())
                prediction = np.stack(predictions)
                for horizon in HORIZONS:
                    steps = int(horizon * 10)
                    errors = displacement_errors(prediction[:, :, :steps], target[:, :steps], valid[:, :steps])
                    for group, mask in groups.items():
                        for metric, values in errors.items():
                            totals[f'{strategy}/{group}/{horizon}s/{metric}'].extend(values[mask].tolist())
                errors_8 = displacement_errors(prediction, target, valid)
                per_scene[strategy] = {key: summarize(values) for key, values in errors_8.items()}
                if strategy == 'full_sampling':
                    error = per_scene[strategy]['meanADE_m']['value']
                    if error is not None and error > worst_error:
                        worst_error = error
                        plot_rollout(data, prediction, target, valid, output / 'worst_sampling.png')
                    if number <= 3:
                        plot_rollout(data, prediction, target, valid, output / f'sampling_{number:02d}.png')
                elif strategy == 'top1' and number <= 3:
                    plot_rollout(data, prediction, target, valid, output / f'top1_{number:02d}.png')
        scenes.append({'scenario_id': data['scenario_id'], 'strategies': per_scene,
                       'stationary_gt_agents': int(groups['stationary_gt'].sum()),
                       'raw_static_agents': int(groups['raw_static'].sum())})
        summary = {'checkpoint': str(checkpoint), 'epoch': int(torch.load(checkpoint, map_location='cpu').get('epoch', -1)) if number == 1 else summary['epoch'],
                   'evaluated_windows': len(scenes), 'rollouts': args.rollouts, 'seed': args.seed,
                   'anchor_statistics': anchor_stats,
                   'training_style_rollout': {'candidate_agent_steps': candidates, 'accepted_agent_steps': accepted,
                                              'replacement_fraction': accepted / candidates if candidates else None,
                                              'note': 'Measured on validation windows with the saved training configuration; not a training-set statistic.'},
                   'data_audit': {key: summarize(values) for key, values in data_audit.items()},
                   'anchor_oracle_xy_ADE_m': {key: summarize(values) for key, values in coverage.items()},
                   'policy_group_horizon_metrics': {key: summarize(values) for key, values in totals.items()},
                   'scenarios': scenes,
                   'notes': ['Top-1 and top-6 diagnose sampling; this does not establish which policy is most realistic.',
                             'stationary_gt uses observed future only for analysis: max displacement <0.5m and >=40 valid frames.',
                             'Map-distance checks diagnose coverage, not drivable-area violations.',
                             'Anchor oracle uses XY distance, not the training bounding-box matching distance.']}
        (output / 'diagnostics.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
        print(f'[{time.monotonic()-started:.1f}s] {number}/{len(chosen)} {data["scenario_id"]}: '
              f'sample ADE={per_scene["full_sampling"]["meanADE_m"]["value"]}, '
              f'top1 ADE={per_scene["top1"]["meanADE_m"]["value"]}; '
              f'accepted replacement={accepted}/{candidates}', flush=True)
    compact = {'checkpoint': str(checkpoint), 'epoch': summary['epoch'],
               'training_style_rollout': summary['training_style_rollout'],
               'data_audit': summary['data_audit'], 'anchor_statistics': anchor_stats,
               'anchor_oracle_xy_ADE_m': summary['anchor_oracle_xy_ADE_m'],
               'comparisons': {key: value for key, value in summary['policy_group_horizon_metrics'].items()
                               if '/all/' in key or ('/8s/' in key and '/meanADE_m' in key)}}
    (output / 'diagnostics_summary.json').write_text(json.dumps(compact, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(compact, indent=2), flush=True)
    print(f'Results: {output.resolve()}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--raw-root', required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--run-dir')
    choice.add_argument('--ckpt')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-scenes', type=int, default=32)
    parser.add_argument('--rollouts', type=int, default=8)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    main(parser.parse_args())
