"""Evaluate learned autoregressive UniMM trajectories on Bench2Drive validation.

These masked displacement errors are diagnostics, not WOSAC or CARLA driving scores.
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time

import numpy as np


TYPE_NAMES = ['Unset', 'Vehicle', 'Pedestrian', 'Cyclist', 'Other']


def displacement_errors(prediction, target, valid):
    """Return per-agent sampled mean/oracle errors; FDE uses the true horizon end."""
    if prediction.ndim != 4 or prediction.shape[1:] != target.shape:
        raise ValueError('Predictions must be [rollouts, agents, steps, 2], matching target')
    if valid.shape != target.shape[:2]:
        raise ValueError('Validity shape does not match target')
    if not np.isfinite(prediction).all() or not np.isfinite(target[valid]).all():
        raise ValueError('Non-finite rollout or valid target values')
    distance = np.linalg.norm(prediction - target[None], axis=-1)
    distance = np.where(valid[None], distance, 0.0)
    counts = valid.sum(axis=1)
    eligible = counts > 0
    ade = distance.sum(axis=-1) / np.maximum(counts, 1)[None]
    mean_ade = np.where(eligible, ade.mean(axis=0), np.nan)
    min_ade = np.where(eligible, ade.min(axis=0), np.nan)
    fde = distance[..., -1]
    mean_fde = np.where(valid[:, -1], fde.mean(axis=0), np.nan)
    min_fde = np.where(valid[:, -1], fde.min(axis=0), np.nan)
    return {'meanADE_m': mean_ade, 'minADE_m': min_ade,
            'meanFDE_m': mean_fde, 'minFDE_m': min_fde}


def summarize(values):
    array = np.asarray(values, dtype=float)
    usable = array[np.isfinite(array)]
    return {'value': float(usable.mean()) if len(usable) else None, 'agents': int(len(usable))}


def choose_windows(infos, count, seed):
    """Spread the subset over validation clips before selecting second windows."""
    if count == 0 or count >= len(infos):
        return list(range(len(infos)))
    groups = defaultdict(list)
    for i, info in enumerate(infos):
        groups[info['scenario_id'].rsplit('__', 1)[0]].append(i)
    rng = np.random.default_rng(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(groups[key])
    chosen = []
    depth = 0
    while len(chosen) < count:
        for key in keys:
            if depth < len(groups[key]):
                chosen.append(groups[key][depth])
                if len(chosen) == count:
                    return chosen
        depth += 1
    return chosen


def select_checkpoint(run_dir, explicit):
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    import torch
    candidates = list(Path(run_dir).rglob('last.ckpt'))
    if not candidates:
        raise FileNotFoundError(f'No last.ckpt under {run_dir}; provide --ckpt')
    last = max(candidates, key=lambda path: path.stat().st_mtime)
    checkpoint = torch.load(last, map_location='cpu')
    for state in checkpoint.get('callbacks', {}).values():
        if not isinstance(state, dict) or not state.get('best_model_path'):
            continue
        best = Path(state['best_model_path'])
        if not best.is_file():
            best = last.parent / best.name
        if best.is_file():
            print(f'Using checkpoint selected by validation classification loss: {best}', flush=True)
            return best
    print(f'No saved best checkpoint path found; using {last}', flush=True)
    return last


def plot_rollout(data, prediction, target, valid, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    positions = data['agent']['position'].cpu().numpy()
    origin = positions[0, 10, :2]
    points = data['map_point']['position'].cpu().numpy()[:, :2] - origin
    types = data['agent']['type'].cpu().numpy()
    colors = ['#666666', '#2878b5', '#2c9c58', '#a13fba', '#666666']
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for ax, title in zip(axes, ['Recorded future (valid frames)', 'Model rollout 1 (no future input)']):
        ax.scatter(points[:, 0], points[:, 1], c='#c6cbd0', s=2, linewidths=0)
        ax.set(xlim=(-80, 80), ylim=(-80, 80), title=title,
               xlabel='X relative to current ego (m)', ylabel='Y relative to current ego (m)')
        ax.set_aspect('equal'); ax.grid(alpha=0.15)
    for i, category in enumerate(types):
        color = '#d62728' if i == 0 else colors[int(category)]
        gt = target[i].copy() - origin
        gt[~valid[i]] = np.nan
        axes[0].plot(gt[:, 0], gt[:, 1], color=color, linewidth=1.2)
        pred = prediction[0, i] - origin
        axes[1].plot(pred[:, 0], pred[:, 1], color=color, linewidth=1.2)
        for ax in axes:
            current = positions[i, 10, :2] - origin
            ax.scatter(*current, c=color, s=14)
    fig.suptitle(f"{data['scenario_id']} | red: ego | blue: vehicle | green: pedestrian | purple: cyclist", fontsize=10)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main(args):
    import torch
    from torch_geometric.data import Batch
    from datasets import WaymoDataset
    from models import UniMM

    if args.rollouts <= 0 or args.max_scenes < 0 or args.plot_scenes < 0:
        raise ValueError('Invalid rollout/scene counts')
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; activate the unimm environment')
    checkpoint = select_checkpoint(args.run_dir, args.ckpt)
    overrides = {'anchor_file': args.anchor_file} if args.anchor_file else {}
    model = UniMM.load_from_checkpoint(str(checkpoint), map_location='cpu', **overrides)
    model.eval().to(args.device)
    if model.num_historical_steps != 11:
        raise ValueError('This Bench2Drive adapter expects 11 history/current steps')
    dataset = WaymoDataset(data_root=args.root,
                           split_dir='processed_scenarios_validation',
                           info_file='processed_scenarios_validation_infos.pkl')
    if not len(dataset):
        raise ValueError('Validation split is empty')
    chosen = choose_windows(dataset.infos, args.max_scenes, args.seed)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    metrics = defaultdict(list)
    type_metrics = {name: defaultdict(list) for name in TYPE_NAMES}
    records = []
    started = time.monotonic()
    for number, index in enumerate(chosen, 1):
        data = dataset[index]
        target = data['agent']['position'][:, 11:91, :2].numpy().copy()
        valid = data['agent']['valid_mask'][:, 11:91].numpy().copy()
        types = data['agent']['type'].numpy().copy()
        origin = data['agent']['position'][:, 10, :2].numpy()
        velocity = data['agent']['velocity'][:, 10, :2].numpy()
        constant_velocity = origin[:, None] + velocity[:, None] * (np.arange(1, 81) * 0.1)[None, :, None]
        baseline = displacement_errors(constant_velocity[None], target, valid)
        # Cached agent inference requires per-node batch indices even for one scene.
        device_data = Batch.from_data_list([data.clone()]).to(args.device)
        predictions = []
        with torch.inference_mode():
            for _ in range(args.rollouts):
                result = model.inference(device_data, sim_steps=math.ceil(80 / model.num_execution_steps))
                predictions.append(result['sim_positions'][:, :80, :2].cpu().numpy())
        prediction = np.stack(predictions)
        errors = displacement_errors(prediction, target, valid)
        errors['CV_ADE_m'] = baseline['meanADE_m']
        errors['CV_FDE_m'] = baseline['meanFDE_m']
        for key, values in errors.items():
            metrics[key].extend(values.tolist())
            for category, name in enumerate(TYPE_NAMES):
                type_metrics[name][key].extend(values[types == category].tolist())
        scene_summary = {key: summarize(values) for key, values in errors.items()}
        records.append({'scenario_id': data['scenario_id'], 'metrics': scene_summary})
        if number <= args.plot_scenes:
            plot_rollout(data, prediction, target, valid, output / f'rollout_{number:02d}.png')
        print(f'[{time.monotonic() - started:.1f}s] {number}/{len(chosen)} {data["scenario_id"]}: '
              f'meanADE {scene_summary["meanADE_m"]["value"]}, '
              f'minADE {scene_summary["minADE_m"]["value"]}', flush=True)
        # Save progress so an interruption does not discard completed evaluation windows.
        summary = {
            'checkpoint': str(checkpoint), 'seed': args.seed, 'rollouts': args.rollouts,
            'horizon_seconds': 8, 'evaluated_windows': len(records),
            'unique_clips': len({record['scenario_id'].rsplit('__', 1)[0] for record in records}),
            'metrics': {key: summarize(values) for key, values in metrics.items()},
            'by_type': {name: {key: summarize(values) for key, values in group.items()}
                        for name, group in type_metrics.items()},
            'notes': ['ADE averages valid future frames per agent, then agents across windows.',
                      'FDE includes only agents valid at the 8-second endpoint.',
                      'minADE/minFDE are oracle minima over sampled rollouts for each agent.',
                      'CV is a constant-velocity baseline from current position and velocity.',
                      'Overlapping windows are correlated; these are diagnostic errors, not WOSAC/CARLA scores.'],
            'scenarios': records,
        }
        (output / 'metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(summary['metrics'], indent=2), flush=True)
    print(f'Results: {output.resolve()}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run-dir', help='Training run directory; select its recorded best checkpoint')
    group.add_argument('--ckpt', help='Explicit checkpoint path')
    parser.add_argument('--anchor-file', help='Override anchor path if checkpoint paths moved')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-scenes', type=int, default=32, help='Validation windows; 0 for all')
    parser.add_argument('--rollouts', type=int, default=8)
    parser.add_argument('--plot-scenes', type=int, default=3)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--seed', type=int, default=2025)
    main(parser.parse_args())
