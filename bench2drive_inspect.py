"""Inspect a converted Bench2Drive scene and save a map/trajectory plot on server."""

import argparse
from collections import Counter
import pickle
from pathlib import Path

import numpy as np

from bench2drive_preprocess import read_frame


def check_sample(sample):
    tracks = sample['track_infos']['trajs']
    points = sample['map_infos']['all_polylines']
    if tracks.ndim != 3 or tracks.shape[1:] != (91, 10):
        raise ValueError(f'Expected trajectories [N,91,10], got {tracks.shape}')
    valid = tracks[..., 9] > 0
    if not np.isfinite(tracks[valid]).all() or not np.isfinite(points).all():
        raise ValueError('Non-finite trajectory or map values')
    ego = sample['sdc_track_index']
    if not valid[ego, 10]:
        raise ValueError('Ego invalid at current frame')
    if points.ndim != 2 or points.shape[1] != 7 or len(points) == 0:
        raise ValueError(f'Invalid map shape: {points.shape}')
    print(f"Scenario: {sample['scenario_id']}")
    print(f'Trajectories: {tracks.shape}; map points: {len(points)}')
    print(f'Agents valid now: {valid[:, 10].sum()}; complete future tracks: {valid[:, 10:].all(axis=1).sum()}')
    print(f"Agent types: {dict(Counter(sample['track_infos']['object_type']))}")
    print(f'Ego length/width/height (m): {tracks[ego, 10, 3:6].round(3).tolist()}')
    nearest = np.linalg.norm(points[:, :2] - tracks[ego, 10, :2], axis=1).min()
    print(f'Nearest map point to ego: {nearest:.2f} m')
    return tracks, points, ego


def check_raw_ego(sample, raw_root):
    clip_name, start = sample['scenario_id'].rsplit('__', 1)
    root = Path(raw_root)
    if (root / 'v1').is_dir():
        root = root / 'v1'
    frame = read_frame(root / clip_name / 'anno' / f'{int(start) + 10:05d}.json.gz')
    actor = next(box for box in frame['bounding_boxes'] if box['class'] == 'ego_vehicle')
    expected = 2 * np.asarray(actor['extent'][:3], dtype=np.float32)
    actual = sample['track_infos']['trajs'][sample['sdc_track_index'], 10, 3:6]
    np.testing.assert_allclose(actual, expected, atol=1e-5,
                               err_msg='Box dimensions differ from raw annotations; reconvert with latest code')
    print('Raw ego box dimensions: matched')


def plot_sample(sample, output, radius):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    tracks, points, ego = check_sample(sample)
    origin = tracks[ego, 10, :2]
    fig, ax = plt.subplots(figsize=(10, 10))
    local_map = points[:, :2] - origin
    visible = np.linalg.norm(local_map, axis=1) <= radius * 1.5
    ax.scatter(local_map[visible, 0], local_map[visible, 1], s=2,
               c='#b5bcc4', linewidths=0, label='HD map')
    for index, trajectory in enumerate(tracks):
        if not trajectory[10, 9]:
            continue
        position = trajectory[:, :2] - origin
        if np.linalg.norm(position[10]) > radius:
            continue
        color = '#d62728' if index == ego else '#2878b5'
        # NaNs prevent lines joining across missing observations.
        position = position.copy()
        position[trajectory[:, 9] == 0] = np.nan
        label = 'Ego future' if index == ego else None
        ax.plot(position[10:, 0], position[10:, 1], color=color, lw=1.5, label=label)
        ax.plot(position[:11, 0], position[:11, 1], color=color, lw=1, linestyle='--')
        length, width = trajectory[10, 3:5]
        heading = trajectory[10, 6]
        corners = np.array([[length, width], [length, -width],
                            [-length, -width], [-length, width]]) / 2
        rotation = np.array([[np.cos(heading), -np.sin(heading)],
                             [np.sin(heading), np.cos(heading)]])
        ax.add_patch(Polygon(corners @ rotation.T + position[10],
                             closed=True, fill=False, edgecolor=color, lw=1.5))
        ax.arrow(*position[10], 3 * np.cos(heading), 3 * np.sin(heading),
                 color=color, head_width=0.7, length_includes_head=True)
    ax.set(xlim=(-radius, radius), ylim=(-radius, radius),
           xlabel='World X relative to ego (m)', ylabel='World Y relative to ego (m)',
           title=f"{sample['scenario_id']}\nDashed: history; solid: future; boxes: current frame")
    ax.set_aspect('equal')
    ax.grid(alpha=0.15)
    ax.legend(loc='upper right')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f'Inspection image: {output.resolve()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='Converted scenario root')
    parser.add_argument('--split', choices=['training', 'validation'], default='training')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--raw-root', help='Check converted ego dimensions against original annotations')
    parser.add_argument('--output', required=True, help='PNG output on server')
    parser.add_argument('--radius', type=float, default=60)
    args = parser.parse_args()
    root = Path(args.root)
    with open(root / f'processed_scenarios_{args.split}_infos.pkl', 'rb') as stream:
        infos = pickle.load(stream)
    if not infos:
        raise ValueError(f'{args.split} split is empty; choose a populated split')
    with open(root / f'processed_scenarios_{args.split}' /
              f"sample_{infos[args.index]['scenario_id']}.pkl", 'rb') as stream:
        sample = pickle.load(stream)
    if args.raw_root:
        check_raw_ego(sample, args.raw_root)
    plot_sample(sample, args.output, args.radius)


if __name__ == '__main__':
    main()
