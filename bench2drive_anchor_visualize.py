"""Plot UniMM anchors on the server without downloading data or opening a GUI."""

import argparse
import json
from pathlib import Path

import numpy as np


TYPE_NAMES = ['Unset', 'Vehicle', 'Pedestrian', 'Cyclist', 'Other']


def describe_anchors(anchors):
    if anchors.ndim != 4 or anchors.shape[0] != 5 or anchors.shape[-1] != 3:
        raise ValueError(f'Expected [5, modes, steps, 3 (x,y,heading)], got {anchors.shape}')
    if not anchors.shape[1] or not anchors.shape[2] or not np.isfinite(anchors).all():
        raise ValueError('Anchors must contain nonempty finite trajectories')
    stats = []
    unique_sets = []
    for index, name in enumerate(TYPE_NAMES):
        unique, counts = np.unique(anchors[index].reshape(anchors.shape[1], -1),
                                   axis=0, return_counts=True)
        unique = unique.reshape(-1, anchors.shape[2], 3)
        matches = [other for j, other in enumerate(TYPE_NAMES)
                   if j != index and np.array_equal(anchors[index], anchors[j])]
        displacement = np.linalg.norm(anchors[index, :, -1, :2], axis=-1)
        stats.append({
            'type': name,
            'modes': int(anchors.shape[1]),
            'unique_full_trajectories': int(len(unique)),
            'duplicate_modes': int(anchors.shape[1] - len(unique)),
            'largest_repeat_count': int(counts.max()),
            'identical_type_arrays': matches,
            'endpoint_displacement_m': {
                'min': float(displacement.min()),
                'median': float(np.median(displacement)),
                'max': float(displacement.max()),
            },
        })
        unique_sets.append((unique, counts))
    return stats, unique_sets


def render(anchors, output, max_lines=256, horizon_seconds=8.0, sample_hz=10.0, seed=2025):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    if max_lines <= 0 or horizon_seconds <= 0 or sample_hz <= 0:
        raise ValueError('max-lines, horizon-seconds and sample-hz must be positive')
    steps = min(anchors.shape[2], max(1, int(round(horizon_seconds * sample_hz))))
    stats, unique_sets = describe_anchors(anchors)
    fig, axes = plt.subplots(3, 2, figsize=(13, 15), constrained_layout=True)
    fig.suptitle(f'UniMM Anchor Trajectories on Bench2Drive\n'
                 f'{anchors.shape[1]} modes per type | first {steps / sample_hz:g} s at {sample_hz:g} Hz',
                 fontsize=17, fontweight='bold')
    normalizer = Normalize(-180, 180)
    color_map = plt.get_cmap('twilight')
    for row, type_index in enumerate([1, 2, 3]):
        unique, repeats = unique_sets[type_index]
        count = min(max_lines, len(unique))
        # Identical fallback arrays must show the same subset in each row.
        chosen = np.sort(np.random.default_rng(seed).choice(len(unique), count, replace=False))
        paths = unique[chosen, :steps, :2]
        # The current position is the origin; anchor point 0 is the next frame.
        paths = np.concatenate((np.zeros((count, 1, 2)), paths), axis=1)
        headings = (np.rad2deg(unique[:, steps - 1, 2]) + 180) % 360 - 180
        trajectory_ax, endpoint_ax = axes[row]
        collection = LineCollection(paths, colors=color_map(normalizer(headings[chosen])),
                                    linewidths=0.8, alpha=0.5)
        trajectory_ax.add_collection(collection)
        trajectory_ax.autoscale_view()
        trajectory_ax.set_title(f'{TYPE_NAMES[type_index]} paths | showing {count} unique anchors', fontsize=12)
        endpoints = unique[:, steps - 1, :2]
        points = endpoint_ax.scatter(endpoints[:, 0], endpoints[:, 1], c=headings,
                                      cmap=color_map, norm=normalizer,
                                      s=10 + 5 * np.log2(repeats), alpha=0.7, linewidths=0)
        endpoint_ax.set_title(f'Endpoints | {len(unique)} unique full trajectories / {anchors.shape[1]} modes',
                              fontsize=11)
        colorbar = fig.colorbar(points, ax=endpoint_ax, fraction=0.04, pad=0.02)
        colorbar.set_label('Final relative heading (degrees)', fontsize=9)
        # Both panels use the same ranges, including all unique anchors.
        all_xy = np.concatenate((np.zeros((1, 2)), unique[:, :steps, :2].reshape(-1, 2)))
        low, high = all_xy.min(axis=0), all_xy.max(axis=0)
        padding = np.maximum((high - low) * 0.05, 1.0)
        for ax in (trajectory_ax, endpoint_ax):
            ax.scatter([0], [0], marker='x', s=40, c='black', zorder=5)
            ax.axhline(0, c='#b0b0b0', lw=0.6)
            ax.axvline(0, c='#b0b0b0', lw=0.6)
            ax.set_xlim(low[0] - padding[0], high[0] + padding[0])
            ax.set_ylim(low[1] - padding[1], high[1] + padding[1])
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel('Local X forward (m)')
            ax.set_ylabel('Local Y lateral (m)')
            ax.grid(alpha=0.18)
        identical = stats[type_index]['identical_type_arrays']
        if identical:
            trajectory_ax.text(0.02, 0.02, 'Identical array: ' + ', '.join(identical),
                               transform=trajectory_ax.transAxes, fontsize=9)
    fig.supxlabel('Black cross: current pose | colors: end heading | endpoint size: duplicate multiplicity',
                  fontsize=10)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anchor-file', required=True)
    parser.add_argument('--output', required=True, help='PNG path on server')
    parser.add_argument('--max-lines', type=int, default=256, help='Maximum unique paths drawn per type')
    parser.add_argument('--horizon-seconds', type=float, default=8.0, help='Use 4 to inspect the training prefix')
    parser.add_argument('--sample-hz', type=float, default=10.0)
    parser.add_argument('--seed', type=int, default=2025)
    args = parser.parse_args()
    anchors = np.load(args.anchor_file, allow_pickle=False)
    stats = render(anchors, args.output, args.max_lines, args.horizon_seconds, args.sample_hz, args.seed)
    report = Path(args.output).with_suffix('.json')
    report.write_text(json.dumps({'shape': list(anchors.shape), 'types': stats}, indent=2), encoding='utf-8')
    for item in stats:
        print(f"{item['type']}: {item['modes']} modes, {item['unique_full_trajectories']} unique, "
              f"{item['duplicate_modes']} duplicates; identical arrays: {item['identical_type_arrays']}",
              flush=True)
    print(f'Image: {Path(args.output).resolve()}', flush=True)
    print(f'Statistics: {report.resolve()}', flush=True)


if __name__ == '__main__':
    main()
