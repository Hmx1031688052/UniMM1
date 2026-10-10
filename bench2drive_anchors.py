"""Fit UniMM anchor trajectories from converted Bench2Drive training scenes."""

import argparse
import pickle
from pathlib import Path

import numpy as np


TYPES = ['TYPE_UNSET', 'TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST', 'TYPE_OTHER']


def collect(root, max_per_type, seed):
    rng = np.random.default_rng(seed)
    buckets = [[] for _ in TYPES]
    seen = np.zeros(len(TYPES), dtype=np.int64)
    with open(root / 'processed_scenarios_training_infos.pkl', 'rb') as stream:
        infos = pickle.load(stream)
    for info in infos:
        path = root / 'processed_scenarios_training' / f"sample_{info['scenario_id']}.pkl"
        with open(path, 'rb') as stream:
            sample = pickle.load(stream)
        tracks = sample['track_infos']
        trajectories = tracks['trajs']
        for actor_type, trajectory in zip(tracks['object_type'], trajectories):
            if actor_type not in TYPES or not trajectory[10:91, 9].all():
                continue
            index = TYPES.index(actor_type)
            origin = trajectory[10, :2]
            heading = trajectory[10, 6]
            rotation = np.array([[np.cos(heading), -np.sin(heading)],
                                 [np.sin(heading), np.cos(heading)]], dtype=np.float32)
            local = np.empty((80, 3), dtype=np.float32)
            local[:, :2] = (trajectory[11:91, :2] - origin) @ rotation
            local[:, 2] = (trajectory[11:91, 6] - heading + np.pi) % (2 * np.pi) - np.pi
            seen[index] += 1
            if len(buckets[index]) < max_per_type:
                buckets[index].append(local)
            else:
                slot = rng.integers(seen[index])
                if slot < max_per_type:
                    buckets[index][slot] = local
    return buckets


def main(args):
    buckets = collect(Path(args.root), args.max_per_type, args.seed)
    anchors = np.zeros((len(TYPES), args.num_modes, 80, 3), dtype=np.float32)
    for index in [1, 2, 3, 4, 0]:
        source = np.asarray(buckets[index], dtype=np.float32)
        if not len(source):
            fallback = 2 if index == 3 and len(buckets[2]) else 1
            anchors[index] = anchors[fallback]
            print(f'{TYPES[index]}: no complete trajectories, using {TYPES[fallback]} anchors')
            continue
        if len(source) <= args.num_modes:
            choices = np.arange(args.num_modes) % len(source)
            anchors[index] = source[choices]
            print(f'{TYPES[index]}: {len(source)} trajectories, repeated to {args.num_modes} modes')
            continue
        from sklearn.cluster import MiniBatchKMeans
        model = MiniBatchKMeans(n_clusters=args.num_modes, random_state=args.seed,
                                batch_size=max(1024, 3 * args.num_modes), n_init=3)
        model.fit(source.reshape(len(source), -1))
        anchors[index] = model.cluster_centers_.reshape(args.num_modes, 80, 3)
        print(f'{TYPES[index]}: clustered {len(source)} trajectories')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, anchors)
    print(f'Saved {anchors.shape} to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='Converted scenario root on server')
    parser.add_argument('--output', required=True, help='Anchor .npy output on server')
    parser.add_argument('--num-modes', type=int, default=2048)
    parser.add_argument('--max-per-type', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=2025)
    main(parser.parse_args())
