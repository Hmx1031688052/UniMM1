"""Correct bicycle types in existing processed scenarios without rebuilding maps.

Dry run by default. Use --apply on the server to update scenario pickles.
Regenerate anchors after applying corrections.
"""

import argparse
import os
import pickle
from pathlib import Path
import time

from bench2drive_preprocess import classify_actor, read_frame


def relabel(root, raw_root, apply=False):
    root, raw_root = Path(root), Path(raw_root)
    if (raw_root / 'v1').is_dir():
        raw_root = raw_root / 'v1'
    started = time.monotonic()
    changed_scenes = changed_tracks = 0
    for split in ('training', 'validation'):
        with open(root / f'processed_scenarios_{split}_infos.pkl', 'rb') as stream:
            infos = pickle.load(stream)
        for number, info in enumerate(infos, 1):
            scene_id = info['scenario_id']
            clip, start = scene_id.rsplit('__', 1)
            frame = read_frame(raw_root / clip / 'anno' / f'{int(start) + 10:05d}.json.gz')
            cyclists = {str(actor['id']) for actor in frame['bounding_boxes']
                        if 'id' in actor and classify_actor(actor) == 'TYPE_CYCLIST'}
            if cyclists:
                path = root / f'processed_scenarios_{split}' / f'sample_{scene_id}.pkl'
                with open(path, 'rb') as stream:
                    sample = pickle.load(stream)
                tracks = sample['track_infos']
                changes = 0
                for i, actor_id in enumerate(tracks['object_id']):
                    if str(actor_id) in cyclists and tracks['object_type'][i] != 'TYPE_CYCLIST':
                        tracks['object_type'][i] = 'TYPE_CYCLIST'
                        changes += 1
                if changes:
                    changed_scenes += 1
                    changed_tracks += changes
                    if apply:
                        temp = path.with_suffix('.pkl.relabel-tmp')
                        try:
                            with open(temp, 'wb') as stream:
                                pickle.dump(sample, stream, protocol=pickle.HIGHEST_PROTOCOL)
                            os.replace(temp, path)
                        finally:
                            if temp.exists():
                                temp.unlink()
            if number == 1 or number % 500 == 0 or number == len(infos):
                print(f'[{time.monotonic() - started:.1f}s] {split}: {number}/{len(infos)}; '
                      f'affected scenarios {changed_scenes}, track entries {changed_tracks}', flush=True)
    mode = 'Updated' if apply else 'Would update'
    print(f'{mode} {changed_scenes} scenarios and {changed_tracks} track entries', flush=True)
    if apply and changed_scenes:
        print('Regenerate anchors from corrected training data before training.', flush=True)
    return changed_scenes, changed_tracks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='Existing converted scenario root')
    parser.add_argument('--raw-root', required=True, help='Base root or v1 directory')
    parser.add_argument('--apply', action='store_true', help='Update affected scenario pickles atomically')
    args = parser.parse_args()
    relabel(args.root, args.raw_root, args.apply)
