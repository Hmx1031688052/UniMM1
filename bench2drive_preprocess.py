"""Convert Bench2Drive Base annotations to UniMM's processed scenario format.

Run this on the server that stores the dataset. Only numpy is required.
"""

import argparse
import gzip
import json
import pickle
import re
import time
from pathlib import Path

import numpy as np


STEPS = 91  # 1 s history + 8 s future at 10 Hz
VALID_CLASSES = {'ego_vehicle': 'TYPE_VEHICLE', 'vehicle': 'TYPE_VEHICLE',
                 'walker': 'TYPE_PEDESTRIAN', 'pedestrian': 'TYPE_PEDESTRIAN',
                 'bicycle': 'TYPE_CYCLIST', 'cyclist': 'TYPE_CYCLIST'}
LINE_TYPES = {'Center': 2, 'Broken': 6, 'Solid': 7, 'SolidSolid': 8}


def town_from_name(name):
    match = re.search(r'(Town\d+(?:HD)?)', name)
    if match is None:
        raise ValueError(f'Cannot determine town from clip name: {name}')
    return match.group(1)


def map_polylines(map_file):
    with np.load(map_file, allow_pickle=True) as archive:
        roads = dict(archive['arr'])
    polylines = []
    for road in roads.values():
        for lane_id, lane in road.items():
            if lane_id == 'Trigger_Volumes':
                continue
            for segment in lane:
                points = np.asarray([item[0] for item in segment['Points']], dtype=np.float32)
                if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 3:
                    continue
                points = points[:, :3].copy()
                points[:, 1] *= -1  # CARLA left hand to right hand
                delta = np.diff(points, axis=0, prepend=points[:1])
                norm = np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-6)
                direction = delta / norm
                kind = LINE_TYPES.get(segment.get('Type'), 0)
                polylines.append(np.column_stack((points, direction,
                                                   np.full(len(points), kind))).astype(np.float32))
    return polylines


def local_map(polylines, center, radius):
    selected = []
    for line in polylines:
        near = np.linalg.norm(line[:, :2] - center[:2], axis=1) <= radius
        if near.any():
            selected.append(line)
    if not selected:
        return np.zeros((2, 7), dtype=np.float32)
    return np.concatenate(selected, axis=0)


def read_frame(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def actor_state(actor):
    center = actor.get('center', actor.get('location'))
    extent = actor.get('extent')
    rotation = actor.get('rotation')
    if center is None or extent is None or rotation is None:
        return None
    center = np.asarray(center, dtype=np.float32)
    extent = np.asarray(extent, dtype=np.float32)
    if center.size < 3 or extent.size < 3 or not np.isfinite(center[:3]).all():
        return None
    state = np.zeros(10, dtype=np.float32)
    state[:3] = center[:3]
    state[1] *= -1
    # Bench2Drive extent order follows CARLA; match its official Zoo box conversion.
    state[3:6] = 2 * extent[[1, 0, 2]]
    state[6] = -np.deg2rad(float(rotation[-1]))
    state[9] = 1
    return state


def frame_actors(frame):
    actors = {}
    for actor in frame.get('bounding_boxes', []):
        actor_type = VALID_CLASSES.get(actor.get('class'))
        if actor_type is None or 'id' not in actor:
            continue
        state = actor_state(actor)
        if state is not None:
            actors[str(actor['id'])] = (actor_type, state)
    return actors


def make_sample(frames, scene_id, polylines, radius):
    actor_frames = [frame_actors(frame) for frame in frames]
    current = actor_frames[10]
    ego_ids = [str(actor['id']) for actor in frames[10]['bounding_boxes']
               if actor.get('class') == 'ego_vehicle' and 'id' in actor]
    if not ego_ids or ego_ids[0] not in current:
        return None
    ego_id = ego_ids[0]
    ids = [ego_id] + sorted(set().union(*(set(actors) for actors in actor_frames)) - {ego_id})
    trajectories = np.zeros((len(ids), STEPS, 10), dtype=np.float32)
    types = []
    for i, actor_id in enumerate(ids):
        actor_type = next((actors[actor_id][0] for actors in actor_frames if actor_id in actors))
        types.append(actor_type)
        for t, actors in enumerate(actor_frames):
            if actor_id in actors:
                trajectories[i, t] = actors[actor_id][1]
        valid = trajectories[i, :, 9].astype(bool)
        indices = np.flatnonzero(valid)
        for first, second in zip(indices[:-1], indices[1:]):
            if second == first + 1:
                trajectories[i, second, 7:9] = (trajectories[i, second, :2] -
                                                  trajectories[i, first, :2]) * 10
        if len(indices) > 1 and indices[1] == indices[0] + 1:
            trajectories[i, indices[0], 7:9] = trajectories[i, indices[1], 7:9]
    current_ids = [i for i in range(len(ids)) if trajectories[i, 10, 9] > 0]
    center = trajectories[0, 10, :3]
    return {
        'scenario_id': scene_id,
        'tfrecord_file': '', 'tfrecord_idx': 0,
        'sdc_track_index': 0,
        'tracks_to_predict': {'track_index': current_ids},
        'track_infos': {'object_id': ids, 'object_type': types, 'trajs': trajectories},
        'map_infos': {'all_polylines': local_map(polylines, center, radius)},
    }


def process(args):
    started = time.monotonic()
    def log(message):
        print(f'[{time.monotonic() - started:8.1f}s] {message}', flush=True)

    if args.stride <= 0:
        raise ValueError('--stride must be positive')
    root = Path(args.raw_root)
    clips_root = root / 'v1' if (root / 'v1').is_dir() else root
    map_root = Path(args.map_root)
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    with open(args.split_file, encoding='utf-8') as stream:
        val_names = {Path(name).name for name in json.load(stream)['val']}
    log(f'Scanning clip directories: {clips_root}')
    clips = sorted(path.parent for path in clips_root.glob('*/anno') if path.is_dir())
    if not clips:
        raise FileNotFoundError(f'No clip/anno directories under {clips_root}')
    map_cache = {}
    infos = {'training': [], 'validation': []}
    log(f'Found {len(clips)} clips; output: {output}')
    for clip_index, clip in enumerate(clips, 1):
        log(f'Clip {clip_index}/{len(clips)}: {clip.name}')
        town = town_from_name(clip.name)
        if town not in map_cache:
            matches = sorted(map_root.glob(f'{town}_HD_map.npz'))
            if not matches:
                raise FileNotFoundError(f'Missing {town}_HD_map.npz in {map_root}')
            log(f'Loading map: {matches[0]}')
            map_cache[town] = map_polylines(matches[0])
            log(f'Map ready: {len(map_cache[town])} polylines')
        split = 'validation' if clip.name in val_names else 'training'
        frame_paths = sorted((clip / 'anno').glob('*.json.gz'))
        log(f'{split}: {len(frame_paths)} annotation files')
        if len(frame_paths) < STEPS:
            continue
        frame_ids = [int(path.name.split('.')[0]) for path in frame_paths]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError(f'Duplicate frame index in {clip}')
        frame_cache = {}
        for window_index, start in enumerate(range(0, len(frame_paths) - STEPS + 1, args.stride), 1):
            if frame_ids[start + STEPS - 1] - frame_ids[start] != STEPS - 1:
                continue
            window_paths = frame_paths[start:start + STEPS]
            # Retain only the current window; overlapping frames are read once.
            wanted = set(window_paths)
            frame_cache = {path: frame for path, frame in frame_cache.items() if path in wanted}
            verbose = bool(args.max_scenarios) or window_index == 1 or window_index % 10 == 0
            missing = [path for path in window_paths if path not in frame_cache]
            if verbose:
                log(f'Window {window_index}: reading {len(missing)} new annotation files')
            for read_index, path in enumerate(missing, 1):
                if verbose and (read_index == 1 or read_index % 25 == 0):
                    log(f'Reading {read_index}/{len(missing)}: {path.name}')
                frame_cache[path] = read_frame(path)
            frames = [frame_cache[path] for path in window_paths]
            scene_id = f'{clip.name}__{frame_ids[start]:05d}'
            if verbose:
                log(f'Building trajectories and selecting map for {scene_id}')
            sample = make_sample(frames, scene_id, map_cache[town], args.map_radius)
            if sample is None:
                log(f'Skipped {scene_id}: no valid ego_vehicle at current frame')
                continue
            destination = output / f'processed_scenarios_{split}'
            destination.mkdir(parents=True, exist_ok=True)
            with open(destination / f'sample_{scene_id}.pkl', 'wb') as stream:
                pickle.dump(sample, stream, protocol=pickle.HIGHEST_PROTOCOL)
            infos[split].append({'scenario_id': scene_id, 'tfrecord_file': '', 'tfrecord_idx': 0})
            if verbose:
                log(f'Saved {scene_id}; total {sum(map(len, infos.values()))} scenarios')
            if args.max_scenarios and sum(map(len, infos.values())) >= args.max_scenarios:
                break
        if args.max_scenarios and sum(map(len, infos.values())) >= args.max_scenarios:
            break
    for split, records in infos.items():
        with open(output / f'processed_scenarios_{split}_infos.pkl', 'wb') as stream:
            pickle.dump(records, stream, protocol=pickle.HIGHEST_PROTOCOL)
        log(f'{split}: {len(records)} scenarios')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', required=True, help='Bench2Drive Base root or its v1 directory')
    parser.add_argument('--map-root', required=True, help='Directory containing Town*_HD_map.npz')
    parser.add_argument('--split-file', required=True, help='Official Base train/val split JSON')
    parser.add_argument('--output-root', required=True, help='Server path for processed scenarios')
    parser.add_argument('--stride', type=int, default=10, help='Window stride in frames')
    parser.add_argument('--map-radius', type=float, default=100.0)
    parser.add_argument('--max-scenarios', type=int, default=0, help='For a small smoke run')
    process(parser.parse_args())
