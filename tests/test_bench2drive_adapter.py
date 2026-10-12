import gzip
import json
import pickle
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from bench2drive_anchors import main as make_anchors
from bench2drive_inspect import check_raw_ego, check_sample
from bench2drive_preprocess import process, read_frame, thin_map_points, local_map, classify_actor
from bench2drive_relabel import relabel


class Bench2DriveAdapterTest(unittest.TestCase):
    def test_carla_bicycle_mapping_and_existing_data_repair(self):
        self.assertEqual(classify_actor({'class': 'vehicle', 'type_id': 'vehicle.bh.crossbike'}), 'TYPE_CYCLIST')
        self.assertEqual(classify_actor({'class': 'vehicle', 'base_type': 'bicycle'}), 'TYPE_CYCLIST')
        self.assertEqual(classify_actor({'class': 'vehicle', 'base_type': 'motorcycle'}), 'TYPE_VEHICLE')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annos = root / 'raw' / 'clip' / 'anno'
            annos.mkdir(parents=True)
            with gzip.open(annos / '00010.json.gz', 'wt', encoding='utf-8') as stream:
                json.dump({'bounding_boxes': [{'id': 7, 'class': 'vehicle', 'base_type': 'bicycle'}]}, stream)
            processed = root / 'processed'
            scenes = processed / 'processed_scenarios_training'
            scenes.mkdir(parents=True)
            path = scenes / 'sample_clip__00000.pkl'
            sample = {'track_infos': {'object_id': ['7'], 'object_type': ['TYPE_VEHICLE']},
                      'map_infos': {'all_polylines': np.zeros((3, 7))}}
            with open(path, 'wb') as stream:
                pickle.dump(sample, stream)
            for split, infos in [('training', [{'scenario_id': 'clip__00000'}]), ('validation', [])]:
                with open(processed / f'processed_scenarios_{split}_infos.pkl', 'wb') as stream:
                    pickle.dump(infos, stream)
            before = path.read_bytes()
            self.assertEqual(relabel(processed, root / 'raw'), (1, 1))
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(relabel(processed, root / 'raw', apply=True), (1, 1))
            with open(path, 'rb') as stream:
                corrected = pickle.load(stream)
            self.assertEqual(corrected['track_infos']['object_type'], ['TYPE_CYCLIST'])
            np.testing.assert_array_equal(corrected['map_infos']['all_polylines'], sample['map_infos']['all_polylines'])
            self.assertEqual(relabel(processed, root / 'raw', apply=True), (0, 0))

    def test_dense_map_is_thinned_and_cropped(self):
        points = np.column_stack((np.linspace(-100, 100, 2001), np.zeros((2001, 2))))
        thinned = thin_map_points(points, 0.5)
        self.assertLessEqual(len(thinned), 402)
        np.testing.assert_allclose(thinned[[0, -1]], points[[0, -1]])
        line = np.column_stack((thinned, np.zeros((len(thinned), 3)), np.full(len(thinned), 2)))
        cropped = local_map([line], np.zeros(3), 10)
        self.assertTrue((np.linalg.norm(cropped[:, :2], axis=1) <= 10).all())
        self.assertLessEqual(len(cropped), 41)

    def test_conversion_schema_and_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips = root / 'raw' / 'v1'
            maps = root / 'raw' / 'maps'
            maps.mkdir(parents=True)
            line = {'Points': [((float(x), 0.0, 0.0), (0, 0, 0)) for x in range(20)],
                    'Type': 'Center'}
            np.savez(maps / 'Town01_HD_map.npz',
                     arr=np.array([['1', {'1': [line]}]], dtype=object))
            names = ['Route_Town01_Train', 'Route_Town01_Val']
            for name in names:
                annos = clips / name / 'anno'
                annos.mkdir(parents=True)
                for frame in range(101):
                    actor = {'class': 'ego_vehicle', 'id': 1,
                             'center': [frame * 0.1, 0, 0],
                             'extent': [2, 1, 1], 'rotation': [0, 0, 0]}
                    with gzip.open(annos / f'{frame:05d}.json.gz', 'wt', encoding='utf-8') as stream:
                        json.dump({'bounding_boxes': [actor]}, stream)
            split_file = root / 'split.json'
            split_file.write_text(json.dumps({'val': [f'v1/{names[1]}']}), encoding='utf-8')
            output = root / 'processed'
            with patch('bench2drive_preprocess.read_frame', wraps=read_frame) as reader:
                process(SimpleNamespace(raw_root=str(root / 'raw'), map_root=str(maps),
                                        split_file=str(split_file), output_root=str(output),
                                        stride=10, map_radius=100, max_scenarios=0))
                self.assertEqual(reader.call_count, 202)  # Overlapping windows reuse frames.
            for split in ('training', 'validation'):
                with open(output / f'processed_scenarios_{split}_infos.pkl', 'rb') as stream:
                    infos = pickle.load(stream)
                self.assertEqual(len(infos), 2)
                with open(output / f'processed_scenarios_{split}' /
                          f"sample_{infos[0]['scenario_id']}.pkl", 'rb') as stream:
                    sample = pickle.load(stream)
                self.assertEqual(sample['track_infos']['trajs'].shape, (1, 91, 10))
                np.testing.assert_allclose(sample['track_infos']['trajs'][0, 10, 3:6], [4, 2, 2])
                self.assertAlmostEqual(sample['track_infos']['trajs'][0, 10, 7], 1.0, places=5)
                self.assertEqual(sample['map_infos']['all_polylines'].shape, (20, 7))
                self.assertEqual(sample['tracks_to_predict']['track_index'], [0])
                check_sample(sample)
                check_raw_ego(sample, clips)
            anchor_file = root / 'anchors.npy'
            make_anchors(SimpleNamespace(root=str(output), output=str(anchor_file),
                                         num_modes=2, max_per_type=100, seed=2025))
            anchors = np.load(anchor_file)
            self.assertEqual(anchors.shape, (5, 2, 80, 3))
            self.assertAlmostEqual(anchors[1, 0, 0, 0], 0.1, places=5)


if __name__ == '__main__':
    unittest.main()
