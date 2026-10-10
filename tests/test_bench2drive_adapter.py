import gzip
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from bench2drive_anchors import main as make_anchors
from bench2drive_preprocess import process


class Bench2DriveAdapterTest(unittest.TestCase):
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
                for frame in range(91):
                    actor = {'class': 'ego_vehicle', 'id': 1,
                             'center': [frame * 0.1, 0, 0],
                             'extent': [1, 2, 1], 'rotation': [0, 0, 0]}
                    with gzip.open(annos / f'{frame:05d}.json.gz', 'wt', encoding='utf-8') as stream:
                        json.dump({'bounding_boxes': [actor]}, stream)
            split_file = root / 'split.json'
            split_file.write_text(json.dumps({'val': [f'v1/{names[1]}']}), encoding='utf-8')
            output = root / 'processed'
            process(SimpleNamespace(raw_root=str(root / 'raw'), map_root=str(maps),
                                    split_file=str(split_file), output_root=str(output),
                                    stride=10, map_radius=100, max_scenarios=0))
            for split in ('training', 'validation'):
                with open(output / f'processed_scenarios_{split}_infos.pkl', 'rb') as stream:
                    infos = pickle.load(stream)
                self.assertEqual(len(infos), 1)
                with open(output / f'processed_scenarios_{split}' /
                          f"sample_{infos[0]['scenario_id']}.pkl", 'rb') as stream:
                    sample = pickle.load(stream)
                self.assertEqual(sample['track_infos']['trajs'].shape, (1, 91, 10))
                self.assertAlmostEqual(sample['track_infos']['trajs'][0, 10, 7], 1.0, places=5)
                self.assertEqual(sample['map_infos']['all_polylines'].shape, (20, 7))
                self.assertEqual(sample['tracks_to_predict']['track_index'], [0])
            anchor_file = root / 'anchors.npy'
            make_anchors(SimpleNamespace(root=str(output), output=str(anchor_file),
                                         num_modes=1, max_per_type=100, seed=2025))
            anchors = np.load(anchor_file)
            self.assertEqual(anchors.shape, (5, 1, 80, 3))
            self.assertAlmostEqual(anchors[1, 0, 0, 0], 0.1, places=5)


if __name__ == '__main__':
    unittest.main()
