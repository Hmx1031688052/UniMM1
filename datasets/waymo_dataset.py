"""
Based on https://github.com/sshaoshuai/MTR/blob/master/mtr/datasets/waymo/waymo_dataset.py and
    https://github.com/ZikangZhou/QCNet/blob/main/datasets/argoverse_v2_dataset.py
"""


import os
import sys
sys.path.append(os.getcwd())

import pickle
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import HeteroData


class WaymoDataset(Dataset):
    """Dataset class for Waymo Open Motion Dataset.

    Args:
        data_root (string): the root folder of the dataset.
        split_dir (string): specify the directory of the split.
        info_file (string): the file name of the info file.
        interval (int, Optional): the interval of the time steps. (default: 1)
        dim (int, Optional): 2D or 3D data. (default: 3)
        num_historical_steps (int, Optional): the number of historical time steps. (default: 11)
        num_future_steps (int, Optional): the number of future time steps. (default: 80)
        filter_current_valid_agents (boolean, Optional): if True, filter out agents that are not valid at the current time step.
            (default: True)
        max_number_agents (int, Optional): the maximum number of agents. if -1, no limit. (default: -1)
        predict_unseen_agents (boolean, Optional): if False, filter out agents that are unseen during the historical
            time steps. (default: False)
        vector_repr (boolean, Optional): if True, a time step t is valid only when both t and t-1 are valid.
            (default: True)
        num_points_each_polyline (int, Optional): the number of points for each polyline. (default: 11)
        map_downsample_factor (int, Optional): the downsample factor of the map. (default: 5)
        test (bool, Optional): if True, the dataset is used for testing. (default: False)
    """
    def __init__(
        self,
        data_root: str,
        split_dir: str,
        info_file: str,
        interval: int = 1,
        dim: int = 3,
        num_historical_steps: int = 11,
        num_future_steps: int = 80,
        filter_current_valid_agents: bool = True,
        max_number_agents: int = -1,
        predict_unseen_agents: bool = False,
        vector_repr: bool = True,
        num_points_each_polyline: int = 11,
        map_downsample_factor: int = 5,
        test: bool = False,
    ) -> None:
        self.data_root = os.path.expanduser(os.path.normpath(data_root))
        self.data_path = os.path.join(self.data_root, split_dir)

        self.interval = interval
        self.infos = self.get_all_infos(os.path.join(self.data_root, info_file))

        self.dim = dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_steps = num_historical_steps + num_future_steps
        self.filter_current_valid_agents = filter_current_valid_agents
        self.max_number_agents = max_number_agents
        self.predict_unseen_agents = predict_unseen_agents
        self.vector_repr = vector_repr
        self.num_points_each_polyline = num_points_each_polyline
        self.map_downsample_factor = map_downsample_factor
        self.test = test

        self._agent_types = ['TYPE_UNSET', 'TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST', 'TYPE_OTHER']

        super(WaymoDataset, self).__init__()

    def get_all_infos(self, info_path):
        print(f'Start to load infos from {info_path}')
        with open(info_path, 'rb') as f:
            src_infos = pickle.load(f)

        infos = src_infos[::self.interval]
        print(f'Number of scenes loaded: {len(infos)}')

        return infos

    def len(self):
        return len(self.infos)
    
    def get(self, idx: int) -> HeteroData:
        ret_infos = self.create_scene_level_data(idx)
        return HeteroData(ret_infos)

    def create_scene_level_data(self, idx: int) -> dict:
        info = self.infos[idx]
        scenario_id = info['scenario_id']
        with open(os.path.join(self.data_path, f'sample_{scenario_id}.pkl'), 'rb') as f:
            sample = pickle.load(f)

        data = dict()
        data['scenario_id'] = scenario_id
        data['tfrecord_file'] = info['tfrecord_file']
        data['tfrecord_idx'] = info['tfrecord_idx']
        data['agent'] = self.get_agent_features(sample)
        
        if sample['map_infos']['all_polylines'].__len__() == 0:
            sample['map_infos']['all_polylines'] = np.zeros((2, 7), dtype=np.float32)
            print(f'Warning: empty HDMap {scenario_id}')
        data.update(self.get_map_features(sample['map_infos']))

        return data

    def get_agent_features(self, sample: dict) -> Dict[str, Any]:
        track_infos = sample['track_infos']
        sdc_track_index = sample['sdc_track_index']
        track_index_to_predict = sample['tracks_to_predict']['track_index']

        object_id = track_infos['object_id']
        trajs_full = track_infos['trajs']  # (num_objects, num_steps, [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid])
        assert trajs_full.shape[1] <= self.num_steps
        if not self.test and trajs_full.shape[1] < self.num_steps:
            trajs_full = np.concatenate([
                trajs_full,
                np.zeros((trajs_full.shape[0], self.num_steps - trajs_full.shape[1], trajs_full.shape[2]))
            ], axis=1)

        object_type = np.array([self._agent_types.index(TYPE) for TYPE in track_infos['object_type']])  # (num_objects)
        
        valid_mask_full = (trajs_full[:, :, -1] > 0)  # (num_objects, num_steps)
        trajs_full[~valid_mask_full] = 0

        filter_mask = (np.ones(len(object_id)) > 0)
        if self.filter_current_valid_agents:  # filter out agents that are not valid at the current time step
            filter_mask = valid_mask_full[:, self.num_historical_steps - 1]
        elif not self.predict_unseen_agents:  # filter out agents that are unseen during the historical time steps
            valid_mask_past = valid_mask_full[:, :self.num_historical_steps]
            filter_mask = (valid_mask_past.sum(axis=-1) > 0)

        if self.max_number_agents > 0:
            # TODO choose the closest agents to the AV
            if filter_mask.sum() > self.max_number_agents:
                dist_to_av = np.linalg.norm(
                    trajs_full[:, self.num_historical_steps - 1, :2] - \
                        trajs_full[sdc_track_index, self.num_historical_steps - 1, :2],
                    axis=-1
                )
                dist_to_av[~filter_mask] = np.inf
                closest_agent_idx = np.argsort(dist_to_av)
                filter_mask = np.zeros(len(object_id), dtype=bool)
                filter_mask[closest_agent_idx[:self.max_number_agents]] = True

        agent_ids = list(np.array(object_id)[filter_mask])
        num_agents = len(agent_ids)

        valid_mask = torch.tensor(valid_mask_full[filter_mask], dtype=torch.bool)  # (num_agents, num_steps)
        current_valid_mask = valid_mask[:, self.num_historical_steps - 1].clone()  # (num_agents)

        trajs_full = trajs_full[filter_mask]  # (num_agents, num_steps, 10)
        if self.vector_repr:  # recover the unvalid frame if the next frame is valid
            recover_mask = (~valid_mask[:, :-1] & valid_mask[:, 1:])
            pad_mask = torch.zeros(num_agents, 1, dtype=torch.bool)
            current_mask = torch.cat([recover_mask, pad_mask], dim=1)  # (num_agents, num_steps)
            next_mask = torch.cat([pad_mask, recover_mask], dim=1)  # (num_agents, num_steps)
            # TODO recover (x, y) using (vel_x, vel_y)
            next_position = trajs_full[next_mask, :2]
            next_velocity = trajs_full[next_mask, 7:9]
            current_position = next_position - next_velocity * 0.1  # 10 Hz
            trajs_full[current_mask, :2] = current_position

        agent_type = torch.tensor(object_type[filter_mask], dtype=torch.uint8)  # (num_agents)
        
        agent_category = torch.zeros(len(object_id), dtype=torch.uint8)
        agent_category[track_index_to_predict] = 1
        agent_category = agent_category[filter_mask]  # (num_agents)

        position = torch.tensor(trajs_full[..., :self.dim], dtype=torch.float)  # (num_agents, num_steps, dim)
        
        heading = torch.tensor(
            trajs_full[..., 6],
            dtype=torch.float
        )  # (num_agents, num_steps)
        heading_cos = torch.cos(heading)
        heading_sin = torch.sin(heading)

        velocity = torch.zeros(num_agents, trajs_full.shape[1], self.dim, dtype=torch.float)
        velocity[:, :, :2] = torch.from_numpy(trajs_full[..., 7:9]).float()

        length = torch.tensor(trajs_full[:, self.num_historical_steps - 1, 3], dtype=torch.float)  # (num_agents)
        width = torch.tensor(trajs_full[:, self.num_historical_steps - 1, 4], dtype=torch.float)  # (num_agents)
        height = torch.tensor(trajs_full[:, self.num_historical_steps - 1, 5], dtype=torch.float)  # (num_agents)

        return {
            'num_nodes': num_agents,
            'valid_mask': valid_mask,
            'current_valid_mask': current_valid_mask,
            'id': agent_ids,
            'type': agent_type,
            'category': agent_category,
            'position': position,
            'heading': heading,
            'heading_cos': heading_cos,
            'heading_sin': heading_sin,
            'velocity': velocity,
            'length': length,
            'width': width,
            'height': height,
        }


    def get_map_features(
        self, map_infos: dict
    ) -> Dict[Union[str, Tuple[str, str, str]], Any]:
        
        all_polylines = map_infos['all_polylines']  # (num_points, [x, y, z, dir_x, dir_y, dir_z, global_type])
        batch_polylines, batch_polylines_mask = \
            self.generate_batch_polylines_from_map(
                all_polylines, num_points_each_polyline=self.num_points_each_polyline
            )  # (num_polylines, num_points_each_polyline, 7), (num_polylines, num_points_each_polyline)
        batch_polylines_mask = (batch_polylines_mask > 0)
        batch_polylines[~batch_polylines_mask] = 0
        # TODO downsample the map points
        batch_polylines = batch_polylines[:, ::self.map_downsample_factor]
        batch_polylines_mask = batch_polylines_mask[:, ::self.map_downsample_factor]
        map_points = batch_polylines[batch_polylines_mask]  # (num_points, 7)
        
        num_polygons = batch_polylines.size(0)
        
        polygon_position = (batch_polylines[:, :, :self.dim].sum(dim=1) / torch.clamp_min(
            batch_polylines_mask.sum(dim=1)[:, None], min=1.0)).float()  # (num_polygons, dim)
        polygon_orientation = torch.atan2(
            batch_polylines[:, :, 4].sum(dim=1),
            batch_polylines[:, :, 3].sum(dim=1)
        ).float()  # (num_polygons)
        
        polygon_type = batch_polylines[:, :, -1][
            torch.arange(num_polygons), torch.argmax(batch_polylines_mask.int(), dim=1)
        ].long()  # (num_polygons)
        polygon_type[polygon_type < 0] = 0
        polygon_type[polygon_type >= 20] = 0

        num_points = map_points.size(0)

        point_position = map_points[:, :self.dim].clone().float()  # (num_points, dim)
        point_orientation = torch.atan2(map_points[:, 4], map_points[:, 3]).float()  # (num_points)

        point_type = map_points[:, -1].clone().long()  # (num_points)
        point_type[point_type < 0] = 0
        point_type[point_type >= 20] = 0

        polygon_idx, _ = batch_polylines_mask.nonzero(as_tuple=True)
        point_to_polygon_edge_index = torch.stack(
            [torch.arange(num_points, dtype=torch.long), polygon_idx], dim=0
        ) # (2, num_points)

        polygon_to_polygon_edge_index = torch.tensor([[], []], dtype=torch.long)
        polygon_to_polygon_type = torch.tensor([], dtype=torch.uint8)

        map_data = {
            'map_polygon': {},
            'map_point': {},
            ('map_point', 'to', 'map_polygon'): {},
            ('map_polygon', 'to', 'map_polygon'): {},
        }
        map_data['map_polygon']['num_nodes'] = num_polygons
        map_data['map_polygon']['position'] = polygon_position
        map_data['map_polygon']['orientation'] = polygon_orientation
        map_data['map_polygon']['type'] = polygon_type
        
        map_data['map_point']['num_nodes'] = num_points
        map_data['map_point']['position'] = point_position
        map_data['map_point']['orientation'] = point_orientation
        map_data['map_point']['type'] = point_type

        map_data['map_point', 'to', 'map_polygon']['edge_index'] = point_to_polygon_edge_index
        map_data['map_polygon', 'to', 'map_polygon']['edge_index'] = polygon_to_polygon_edge_index
        map_data['map_polygon', 'to', 'map_polygon']['type'] = polygon_to_polygon_type

        return map_data

    @staticmethod
    def generate_batch_polylines_from_map(polylines, point_sampled_interval=1, vector_break_dist_thresh=1.0, num_points_each_polyline=20):
        """
        Args:
            polylines (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, global_type]

        Returns:
            ret_polylines: (num_polylines, num_points_each_polyline, 7)
            ret_polylines_mask: (num_polylines, num_points_each_polyline)
        """
        point_dim = polylines.shape[-1]

        sampled_points = polylines[::point_sampled_interval]
        sampled_points_shift = np.roll(sampled_points, shift=1, axis=0)
        buffer_points = np.concatenate((sampled_points[:, 0:2], sampled_points_shift[:, 0:2]), axis=-1) # [ed_x, ed_y, st_x, st_y]
        buffer_points[0, 2:4] = buffer_points[0, 0:2]

        break_idxs = (np.linalg.norm(buffer_points[:, 0:2] - buffer_points[:, 2:4], axis=-1) > vector_break_dist_thresh).nonzero()[0]
        polyline_list = np.array_split(sampled_points, break_idxs, axis=0)
        ret_polylines = []
        ret_polylines_mask = []

        def append_single_polyline(new_polyline):
            cur_polyline = np.zeros((num_points_each_polyline, point_dim), dtype=np.float32)
            cur_valid_mask = np.zeros((num_points_each_polyline), dtype=np.int32)
            cur_polyline[:len(new_polyline)] = new_polyline
            cur_valid_mask[:len(new_polyline)] = 1
            ret_polylines.append(cur_polyline)
            ret_polylines_mask.append(cur_valid_mask)

        for k in range(len(polyline_list)):
            if polyline_list[k].__len__() <= 0:
                continue
            for idx in range(0, len(polyline_list[k]), num_points_each_polyline):
                append_single_polyline(polyline_list[k][idx: idx + num_points_each_polyline])

        ret_polylines = np.stack(ret_polylines, axis=0)
        ret_polylines_mask = np.stack(ret_polylines_mask, axis=0)

        ret_polylines = torch.from_numpy(ret_polylines)
        ret_polylines_mask = torch.from_numpy(ret_polylines_mask)

        # # CHECK the results
        # polyline_center = ret_polylines[:, :, 0:2].sum(dim=1) / ret_polyline_valid_mask.sum(dim=1).float()[:, None]  # (num_polylines, 2)
        # center_dist = (polyline_center - ret_polylines[:, 0, 0:2]).norm(dim=-1)
        # assert center_dist.max() < 10
        return ret_polylines, ret_polylines_mask


if __name__ == '__main__':
    import torch
    from datasets import WaymoDataset

    val_dataset = WaymoDataset(
        data_root='./data/waymo/processed/',
        split_dir='processed_scenarios_validation',
        info_file='processed_scenarios_validation_infos.pkl',
    )

    data = val_dataset[0]
