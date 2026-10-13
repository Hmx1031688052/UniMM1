"""
Based on https://github.com/ZikangZhou/QCNet/blob/main/predictors/qcnet.py
"""


import os
import math
from pathlib import Path
from typing import Optional
from copy import deepcopy

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
import numpy as np

from losses import NLLLoss
from modules import QCNetEncoder
from modules import MLPDecoder
from layers import MLPLayer
from utils import wrap_angle


class UniMM(pl.LightningModule):

    def __init__(
        self,
        dataset: str,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_freq_bands: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        num_map_layers: int,
        num_agent_layers: int,
        time_span: int,
        pl2pl_knn: int,
        pl2a_knn: int,
        a2a_knn: int,
        anchor_file: str,
        key_frame_interval: int,
        num_historical_steps: int,
        num_prediction_steps: int,
        num_execution_steps: int,
        train_sim_steps: int,
        reg_decoder: str,
        anchor_free: bool,
        num_modes: int,
        open_loop_train: bool,
        match_execution: bool = False,
        align_match: bool = False,
        approx_posterior: bool = False,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        T_max: int = 30,
        submission_dir: str = './',
        submission_file_name: str = 'submission',
        **kwargs
    ) -> None:
        super(UniMM, self).__init__()
        self.save_hyperparameters()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_freq_bands = num_freq_bands
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.num_map_layers = num_map_layers
        self.num_agent_layers = num_agent_layers
        self.time_span = time_span
        self.pl2pl_knn = pl2pl_knn
        self.pl2a_knn = pl2a_knn
        self.a2a_knn = a2a_knn
        self.anchor_file = anchor_file
        self.key_frame_interval = key_frame_interval
        self.num_historical_steps = num_historical_steps
        self.num_prediction_steps = num_prediction_steps
        self.num_execution_steps = num_execution_steps
        self.train_sim_steps = train_sim_steps
        self.reg_decoder = reg_decoder
        self.anchor_free = anchor_free
        self.open_loop_train = open_loop_train
        self.match_execution = match_execution
        self.align_match = align_match
        self.approx_posterior = approx_posterior
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max
        self.submission_dir = submission_dir
        self.submission_file_name = submission_file_name

        ## Load anchor trajectories
        if self.anchor_free:
            self.num_modes = num_modes
        else:
            anchor_trajs = torch.from_numpy(np.load(anchor_file)).float()
            self.anchor_trajs = nn.Parameter(anchor_trajs, requires_grad=False)
            self.num_modes = self.anchor_trajs.size(1)

        ## Initialize the model components
        self.encoder = QCNetEncoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            key_frame_interval=key_frame_interval,
            time_span=time_span,
            pl2pl_knn=pl2pl_knn,
            pl2a_knn=pl2a_knn,
            a2a_knn=a2a_knn,
            num_freq_bands=num_freq_bands,
            num_map_layers=num_map_layers,
            num_agent_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )

        if not self.anchor_free:
            self.scorer = MLPLayer(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=self.num_modes,
            )
        else:
            assert self.reg_decoder is not None

        if self.reg_decoder is not None:
            if 'mlp' in self.reg_decoder:
                self.decoder = MLPDecoder(
                    dataset=dataset,
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    num_prediction_steps=num_prediction_steps,
                    num_freq_bands=num_freq_bands,
                    anchor_free=anchor_free,
                    num_modes=num_modes,
                )
            else:
                raise NotImplementedError

        ## Initialize the loss functions
        self.reg_loss = NLLLoss(component_distribution=['laplace'] * output_dim + ['von_mises'],
                                reduction='none')
        self.cls_loss = nn.CrossEntropyLoss(reduction='none')

        ## Initialize the evaluation metrics
        self.num_eval_rollouts = 32
        self.eval_future_steps = 80

        self.test_predictions = dict()

    def compute_distance_traj(self, src_trajs, tgt_trajs, tgt_mask, widths, lengths):
        '''
        Args:
            src_trajs: (num_agents, num_modes, num_src_steps, [x, y, ..., heading])
            tgt_trajs: (num_agents, num_tgt_steps, [x, y, ..., heading])
            tgt_mask: (num_agents, num_tgt_steps)
            widths: (num_agents,)
            lengths: (num_agents,)
        Returns:
            distances: (num_agents, num_modes, num_tgt_steps)
        '''

        def get_bbox(x, y, heading, width, length):
            corners_local = torch.tensor([
                [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5], [0.5, 0.5]
            ], dtype=x.dtype, device=x.device)  # (4, 2)

            for _ in range(len(x.shape)):
                corners_local = corners_local.unsqueeze(0)

            corners_x = corners_local[..., 0] * length.unsqueeze(-1)
            corners_y = corners_local[..., 1] * width.unsqueeze(-1)

            sin_h = torch.sin(heading)
            cos_h = torch.cos(heading)

            corners_x_rot = cos_h.unsqueeze(-1) * corners_x - sin_h.unsqueeze(-1) * corners_y
            corners_y_rot = sin_h.unsqueeze(-1) * corners_x + cos_h.unsqueeze(-1) * corners_y

            corners_x_final = x.unsqueeze(-1) + corners_x_rot
            corners_y_final = y.unsqueeze(-1) + corners_y_rot

            corners_final = torch.stack([corners_x_final, corners_y_final], dim=-1)
            return corners_final
        
        src_x = src_trajs[..., 0] # (num_agents, num_modes, num_src_steps)
        src_y = src_trajs[..., 1] # (num_agents, num_modes, num_src_steps)
        src_heading = src_trajs[..., -1] # (num_agents, num_modes, num_src_steps)
        src_bbox_trajs = get_bbox(
            src_x, src_y, src_heading,
            widths[:, None, None], lengths[:, None, None]
        ) # (num_agents, num_modes, num_src_steps, 4, 2)

        tgt_x = tgt_trajs[..., 0] # (num_agents, num_tgt_steps)
        tgt_y = tgt_trajs[..., 1] # (num_agents, num_tgt_steps)
        tgt_heading = tgt_trajs[..., -1] # (num_agents, num_tgt_steps)
        tgt_bbox_trajs = get_bbox(
            tgt_x, tgt_y, tgt_heading,
            widths[:, None], lengths[:, None]
        ) # (num_agents, num_tgt_steps, 4, 2)
        
        distances = torch.norm(
            src_bbox_trajs[:, :, :tgt_mask.size(1)] - tgt_bbox_trajs.unsqueeze(1),
            p=2, dim=-1
        ).mean(dim=-1) * tgt_mask.unsqueeze(1) # (num_agents, num_modes, num_tgt_steps)
        return distances

    def posterior_match(
        self, propose_traj, agent_target,
        reg_mask, cls_mask,
        widths, lengths,
        match_execution,
    ):
        distances = self.compute_distance_traj(
            propose_traj, agent_target, reg_mask,
            widths, lengths
        ) # (num_agents, num_modes, num_prediction_steps)

        pred_distance = distances.sum(dim=-1)
        best_mode = pred_distance.argmin(dim=-1)
        
        if match_execution:
            match_distance = distances[..., :self.num_execution_steps].sum(dim=-1)
            match_mode = match_distance.argmin(dim=-1)
            match_cls_mask = reg_mask[:, :self.num_execution_steps].any(dim=1)
        else:
            match_mode = best_mode
            match_cls_mask = cls_mask

        pred_traj = torch.gather(
            propose_traj, 1,
            match_mode[:, None, None, None].expand(
                -1, -1, propose_traj.size(-2), propose_traj.size(-1)
            )
        ).squeeze(1) # (num_agents, num_prediction_steps, traj_dim)

        return best_mode, match_mode, match_cls_mask, pred_traj

    def match_rollout(
        self,
        data: HeteroData,
        sim_steps: int,
        open_loop: bool = False,
        use_model: bool = True,
        match_execution: bool = True,
        match_threshold: float = 0.2,
    ):
        sim_data = deepcopy(data)
        assert self.key_frame_interval <= self.num_execution_steps
        assert self.num_execution_steps <= self.num_prediction_steps
        assert self.input_dim == self.output_dim == 2

        reg_mask_list = []
        cls_mask_list = []
        agent_target_list = []
        best_mode_list = []
        match_mode_list = []
        match_cls_mask_list = []
        replacement_candidates = 0
        replacement_accepted = 0
        sim_masks = [sim_data['agent']['valid_mask'][:, :self.num_historical_steps].clone()]
        sim_positions = [sim_data['agent']['position'][:, :self.num_historical_steps, :self.input_dim].clone()]
        sim_headings = [sim_data['agent']['heading'][:, :self.num_historical_steps].clone()]
        sim_velocities = [sim_data['agent']['velocity'][:, :self.num_historical_steps, :self.input_dim].clone()]
        key_frames = []
        pred_frames = []

        # make sure `num_historical_steps` of data is kept
        sim_data['agent']['valid_mask'] = sim_data['agent']['valid_mask'][:, self.num_historical_steps-1:].clone()
        sim_data['agent']['position'] = sim_data['agent']['position'][:, self.num_historical_steps-1:].clone()
        sim_data['agent']['heading'] = sim_data['agent']['heading'][:, self.num_historical_steps-1:].clone()
        sim_data['agent']['velocity'] = sim_data['agent']['velocity'][:, self.num_historical_steps-1:].clone()

        if not self.anchor_free:
            propose_traj = self.anchor_trajs[
                sim_data['agent']['type'].long(), :, :self.num_prediction_steps
            ]
        else:
            assert use_model, 'Anchor-free model must use model to generate closed-loop samples.'
 
        if not open_loop and self.reg_decoder is not None and use_model:
            model_data = deepcopy(data)
            model_data['agent']['valid_mask'] = model_data['agent']['valid_mask'][:, :self.num_historical_steps].clone()
            model_data['agent']['position'] = model_data['agent']['position'][:, :self.num_historical_steps].clone()
            model_data['agent']['heading'] = model_data['agent']['heading'][:, :self.num_historical_steps].clone()
            model_data['agent']['velocity'] = model_data['agent']['velocity'][:, :self.num_historical_steps].clone()

            map_enc = self.encoder.map_encoder(model_data)
            temporal_cache = None

        current_frame = self.num_historical_steps - 1
        start_frame = (self.num_historical_steps % self.key_frame_interval) - 1
        if start_frame < 0:
            start_frame += self.key_frame_interval
        key_frames.append(
            torch.cat([
                torch.tensor([0], device=self.device),
                torch.arange(start_frame, self.num_historical_steps, self.key_frame_interval, device=self.device)
            ]).sort().values.long().unique()
        )
        for step in range(sim_steps):
            ## whether run out
            if sim_data['agent']['valid_mask'].size(1) <= 1:
                break

            ## predict masks
            reg_mask = sim_data['agent']['valid_mask'][:, 1:1+self.num_prediction_steps].clone()
            current_valid_mask = sim_data['agent']['valid_mask'][:, 0]
            reg_mask[~current_valid_mask] = False
            reg_mask_list.append(reg_mask)

            cls_mask = reg_mask.any(dim=1)
            cls_mask_list.append(cls_mask)

            ## local transformation matrix
            origin_global = sim_data['agent']['position'][:, 0]
            theta_global = sim_data['agent']['heading'][:, 0]
            cos, sin = theta_global.cos(), theta_global.sin()

            rot_mat = torch.zeros(theta_global.shape[0], 2, 2, device=theta_global.device)
            rot_mat[:, 0, 0] = cos
            rot_mat[:, 0, 1] = sin
            rot_mat[:, 1, 0] = -sin
            rot_mat[:, 1, 1] = cos

            inv_rot_mat = torch.zeros_like(rot_mat)
            inv_rot_mat[:, 0, 0] = cos
            inv_rot_mat[:, 0, 1] = -sin
            inv_rot_mat[:, 1, 0] = sin
            inv_rot_mat[:, 1, 1] = cos

            ## local target trajectory
            agent_target = origin_global.new_zeros(sim_data['agent']['num_nodes'], reg_mask.shape[1], 4)
            agent_target[..., :2] = torch.bmm(
                sim_data['agent']['position'][:, 1:1+self.num_prediction_steps, :2] - \
                origin_global[:, :2].unsqueeze(1), inv_rot_mat
            )
            if sim_data['agent']['position'].size(2) == 3:
                agent_target[..., 2] = sim_data['agent']['position'][:, 1:1+self.num_prediction_steps, 2] - \
                    origin_global[:, 2].unsqueeze(-1)
            agent_target[..., 3] = wrap_angle(
                sim_data['agent']['heading'][:, 1:1+self.num_prediction_steps] - \
                theta_global.unsqueeze(-1)
            )
            agent_target_list.append(agent_target)

            ## match anchor
            if not self.anchor_free:
                best_mode, match_mode, match_cls_mask, pred_traj = self.posterior_match(
                    propose_traj, agent_target,
                    reg_mask, cls_mask,
                    data['agent']['width'], data['agent']['length'],
                    match_execution,
                )

                best_mode_list.append(best_mode)
                match_mode_list.append(match_mode)
                match_cls_mask_list.append(match_cls_mask)

                pred_position = pred_traj[..., :self.output_dim]
                pred_heading = pred_traj[..., -1]

            ## refine trajectory
            if not open_loop and self.reg_decoder is not None and use_model:
                if step == 0:
                    model_start_frame = (self.num_historical_steps % self.key_frame_interval) - 1
                    if model_start_frame < 0:
                        model_start_frame += self.key_frame_interval
                    model_key_frames = torch.arange(
                        model_start_frame, self.num_historical_steps, self.key_frame_interval, device=self.device
                    )
                    # add 0 to model_key_frames
                    model_key_frames = torch.cat([torch.tensor([0], device=self.device), model_key_frames])
                    model_key_frames = model_key_frames.sort().values.long().unique()
                else:
                    model_start_frame = self.num_execution_steps % self.key_frame_interval
                    model_key_frames = torch.arange(
                        model_start_frame, self.num_execution_steps+1, self.key_frame_interval, device=self.device
                    )
                    # exclude 0 from model_key_frames
                    model_key_frames = model_key_frames[model_key_frames > 0]
                    model_key_frames = model_key_frames.sort().values.long().unique()
                
                agent_enc = self.encoder.agent_encoder.inference(
                    model_data, map_enc,
                    key_frames=model_key_frames,
                    temporal_cache=temporal_cache,
                )
                temporal_cache = agent_enc['temporal_cache']

                x_a_current = agent_enc['x_a'][:, -1]
                if not self.anchor_free:
                    pred = self.decoder(x_a_current.unsqueeze(1), pred_traj.unsqueeze(1))
                    pred_position = pred['loc_refine_pos'][..., :self.output_dim].squeeze(1)
                    pred_heading = pred['loc_refine_head'][..., -1].squeeze(1)
                else:
                    pred = self.decoder(x_a_current.unsqueeze(1))
                    propose_traj = torch.cat([
                        pred['loc_refine_pos'].squeeze(1),
                        pred['loc_refine_head'].squeeze(1)
                    ], dim=-1) # (num_agents, num_modes, num_prediction_steps, traj_dim)

                    best_mode, match_mode, match_cls_mask, pred_traj = self.posterior_match(
                        propose_traj, agent_target,
                        reg_mask, cls_mask,
                        data['agent']['width'], data['agent']['length'],
                        match_execution,
                    )
                                        
                    best_mode_list.append(best_mode)
                    match_mode_list.append(match_mode)
                    match_cls_mask_list.append(match_cls_mask)

                    pred_position = pred_traj[..., :self.output_dim]
                    pred_heading = pred_traj[..., -1]

            ## update state
            update_mask = sim_data['agent']['valid_mask'][:, 1:1+self.num_execution_steps].clone()
            update_position = sim_data['agent']['position'][:, 1:1+self.num_execution_steps, :self.input_dim].clone()
            update_heading = sim_data['agent']['heading'][:, 1:1+self.num_execution_steps].clone()
            update_velocity = sim_data['agent']['velocity'][:, 1:1+self.num_execution_steps, :self.input_dim].clone()
            
            if not open_loop:
                # filter unmatched trajs to update gt
                valid_match_mask = reg_mask[:, :self.num_execution_steps][:, -1].clone()
                replacement_candidates += valid_match_mask.sum()
                exec_traj = pred_traj[:, :self.num_execution_steps][:, None, -1:]
                exec_gt = agent_target[:, :self.num_execution_steps][:, -1:]
                exec_distance = self.compute_distance_traj(
                    exec_traj, exec_gt, valid_match_mask.unsqueeze(1),
                    data['agent']['width'], data['agent']['length']
                ).squeeze() # (num_agents)
                valid_match_mask[exec_distance > match_threshold] = False
                replacement_accepted += valid_match_mask.sum()

                # convert to global coordinate
                sample_position = torch.matmul(
                    pred_position[:, :, :2], rot_mat
                ) + origin_global[:, :2].unsqueeze(1)
                sample_heading = wrap_angle(pred_heading + theta_global.unsqueeze(1))         
                sample_velocity = (
                    sample_position - torch.cat([origin_global[:, None, :2], sample_position[:, :-1]], dim=1)
                ) * 10 # 10 Hz

                # replace state
                update_mask[valid_match_mask] = True
                update_position[valid_match_mask] = sample_position[valid_match_mask, :self.num_execution_steps]
                update_heading[valid_match_mask] = sample_heading[valid_match_mask, :self.num_execution_steps]
                update_velocity[valid_match_mask] = sample_velocity[valid_match_mask, :self.num_execution_steps]

            sim_data['agent']['valid_mask'] = torch.cat([
                sim_data['agent']['valid_mask'][:, :1], update_mask,
                sim_data['agent']['valid_mask'][:, 1+self.num_execution_steps:],
            ], dim=1)
            sim_data['agent']['valid_mask'] = sim_data['agent']['valid_mask'][:, self.num_execution_steps:]

            sim_data['agent']['position'] = torch.cat([
                sim_data['agent']['position'][:, :1, :self.input_dim], update_position,
                sim_data['agent']['position'][:, 1+self.num_execution_steps:, :self.input_dim],
            ], dim=1)
            sim_data['agent']['position'] = sim_data['agent']['position'][:, self.num_execution_steps:]
            
            sim_data['agent']['heading'] = torch.cat([
                sim_data['agent']['heading'][:, :1], update_heading,
                sim_data['agent']['heading'][:, 1+self.num_execution_steps:],
            ], dim=1)
            sim_data['agent']['heading'] = sim_data['agent']['heading'][:, self.num_execution_steps:]
            
            sim_data['agent']['velocity'] = torch.cat([
                sim_data['agent']['velocity'][:, :1, :self.input_dim], update_velocity,
                sim_data['agent']['velocity'][:, 1+self.num_execution_steps:, :self.input_dim],
            ], dim=1)
            sim_data['agent']['velocity'] = sim_data['agent']['velocity'][:, self.num_execution_steps:]

            if not open_loop and self.reg_decoder is not None and use_model:
                model_data['agent']['valid_mask'] = torch.cat([
                    model_data['agent']['valid_mask'][:, -1:], update_mask,
                ], dim=1)
                model_data['agent']['position'] = torch.cat([
                    model_data['agent']['position'][:, -1:, :self.input_dim], update_position,
                ], dim=1)
                model_data['agent']['heading'] = torch.cat([
                    model_data['agent']['heading'][:, -1:], update_heading,
                ], dim=1)
                model_data['agent']['velocity'] = torch.cat([
                    model_data['agent']['velocity'][:, -1:, :self.input_dim], update_velocity,
                ], dim=1)
            
            sim_masks.append(update_mask)
            sim_positions.append(update_position)
            sim_headings.append(update_heading)
            sim_velocities.append(update_velocity)

            ## key frames & prediction frames
            start_frame = max(current_frame - self.num_execution_steps + \
                self.num_execution_steps % self.key_frame_interval, 0)
            curr_key_frames = torch.arange(
                start_frame, current_frame+1, self.key_frame_interval, device=self.device
            )
            key_frames.append(curr_key_frames)
            pred_frames.append(current_frame)
            current_frame += self.num_execution_steps

        return {
            'reg_mask_list': reg_mask_list,
            'replacement_candidates': replacement_candidates,
            'replacement_accepted': replacement_accepted,
            'cls_mask_list': cls_mask_list,
            'agent_target_list': agent_target_list,
            'best_mode_list': best_mode_list,
            'match_mode_list': match_mode_list,
            'match_cls_mask_list': match_cls_mask_list,
            'sim_masks': torch.cat(sim_masks, dim=1),
            'sim_positions': torch.cat(sim_positions, dim=1),
            'sim_headings': torch.cat(sim_headings, dim=1),
            'sim_velocities': torch.cat(sim_velocities, dim=1),
            'key_frames': torch.cat(key_frames).long().unique(),
            'pred_frames': torch.tensor(pred_frames, device=self.device, dtype=torch.long),
        }

    def inference(
        self,
        data: HeteroData,
        sim_steps: int,
        top_k: int = -1,
        conf_threshold: float = 0.0,
    ):
        sim_data = deepcopy(data)
        assert sim_data['agent']['current_valid_mask'].all()
        assert self.key_frame_interval <= self.num_execution_steps
        assert self.num_execution_steps <= self.num_prediction_steps
        assert self.input_dim == self.output_dim == 2

        key_frames_list = []
        pred_frames_list = []
        sim_positions = []
        sim_velocities = []
        sim_headings = []
        sim_masks = []

        if not self.anchor_free:
            propose_traj = self.anchor_trajs[
                sim_data['agent']['type'].long(), :, :self.num_prediction_steps
            ]

        sim_data['agent']['valid_mask'] = sim_data['agent']['valid_mask'][:, :self.num_historical_steps].clone()
        sim_data['agent']['position'] = sim_data['agent']['position'][:, :self.num_historical_steps].clone()
        sim_data['agent']['heading'] = sim_data['agent']['heading'][:, :self.num_historical_steps].clone()
        sim_data['agent']['velocity'] = sim_data['agent']['velocity'][:, :self.num_historical_steps].clone()
 
        map_enc = self.encoder.map_encoder(sim_data)
        temporal_cache = None
        for step in range(sim_steps):
            ## key frames
            if step == 0:
                start_frame = (self.num_historical_steps % self.key_frame_interval) - 1
                if start_frame < 0:
                    start_frame += self.key_frame_interval
                key_frames = torch.arange(
                    start_frame, self.num_historical_steps, self.key_frame_interval, device=self.device
                )
                # add 0 to key_frames
                key_frames = torch.cat([torch.tensor([0], device=self.device), key_frames])
                key_frames = key_frames.sort().values.long().unique()
            else:
                start_frame = self.num_execution_steps % self.key_frame_interval
                key_frames = torch.arange(
                    start_frame, self.num_execution_steps+1, self.key_frame_interval, device=self.device
                )
                # exclude 0 from key_frames
                key_frames = key_frames[key_frames > 0]
                key_frames = key_frames.sort().values.long().unique()

            key_frames_list.append(key_frames if step == 0 else pred_frame + key_frames)
            pred_frame = (self.num_historical_steps - 1) + step * self.num_execution_steps
            pred_frames_list.append(pred_frame)

            ## encode agent data
            agent_enc = self.encoder.agent_encoder.inference(
                sim_data, map_enc,
                key_frames=key_frames,
                temporal_cache=temporal_cache,
            )
            temporal_cache = agent_enc['temporal_cache']

            ## predict scores
            x_a_current = agent_enc['x_a'][:, -1]
            if not self.anchor_free:
                score = self.scorer(x_a_current)
            else:
                pred = self.decoder(x_a_current.unsqueeze(1))
                score = pred['pi'].squeeze(1)
                loc_refine_pos = pred['loc_refine_pos'][..., :self.output_dim].squeeze(1)
                loc_refine_head = pred['loc_refine_head'][..., -1].squeeze(1)

            ## sample mode
            sample_pi = F.softmax(score, dim=-1)
            sample_pi = sample_pi * (sample_pi > conf_threshold)
            if top_k > 0:
                top_k_values, _ = sample_pi.topk(top_k, dim=-1)
                sample_pi = sample_pi * (sample_pi >= top_k_values[..., -1:])
            sample_mode = torch.multinomial(sample_pi, 1, replacement=True).squeeze(-1)

            if not self.anchor_free:
                pred_traj = torch.gather(
                    propose_traj, 1,
                    sample_mode[:, None, None, None].expand(
                        -1, -1, propose_traj.size(-2), propose_traj.size(-1)
                    )
                ).squeeze(1) # (num_agents, num_prediction_steps, traj_dim)
                pred_position = pred_traj[..., :self.output_dim]
                pred_heading = pred_traj[..., -1]
            else:
                pred_position = torch.gather(
                    loc_refine_pos, 1,
                    sample_mode[:, None, None, None].expand(
                        -1, -1, loc_refine_pos.size(-2), loc_refine_pos.size(-1)
                    )
                ).squeeze(1) # (num_agents, num_prediction_steps, 2)
                pred_heading = torch.gather(
                    loc_refine_head, 1,
                    sample_mode[:, None, None].expand(-1, -1, loc_refine_head.size(-1))
                ).squeeze(1) # (num_agents, num_prediction_steps)

            ## refine trajectory if anchor-based
            if not self.anchor_free and self.reg_decoder is not None:
                pred = self.decoder(x_a_current.unsqueeze(1), pred_traj.unsqueeze(1))
                pred_position = pred['loc_refine_pos'][..., :self.output_dim].squeeze(1)
                pred_heading = pred['loc_refine_head'][..., -1].squeeze(1)

            ## convert to global coordinate
            origin_global = sim_data['agent']['position'][:, -1]
            theta_global = sim_data['agent']['heading'][:, -1]
            cos, sin = theta_global.cos(), theta_global.sin()
            rot_mat = torch.zeros(theta_global.shape[0], 2, 2, device=self.device)
            rot_mat[:, 0, 0] = cos
            rot_mat[:, 0, 1] = sin
            rot_mat[:, 1, 0] = -sin
            rot_mat[:, 1, 1] = cos

            sample_position = torch.matmul(
                pred_position[:, :, :2], rot_mat
            ) + origin_global[:, :2].unsqueeze(1)
            sample_heading = wrap_angle(pred_heading + theta_global.unsqueeze(1))         
            sample_velocity = (
                sample_position - torch.cat([origin_global[:, None, :2], sample_position[:, :-1]], dim=1)
            ) * 10 # 10 Hz

            ## update state
            update_mask = torch.ones(sim_data['agent']['num_nodes'], self.num_execution_steps, device=self.device, dtype=torch.bool)
            update_position = sample_position[:, :self.num_execution_steps]
            update_heading = sample_heading[:, :self.num_execution_steps]
            update_velocity = sample_velocity[:, :self.num_execution_steps]

            sim_data['agent']['valid_mask'] = torch.cat([
                sim_data['agent']['valid_mask'][:, -1:], update_mask,
            ], dim=1)
            sim_data['agent']['position'] = torch.cat([
                sim_data['agent']['position'][:, -1:, :self.input_dim], update_position,
            ], dim=1)
            sim_data['agent']['heading'] = torch.cat([
                sim_data['agent']['heading'][:, -1:], update_heading,
            ], dim=1)
            sim_data['agent']['velocity'] = torch.cat([
                sim_data['agent']['velocity'][:, -1:, :self.input_dim], update_velocity,
            ], dim=1)

            sim_positions.append(update_position)
            sim_velocities.append(update_velocity)
            sim_headings.append(update_heading)
            sim_masks.append(update_mask)

        return {
            'sim_steps': sim_steps,
            'key_frames': torch.cat(key_frames_list).sort().values.long().unique(),
            'pred_frames': torch.tensor(pred_frames_list, device=self.device, dtype=torch.long),
            'sim_masks': torch.cat(sim_masks, dim=1),
            'sim_positions': torch.cat(sim_positions, dim=1),
            'sim_headings': torch.cat(sim_headings, dim=1),
            'sim_velocities': torch.cat(sim_velocities, dim=1),
        }

    def forward(
        self,
        data: HeteroData,
        key_frames: Optional[torch.Tensor] = None,
        pred_frames: Optional[torch.Tensor] = None,
        sample_mode: Optional[torch.Tensor] = None,
    ):
        scene_enc = self.encoder(data, key_frames)
        x_a = scene_enc['x_a']

        if pred_frames is not None and key_frames is not None:
            x_a = x_a[:, torch.isin(key_frames, pred_frames)]

        if self.anchor_free:
            pred = self.decoder(x_a)

            score = pred['pi'] # (num_agents, num_frames, num_modes)
            traj_dist = torch.cat([
                pred['loc_refine_pos'][..., :self.output_dim],
                pred['loc_refine_head'],
                pred['scale_refine_pos'][..., :self.output_dim],
                pred['conc_refine_head']
            ], dim=-1) # (num_agents, num_frames, num_modes, num_prediction_steps, dist_dim)

        else:
            score = self.scorer(x_a.reshape(-1, self.hidden_dim))
            score = score.reshape(x_a.size(0), x_a.size(1), self.num_modes)

            if self.reg_decoder is not None:
                propose_traj = self.anchor_trajs[
                    data['agent']['type'].long(), :, :self.num_prediction_steps
                ]

                if sample_mode is None:
                    sample_mode = score.argmax(dim=-1)
                sample_traj = []
                for t in range(sample_mode.size(1)):
                    sample_traj.append(
                        torch.gather(
                            propose_traj, 1,
                            sample_mode[:, t, None, None, None].expand(
                                -1, -1, propose_traj.size(-2), propose_traj.size(-1)
                            )
                        ).squeeze(1)
                    )
                sample_traj = torch.stack(sample_traj, dim=1)
                
                pred = self.decoder(x_a, sample_traj)
                traj_dist = torch.cat([
                    pred['loc_refine_pos'][..., :self.output_dim],
                    pred['loc_refine_head'],
                    pred['scale_refine_pos'][..., :self.output_dim],
                    pred['conc_refine_head']
                ], dim=-1)
            else:
                traj_dist = None

        return {
            'score': score,
            'traj_dist': traj_dist,
        }

    def compute_loss(self, data, match_dict):
        if self.anchor_free:
            pred = self(
                data,
                key_frames=match_dict['key_frames'],
                pred_frames=match_dict['pred_frames'],
            )

            score = pred['score']
            traj_dist = pred['traj_dist']
            assert traj_dist is not None

            reg_loss_sum = 0
            cls_loss_sum = 0
            reg_cnt = 0
            cls_cnt = 0
            for t, agent_target in enumerate(match_dict['agent_target_list']):
                score_t = score[:, t] # (num_agents, num_modes)
                traj_dist_t = traj_dist[:, t] # (num_agents, num_modes, num_prediction_steps, dist_dim)
                gt = torch.cat([agent_target[..., :self.output_dim], agent_target[..., -1:]], dim=-1) # (num_agents, num_prediction_steps, gt_dim)
                reg_mask_t = match_dict['reg_mask_list'][t] # (num_agents, num_prediction_steps)
                cls_mask_t = match_dict['cls_mask_list'][t] # (num_agents,)

                distances = self.compute_distance_traj(
                    traj_dist_t[..., :self.output_dim+1], gt, reg_mask_t,
                    data['agent']['width'], data['agent']['length']
                ) # (num_agents, num_modes, num_prediction_steps)
                if self.align_match:
                    cls_mask_t = match_dict['match_cls_mask_list'][t]
                    if self.match_execution:
                        distances = distances[..., :self.num_execution_steps]
                pred_distance = distances.sum(dim=-1)
                best_mode = pred_distance.argmin(dim=-1)
                best_traj_dist = torch.gather(
                    traj_dist_t, 1,
                    best_mode[:, None, None, None].expand(
                        -1, -1, traj_dist_t.size(-2), traj_dist_t.size(-1)
                    )
                ).squeeze(1) # (num_agents, num_prediction_steps, dist_dim)
                
                reg_loss_t = self.reg_loss(
                    best_traj_dist[:, :gt.shape[1]],
                    gt[..., :self.output_dim + 1]
                ).sum(dim=-1) * reg_mask_t
                reg_loss_sum += reg_loss_t.sum()
                reg_cnt += reg_mask_t.sum()
                
                cls_loss_t = self.cls_loss(
                    input=score_t[cls_mask_t],
                    target=best_mode[cls_mask_t],
                )
                cls_loss_sum += cls_loss_t.sum()
                cls_cnt += cls_mask_t.sum()
            
            reg_loss = reg_loss_sum / max(reg_cnt, 1)
            cls_loss = cls_loss_sum / max(cls_cnt, 1)

            return reg_loss, cls_loss, reg_cnt, cls_cnt
        
        else:
            if self.align_match:
                best_mode = torch.stack(match_dict['match_mode_list'], dim=1)
                cls_mask = torch.stack(match_dict['match_cls_mask_list'], dim=1)
            else:
                best_mode = torch.stack(match_dict['best_mode_list'], dim=1)
                cls_mask = torch.stack(match_dict['cls_mask_list'], dim=1)

            pred = self(
                data,
                key_frames=match_dict['key_frames'],
                pred_frames=match_dict['pred_frames'],
                sample_mode=best_mode,
            )

            score = pred['score']
            cls_loss = self.cls_loss(
                input=score[cls_mask],
                target=best_mode[cls_mask],
            ).mean()
            cls_cnt = cls_mask.sum()

            if pred['traj_dist'] is not None:
                traj_dist = pred['traj_dist']
                reg_loss_sum = 0
                reg_cnt = 0
                for t, agent_target in enumerate(match_dict['agent_target_list']):
                    traj_dist_t = traj_dist[:, t]
                    gt = torch.cat([agent_target[..., :self.output_dim], agent_target[..., -1:]], dim=-1)
                    reg_mask_t = match_dict['reg_mask_list'][t]
                    
                    reg_loss_t = self.reg_loss(
                        traj_dist_t[:, :gt.shape[1]],
                        gt[..., :self.output_dim + 1]
                    ).sum(dim=-1) * reg_mask_t
                    
                    reg_loss_sum += reg_loss_t.sum()
                    reg_cnt += reg_mask_t.sum()

                reg_loss = reg_loss_sum / max(reg_cnt, 1)
            else:
                reg_loss = 0
                reg_cnt = 1
            
            return reg_loss, cls_loss, reg_cnt, cls_cnt

    def training_step(self, data, batch_idx):
        with torch.no_grad():
            match_dict = self.match_rollout(
                data, sim_steps=self.train_sim_steps,
                open_loop=self.open_loop_train,
                use_model=(self.anchor_free or not self.approx_posterior),
                match_execution=self.match_execution,
            )

        data['agent']['valid_mask'] = match_dict['sim_masks']
        data['agent']['position'] = match_dict['sim_positions']
        data['agent']['heading'] = match_dict['sim_headings']
        data['agent']['velocity'] = match_dict['sim_velocities']

        reg_loss, cls_loss, reg_cnt, cls_cnt = self.compute_loss(data, match_dict)
        self.log('train_reg_loss', reg_loss, on_step=True, on_epoch=True, batch_size=reg_cnt, sync_dist=True, prog_bar=True)
        self.log('train_cls_loss', cls_loss, on_step=True, on_epoch=True, batch_size=cls_cnt, sync_dist=True, prog_bar=True)
        loss = reg_loss + cls_loss
        return loss

    def validation_step(self, data, batch_idx):
        match_dict = self.match_rollout(
            data,
            sim_steps=math.ceil(self.eval_future_steps / self.num_execution_steps),
            open_loop=self.open_loop_train,
            use_model=(self.anchor_free or not self.approx_posterior),
            match_execution=self.match_execution,
        )
        
        data['agent']['valid_mask'] = match_dict['sim_masks']
        data['agent']['position'] = match_dict['sim_positions']
        data['agent']['heading'] = match_dict['sim_headings']
        data['agent']['velocity'] = match_dict['sim_velocities']

        reg_loss, cls_loss, reg_cnt, cls_cnt = self.compute_loss(data, match_dict)
        self.log('val_reg_loss', reg_loss, on_step=False, on_epoch=True, batch_size=reg_cnt, sync_dist=True, prog_bar=True)
        self.log('val_cls_loss', cls_loss, on_step=False, on_epoch=True, batch_size=cls_cnt, sync_dist=True, prog_bar=True)
        return

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('UniMM')
        parser.add_argument('--dataset', type=str, default='waymo')
        parser.add_argument('--input_dim', type=int, default=2)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--output_dim', type=int, default=2)
        parser.add_argument('--num_freq_bands', type=int, default=64)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--head_dim', type=int, default=16)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--num_map_layers', type=int, default=1)
        parser.add_argument('--num_agent_layers', type=int, default=2)
        parser.add_argument('--time_span', type=int, default=30)
        parser.add_argument('--pl2pl_knn', type=int, default=16)
        parser.add_argument('--pl2a_knn', type=int, default=64)
        parser.add_argument('--a2a_knn', type=int, default=32)
        parser.add_argument('--anchor_file', type=str, default='anchors/2048_anchor-trajs_waymo_10Hz-8s_kmeans.npy')
        parser.add_argument('--key_frame_interval', type=int, default=5)
        parser.add_argument('--num_historical_steps', type=int, default=11)
        parser.add_argument('--num_prediction_steps', type=int, default=40)
        parser.add_argument('--num_execution_steps', type=int, default=5)
        parser.add_argument('--train_sim_steps', type=int, default=16)
        parser.add_argument('--reg_decoder', type=str, default=None)
        parser.add_argument('--anchor_free', action='store_true')
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--open_loop_train', action='store_true')
        parser.add_argument('--match_execution', action='store_true')
        parser.add_argument('--align_match', action='store_true')
        parser.add_argument('--approx_posterior', action='store_true')
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--T_max', type=int, default=30)
        parser.add_argument('--submission_dir', type=str, default='./')
        parser.add_argument('--submission_file_name', type=str, default='submission')
        return parent_parser
