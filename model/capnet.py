import torch
import torch.nn as nn
import numpy as np
import sys
import os

from model.pointgroup_module import PointGroupModule
from util.box_util import get_3d_box_batch

sys.path.append(os.path.join(os.getcwd(), "lib"))  # HACK add the lib folder

from model.graph_module import GraphModule
from model.caption_module import SceneCaptionModule, TopDownSceneCaptionModule


def get_bbox_parameters(locs_float, proposals_idx, proposals_offset):
    bbox_parameters = []
    for i in range(1, len(proposals_offset)):
        point_cloud = locs_float[proposals_idx[proposals_offset[i - 1]:proposals_offset[i], 1]]
        bbox_parameters.append(bbox_from_point_cloud(point_cloud))
    bbox_parameters = torch.stack(bbox_parameters, dim=0)
    return bbox_parameters


class CapNet(nn.Module):
    def __init__(self, num_class, vocabulary, embeddings, num_proposal=256, num_locals=-1,
                 no_caption=False, use_topdown=False, query_mode="corner",
                 graph_mode="graph_conv", num_graph_steps=0, use_relation=False, graph_aggr="add",
                 use_orientation=False, num_bins=6, use_distance=False,
                 emb_size=300, hidden_size=512, pg_prepare_epochs=128, pg_input_c=3, pg_m=128, pg_ublocks=(1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 3, 3),
                 pg_cluster_npoint_thre=50,
                 pg_cluster_radius=0.3, pg_meanActive=50, pg_shift_meanActive=300, pg_score_scale=50,
                 pg_score_fullscale=14,
                 pg_score_mode=4, pg_block_reps=2, pg_block_residual=True, pg_fix_module=[], proposal_score_thre=0.09):
        super().__init__()

        self.num_class = num_class
        self.num_proposal = num_proposal
        self.no_caption = no_caption
        self.num_graph_steps = num_graph_steps

        self.pg_prepare_epochs = pg_prepare_epochs

        self.proposal_score_thre = proposal_score_thre
        self.pg_ublock_layers = pg_ublocks

        # --------- PROPOSAL GENERATION ---------
        # -- PointGroup --

        self.pointgroup = PointGroupModule(classes=self.num_class,
                                           prepare_epochs=self.pg_prepare_epochs,
                                           input_c=pg_input_c,
                                           m=pg_m,
                                           cluster_npoint_thre=pg_cluster_npoint_thre,
                                           cluster_radius=pg_cluster_radius,
                                           cluster_meanActive=pg_meanActive,
                                           cluster_shift_meanActive=pg_shift_meanActive,
                                           score_scale=pg_score_scale,
                                           score_fullscale=pg_score_fullscale,
                                           score_mode=pg_score_mode,
                                           block_reps=pg_block_reps,
                                           block_residual=pg_block_residual,
                                           fix_module=pg_fix_module,
                                           ublock_layers=pg_ublocks,
                                           )

        if use_relation: assert use_topdown  # only enable use_relation in topdown captioning module

        # -- Graph Module --
        if num_graph_steps > 0:
            self.graph = GraphModule(pg_ublocks[-1]*pg_m, pg_ublocks[-1]*pg_m, num_graph_steps, num_proposal, pg_ublocks[-1]*pg_m, num_locals,
                                     query_mode, graph_mode, return_edge=use_relation, graph_aggr=graph_aggr,
                                     return_orientation=use_orientation, num_bins=num_bins,
                                     return_distance=use_distance)

        # -- Caption Module --
        if not no_caption:
            if use_topdown:
                self.caption = TopDownSceneCaptionModule(vocabulary, embeddings, emb_size, pg_ublocks[-1]*pg_m,
                                                         hidden_size, num_proposal, num_locals, query_mode,
                                                         use_relation)
            else:
                self.caption = SceneCaptionModule(vocabulary, embeddings, emb_size, pg_ublocks[-1]*pg_m, hidden_size,
                                                  num_proposal)

    def forward(self, data_dict, use_tf=True, is_eval=False):
        """ Forward pass of the network

        Args:
            data_dict: dict
                {
                    point_clouds, 
                    lang_feat
                }

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """

        #######################################
        #                                     #
        #           DETECTION BRANCH          #
        #                                     #
        #######################################

        batch_size = data_dict['batch_size']
        assert batch_size == data_dict['point_clouds'].shape[0]

        # -- PointGroup --
        data_dict = self.pointgroup(data_dict)

        if data_dict['epoch'] <= self.pg_prepare_epochs:
            return data_dict

        # -- Interfacing --
        proposal_feats = data_dict['proposal_feats']  # (nProposal, C)
        proposals_idx = data_dict[
            'proposal_idxs'].long()  # (sumNPoint, 2) [:, 0] is id of cluster, [:, 1] is id of point
        proposals_offset = data_dict[
            'proposal_offsets'].long()  # (nProposal,) = At which index in proposals_idxs the next cluster begins
        proposals_scores = data_dict['proposal_scores'].clone()  # .detach()  # (nProposal, 1)
        proposals_scores = torch.sigmoid(proposals_scores)

        bbox_mask = torch.where(proposals_scores > self.proposal_score_thre, 1, 0)

        batch_offsets = data_dict['batch_offsets']  # int (B+1)

        bbox_parameters = get_bbox_parameters(data_dict['locs_float'], proposals_idx, proposals_offset)
        bbox_corners = bbox_corners_from_params_batch(bbox_parameters)

        point_sem_labels = data_dict['semantic_preds']
        proposal_sem_labels: list = []

        for i in range(1, len(proposals_offset)):
            point_cloud = proposals_idx[proposals_offset[i - 1]:proposals_offset[i], 1]
            assert len(torch.unique(point_sem_labels[point_cloud])) == 1
            proposal_sem_labels.append(point_sem_labels[point_cloud[0]])

        proposal_sem_labels: torch.Tensor = torch.stack(proposal_sem_labels)

        # ===
        data_dict["bbox_corner"] = separate_into_batches(batch_size, proposals_idx, proposals_offset, batch_offsets,
                                                         bbox_corners,
                                                         self.num_proposal).cuda()  # bounding box corner coordinates
        data_dict["bbox_feature"] = separate_into_batches(batch_size, proposals_idx, proposals_offset, batch_offsets,
                                                          proposal_feats, self.num_proposal).cuda()
        data_dict["bbox_mask"] = separate_into_batches(batch_size, proposals_idx, proposals_offset, batch_offsets,
                                                       bbox_mask.squeeze(), self.num_proposal).cuda()
        data_dict['bbox_scores'] = separate_into_batches(batch_size, proposals_idx, proposals_offset, batch_offsets,
                                                         proposals_scores.squeeze(), self.num_proposal).cuda()
        data_dict['bbox_sems'] = separate_into_batches(batch_size, proposals_idx, proposals_offset, batch_offsets,
                                                       proposal_sem_labels, self.num_proposal).cuda()
        data_dict['bbox_parameters'] = separate_into_batches(batch_size, proposals_idx, proposals_offset, batch_offsets,
                                                             bbox_parameters, self.num_proposal).cuda()
        data_dict['sem_cls'] = data_dict['bbox_sems']
        # ===

        #######################################
        #                                     #
        #           GRAPH ENHANCEMENT         #
        #                                     #
        #######################################

        if self.num_graph_steps > 0: data_dict = self.graph(data_dict)

        #######################################
        #                                     #
        #            CAPTION BRANCH           #
        #                                     #
        #######################################

        # --------- CAPTION GENERATION ---------
        if not self.no_caption:
            data_dict = self.caption(data_dict, use_tf, is_eval)

        return data_dict


def separate_into_batches(batch_size, proposals_idx, proposals_offset, batch_offsets, proposal_features: torch.Tensor,
                          max_proposals):
    """

    :param proposal_features: (#Proposals, d1...dn)
    :return: (#Batches, MAX_PROPOSALS, d1...dn)
    """
    t_list = [[] for _ in range(batch_size)]
    for idx, e in enumerate(proposal_features):
        sample_point_in_proposal = proposals_idx[proposals_offset[idx]][1]
        batch = get_batch_id_of_point(sample_point_in_proposal, batch_offsets)

        t_list[batch].append(e)

    if len(proposal_features.shape) == 1:
        t_stacked = [torch.zeros(max_proposals) for _ in range(batch_size)]
    else:
        t_stacked = [torch.zeros(max_proposals, *proposal_features.shape[1:]) for _ in range(batch_size)]

    for i in range(batch_size):
        try:
            stacked = torch.stack(t_list[i])
            t_stacked[i][:stacked.shape[0], ...] = stacked[:max_proposals]
            if stacked.shape[0] > max_proposals:
                print(f"Warning, truncating proposals: max is {max_proposals}, got {stacked.shape[0]}")
        except RuntimeError:
            print("Warning: no proposals")
            pass

    return torch.stack(t_stacked)


def get_batch_id_of_point(point_id, batch_offsets):
    for i, batch_offset in enumerate(batch_offsets):
        if point_id < batch_offset:
            return i - 1

    raise ValueError("Point id out of bounds")


def bbox_corners_from_params_batch(params: torch.Tensor):
    return torch.tensor(
        get_3d_box_batch(box_size=params[:, 3:], center=params[:, :3], heading_angle=torch.zeros(params.shape[0])))


def bbox_from_point_cloud(point_cloud):
    # Compute axis aligned box
    # An axis aligned bounding box is parameterized by
    # (cx,cy,cz) and (dx,dy,dz) and label id
    # where (cx,cy,cz) is the center point of the box,
    # dx is the x-axis length of the box.
    xmin = torch.min(point_cloud[:, 0])
    ymin = torch.min(point_cloud[:, 1])
    zmin = torch.min(point_cloud[:, 2])
    xmax = torch.max(point_cloud[:, 0])
    ymax = torch.max(point_cloud[:, 1])
    zmax = torch.max(point_cloud[:, 2])
    bbox = torch.tensor(
        [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2, xmax - xmin, ymax - ymin, zmax - zmin])
    return bbox


def bbox_from_point_cloud_quantiles(point_cloud, quantile=0.01):
    # Compute axis aligned box
    # An axis aligned bounding box is parameterized by
    # (cx,cy,cz) and (dx,dy,dz) and label id
    # where (cx,cy,cz) is the center point of the box,
    # dx is the x-axis length of the box.
    xmin = torch.quantile(point_cloud[:, 0], quantile)
    ymin = torch.quantile(point_cloud[:, 1], quantile)
    zmin = torch.quantile(point_cloud[:, 2], quantile)
    xmax = torch.quantile(point_cloud[:, 0], 1 - quantile)
    ymax = torch.quantile(point_cloud[:, 1], 1 - quantile)
    zmax = torch.quantile(point_cloud[:, 2], 1 - quantile)
    bbox = torch.tensor(
        [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2, xmax - xmin, ymax - ymin, zmax - zmin])
    return bbox
