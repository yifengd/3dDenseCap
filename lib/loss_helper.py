# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from util.nn_distance import nn_distance, huber_loss
from lib.ap_helper import parse_predictions
from util.box_util import get_3d_box, get_3d_box_batch, box3d_iou, box3d_iou_batch, box3d_iou_batch_tensor
from lib.pointgroup_ops.functions import pointgroup_ops
from config.scan2cap_config import CONF

FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
LOSS_WEIGHTS = CONF.PG.LOSS_WEIGHTS
#GT_VOTE_FACTOR = 3 # number of GT votes per point
#OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8] # put larger weights on positive objectness


def compute_pg_loss(data_dict, config, prepare_epochs):
    """ Compute the loss of pointgroup module components

    Args:
        data_dict: dict (read-only)

    Returns:
        pg_loss: dict with keys semantic_loss, offset_dir_loss, offset_norm_loss, score_loss, pg_loss
    """

    #semantic loss
    semantic_scores = data_dict['semantic_scores']
    semantic_labels = data_dict['pg_semantic_labels']
    # semantic_scores: (N, nClass), float32, cuda
    # semantic_labels: (N), long, cuda

    semantic_loss = nn.CrossEntropyLoss(ignore_index=0)(semantic_scores, semantic_labels).cuda()
    semantic_loss = (semantic_loss, semantic_scores.shape[0])


    #offset loss
    pt_offsets = data_dict['pt_offsets'].cuda()
    coords = data_dict['locs_float'].cuda()
    instance_info = data_dict['instance_info'].cuda()
    instance_labels = data_dict['pg_instance_labels'].cuda()
    # pt_offsets: (N, 3), float, cuda
    # coords: (N, 3), float32
    # instance_info: (N, 9), float32 tensor (meanxyz, minxyz, maxxyz)
    # instance_labels: (N), long

    gt_offsets = instance_info[:, 0:3] - coords   # (N, 3)
    pt_diff = pt_offsets - gt_offsets   # (N, 3)
    pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)   # (N)
    valid = (instance_labels != -100).float()
    offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)

    gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)   # (N), float
    gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
    pt_offsets_norm = torch.norm(pt_offsets, p=2, dim=1)
    pt_offsets_ = pt_offsets / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
    direction_diff = - (gt_offsets_ * pt_offsets_).sum(-1)   # (N)
    offset_dir_loss = torch.sum(direction_diff * valid) / (torch.sum(valid) + 1e-6)

    if data_dict['epoch'] > prepare_epochs:
        '''score loss'''
        # scores, proposals_idx, proposals_offset, instance_pointnum = data_dict['proposal_scores']
        # scores: (nProposal, 1), float32
        # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
        # proposals_offset: (nProposal + 1), int, cpu
        # instance_pointnum: (total_nInst), int

        instance_pointnum = data_dict['instance_pointnum'].cuda()
        score_feats = data_dict['proposal_feats']
        proposals_idx = data_dict['proposal_idxs']
        proposals_offset = data_dict['proposal_offsets']
        scores = data_dict['proposal_scores']

        ious = pointgroup_ops.get_iou(proposals_idx[:, 1].cuda(), proposals_offset.cuda(), instance_labels, instance_pointnum) # (nProposal, nInstance), float
        gt_ious, gt_instance_idxs = ious.max(1)  # (nProposal) float, long
        gt_scores = get_segmented_scores(gt_ious, CONF.PG.FG_THRESH, CONF.PG.BG_THRESH)

        score_loss = nn.BCELoss(reduction='none').cuda()(torch.sigmoid(scores.view(-1)), gt_scores)
        score_loss = score_loss.mean()

        score_loss = (score_loss, gt_ious.shape[0])

    #total pointgroup module loss
    loss = LOSS_WEIGHTS[0] * semantic_loss[0] + LOSS_WEIGHTS[1] * offset_norm_loss + LOSS_WEIGHTS[2] * offset_dir_loss
    pg_loss = {}
    if data_dict['epoch'] > prepare_epochs:
        loss += (LOSS_WEIGHTS[3] * score_loss[0])
        pg_loss['score_loss'] = score_loss[0]

    pg_loss['semantic_loss'] = semantic_loss[0]
    pg_loss['offset_norm_loss'] = offset_norm_loss
    pg_loss['offset_dir_loss'] = offset_dir_loss

    pg_loss['total_loss'] = loss

    return pg_loss


def compute_pred_gt_objects_assignment(data_dict):
    detected_bbox_corners = data_dict["bbox_corner"]  # batch_size, num_proposals, 8, 3
    # detected_bbox_centers = data_dict["center"] # batch_size, num_proposals, 3
    batch_size, num_proposals, _, _ = detected_bbox_corners.shape

    num_gt_objects = data_dict["gt_box_corner_label"].size()[1]
    repeated_detected_bbox = detected_bbox_corners.unsqueeze(2).expand(-1, -1, num_gt_objects, -1,
                                                                       -1)  # batch_size, num_proposals, num_gt_objects, 8, 3
    repeated_gt_bbox = data_dict["gt_box_corner_label"].unsqueeze(1).expand(-1, num_proposals, -1, -1,
                                                                            -1)  # batch_size, num_proposals, num_gt_objects, 8, 3

    # compute IoU between each detected box and each ground truth box
    ious = box3d_iou_batch_tensor(
        repeated_gt_bbox.reshape(-1, 8, 3),  # batch_size * num_proposals * num_gt_objects, 8, 3
        repeated_detected_bbox.reshape(-1, 8, 3)  # batch_size * num_proposals * num_gt_objects, 8, 3
    ).view(batch_size, num_proposals, num_gt_objects)

    ious, max_ids = ious.max(dim=2)

    return ious, max_ids


def compute_cap_loss(data_dict, config, weights):
    """ Compute cluster caption loss

    Args:
        data_dict: dict (read-only)

    Returns:
        cap_loss, cap_acc
    """

    if CONF.NO_CAPTION: #zero placedholder if no caption
        return torch.zeros(1)[0].cuda(), torch.zeros(1)[0].cuda()

    # unpack
    pred_caps = data_dict["lang_cap"] # (B, num_words - 1, num_vocabs)
    num_words = data_dict["lang_len"].max()
    target_caps = data_dict["lang_ids"][:, 1:num_words] # (B, num_words - 1)
    
    _, _, num_vocabs = pred_caps.shape

    # caption loss
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
    cap_loss = criterion(pred_caps.reshape(-1, num_vocabs), target_caps.reshape(-1))

    # mask out bad boxes
    good_bbox_masks = data_dict["good_bbox_masks"].unsqueeze(1).repeat(1, num_words-1) # (B, num_words - 1)
    good_bbox_masks = good_bbox_masks.reshape(-1) # (B * num_words - 1)
    cap_loss = torch.sum(cap_loss * good_bbox_masks) / (torch.sum(good_bbox_masks) + 1e-6)

    num_good_bbox = data_dict["good_bbox_masks"].sum()
    if num_good_bbox > 0: # only apply loss on the good boxes
        pred_caps = pred_caps[data_dict["good_bbox_masks"]] # num_good_bbox
        target_caps = target_caps[data_dict["good_bbox_masks"]] # num_good_bbox

        # caption acc
        pred_caps = pred_caps.reshape(-1, num_vocabs).argmax(-1) # num_good_bbox * (num_words - 1)
        target_caps = target_caps.reshape(-1) # num_good_bbox * (num_words - 1)
        masks = target_caps != 0
        masked_pred_caps = pred_caps[masks]
        masked_target_caps = target_caps[masks]
        cap_acc = (masked_pred_caps == masked_target_caps).sum().float() / masks.sum().float()
    else: # zero placeholder if there is no good box
        cap_acc = torch.zeros(1)[0].cuda()
    
    return cap_loss, cap_acc

def radian_to_label(radians, num_bins=6):
    """
        convert radians to labels

        Arguments:
            radians: a tensor representing the rotation radians, (batch_size)
            radians: a binary tensor representing the valid masks, (batch_size)
            num_bins: number of bins for discretizing the rotation degrees

        Return:
            labels: a long tensor representing the discretized rotation degree classes, (batch_size)
    """

    boundaries = torch.arange(np.pi / num_bins, np.pi-1e-8, np.pi / num_bins).cuda()
    labels = torch.bucketize(radians, boundaries)

    return labels

def compute_node_orientation_loss(data_dict, num_bins=6):
    object_assignment = data_dict["object_assignment"]
    edge_indices = data_dict["edge_index"]
    edge_preds = data_dict["edge_orientations"]
    num_sources = data_dict["num_edge_source"]
    num_targets = data_dict["num_edge_target"]
    batch_size, num_proposals = object_assignment.shape

    object_rotation_matrices = torch.gather(
        data_dict["scene_object_rotations"], 
        1, 
        object_assignment.view(batch_size, num_proposals, 1, 1).repeat(1, 1, 3, 3)
    ) # batch_size, num_proposals, 3, 3
    object_rotation_masks = torch.gather(
        data_dict["scene_object_rotation_masks"], 
        1, 
        object_assignment
    ) # batch_size, num_proposals
    
    preds = []
    labels = []
    masks = []
    for batch_id in range(batch_size):
        batch_rotations = object_rotation_matrices[batch_id] # num_proposals, 3, 3
        batch_rotation_masks = object_rotation_masks[batch_id] # num_proposals

        batch_num_sources = num_sources[batch_id]
        batch_num_targets = num_targets[batch_id]
        batch_edge_indices = edge_indices[batch_id, :batch_num_sources * batch_num_targets]

        source_indices = edge_indices[batch_id, 0, :batch_num_sources*batch_num_targets].long()
        target_indices = edge_indices[batch_id, 1, :batch_num_sources*batch_num_targets].long()

        source_rot = torch.index_select(batch_rotations, 0, source_indices)
        target_rot = torch.index_select(batch_rotations, 0, target_indices)

        relative_rot = torch.matmul(source_rot, target_rot.transpose(2, 1))
        relative_rot = torch.acos(torch.clamp(0.5 * (torch.diagonal(relative_rot, dim1=-2, dim2=-1).sum(-1) - 1), -1, 1))
        assert torch.isfinite(relative_rot).sum() == source_indices.shape[0]

        source_masks = torch.index_select(batch_rotation_masks, 0, source_indices)
        target_masks = torch.index_select(batch_rotation_masks, 0, target_indices)
        batch_edge_masks = source_masks * target_masks
        
        batch_edge_labels = radian_to_label(relative_rot, num_bins)
        batch_edge_preds = edge_preds[batch_id, :batch_num_sources * batch_num_targets]

        preds.append(batch_edge_preds)
        labels.append(batch_edge_labels)
        masks.append(batch_edge_masks)

    # aggregate
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    masks = torch.cat(masks, dim=0)

    criterion = nn.CrossEntropyLoss(reduction="none")
    loss = criterion(preds, labels)
    loss = (loss * masks).sum() / (masks.sum() + 1e-8)

    preds = preds.argmax(-1)
    acc = (preds[masks==1] == labels[masks==1]).sum().float() / (masks.sum().float() + 1e-8)

    return loss, acc

def compute_node_distance_loss(data_dict):
    gt_center = data_dict["center_label"][:,:,0:3]
    object_assignment = data_dict["object_assignment"]
    
    gt_center = torch.gather(gt_center, 1, object_assignment.unsqueeze(-1).repeat(1, 1, 3))
    batch_size, _, _ = gt_center.shape

    edge_indices = data_dict["edge_index"]
    edge_preds = data_dict["edge_distances"]
    num_sources = data_dict["num_edge_source"]
    num_targets = data_dict["num_edge_target"]

    preds = []
    labels = []
    for batch_id in range(batch_size):
        batch_gt_center = gt_center[batch_id]

        batch_num_sources = num_sources[batch_id]
        batch_num_targets = num_targets[batch_id]
        batch_edge_indices = edge_indices[batch_id, :batch_num_sources * batch_num_targets]

        source_indices = edge_indices[batch_id, 0, :batch_num_sources*batch_num_targets].long()
        target_indices = edge_indices[batch_id, 1, :batch_num_sources*batch_num_targets].long()

        source_centers = torch.index_select(batch_gt_center, 0, source_indices)
        target_centers = torch.index_select(batch_gt_center, 0, target_indices)

        batch_edge_labels = torch.norm(source_centers - target_centers, dim=1)
        batch_edge_preds = edge_preds[batch_id, :batch_num_sources * batch_num_targets]

        preds.append(batch_edge_preds)
        labels.append(batch_edge_labels)

    # aggregate
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)

    criterion = nn.MSELoss()
    loss = criterion(preds, labels)

    return loss

def compute_object_cls_loss(data_dict, weights):
    """ Compute object classification loss

    Args:
        data_dict: dict (read-only)

    Returns:
        cls_loss, cls_acc
    """

    # unpack
    preds = data_dict["enc_preds"] # (B, num_cls)
    targets = data_dict["object_cat"] # (B,)
    
    # classification loss
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).cuda())
    cls_loss = criterion(preds, targets)

    # classification acc
    preds = preds.argmax(-1) # (B,)
    cls_acc = (preds == targets).sum().float() / targets.shape[0]

    return cls_loss, cls_acc

def get_scene_cap_loss(data_dict, device, config, weights, prepare_epochs,
    detection=True, caption=True, orientation=False, distance=False, num_bins=CONF.TRAIN.NUM_BINS,):
    """ Loss functions

    Args:
        data_dict: dict
        config: dataset config instance
        reference: flag (False/True)
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    """

    # Unpack loss of pgmodule parts
    pg_loss = compute_pg_loss(data_dict, config, prepare_epochs)
    semantic_loss = pg_loss["semantic_loss"]
    offset_norm_loss = pg_loss["offset_norm_loss"]
    offset_dir_loss = pg_loss["offset_dir_loss"]

    pg_total_loss = pg_loss["total_loss"]

    if detection:
        data_dict["pg_semantic_loss"] = semantic_loss
        data_dict["pg_offset_norm_loss"] = offset_norm_loss
        data_dict["pg_offset_dir_loss"] = offset_dir_loss
        data_dict["pg_total_loss"] = pg_total_loss

        if data_dict['epoch'] > prepare_epochs:
            score_loss = pg_loss["score_loss"]
            data_dict["pg_score_loss"] = score_loss

            max_ious, max_ids = compute_pred_gt_objects_assignment(data_dict)
            data_dict["object_assignment"] = max_ids
    else:
        pass
        # data_dict["vote_loss"] = torch.zeros(1)[0].to(device)
        # data_dict["objectness_loss"] = torch.zeros(1)[0].to(device)
        # data_dict["center_loss"] = torch.zeros(1)[0].to(device)
        # data_dict["heading_cls_loss"] = torch.zeros(1)[0].to(device)
        # data_dict["heading_reg_loss"] = torch.zeros(1)[0].to(device)
        # data_dict["size_cls_loss"] = torch.zeros(1)[0].to(device)
        # data_dict["size_reg_loss"] = torch.zeros(1)[0].to(device)
        # data_dict["sem_cls_loss"] = torch.zeros(1)[0].to(device)
        # data_dict["box_loss"] = torch.zeros(1)[0].to(device)

    if caption and data_dict['epoch'] > prepare_epochs:
        cap_loss, cap_acc = compute_cap_loss(data_dict, config, weights)

        # store
        data_dict["cap_loss"] = cap_loss
        data_dict["cap_acc"] = cap_acc
    else:
        # store
        data_dict["cap_loss"] = torch.zeros(1)[0].to(device)
        data_dict["cap_acc"] = torch.zeros(1)[0].to(device)
        data_dict["pred_ious"] =  torch.zeros(1)[0].to(device)

    if orientation and data_dict['epoch'] > prepare_epochs:
        ori_loss, ori_acc = compute_node_orientation_loss(data_dict, num_bins)

        # store
        data_dict["ori_loss"] = ori_loss
        data_dict["ori_acc"] = ori_acc
    else:
        # store
        data_dict["ori_loss"] = torch.zeros(1)[0].to(device)
        data_dict["ori_acc"] = torch.zeros(1)[0].to(device)

    if distance and data_dict['epoch'] > prepare_epochs:
        dist_loss = compute_node_distance_loss(data_dict)

        # store
        data_dict["dist_loss"] = dist_loss
    else:
        # store
        data_dict["dist_loss"] = torch.zeros(1)[0].to(device)

    # Final loss function
    # loss = data_dict["vote_loss"] + 0.5*data_dict["objectness_loss"] + data_dict["box_loss"] + 0.1*data_dict["sem_cls_loss"] + data_dict["cap_loss"]

    if detection:
        loss = data_dict["pg_total_loss"]
        # loss *= 10 # amplify
        if caption:
            loss += data_dict["cap_loss"]
        if orientation:
            loss += 0.1*data_dict["ori_loss"]
        if distance:
            loss += 0.1*data_dict["dist_loss"]
            # loss += data_dict["dist_loss"]
    else:
        loss = data_dict["cap_loss"]
        if orientation:
            loss += 0.1*data_dict["ori_loss"]
        if distance:
            loss += 0.1*data_dict["dist_loss"]

    data_dict["loss"] = loss

    return data_dict

# def get_object_cap_loss(data_dict, config, weights, classify=True, caption=True):
#     """ Loss functions

#     Args:
#         data_dict: dict
#         config: dataset config instance
#         reference: flag (False/True)
#     Returns:
#         loss: pytorch scalar tensor
#         data_dict: dict
#     """

#     if classify:
#         cls_loss, cls_acc = compute_object_cls_loss(data_dict, weights)

#         data_dict["cls_loss"] = cls_loss
#         data_dict["cls_acc"] = cls_acc
#     else:
#         data_dict["cls_loss"] = torch.zeros(1)[0].cuda()
#         data_dict["cls_acc"] = torch.zeros(1)[0].cuda()

#     if caption:
#         cap_loss, cap_acc = compute_cap_loss(data_dict, config, weights)

#         # store
#         data_dict["cap_loss"] = cap_loss
#         data_dict["cap_acc"] = cap_acc
#     else:
#         # store
#         data_dict["cap_loss"] = torch.zeros(1)[0].cuda()
#         data_dict["cap_acc"] = torch.zeros(1)[0].cuda()

#     # Final loss function
#     loss = data_dict["cls_loss"] + data_dict["cap_loss"]

#     # loss *= 10 # amplify

#     data_dict["loss"] = loss

#     return data_dict

def get_segmented_scores(scores, fg_thresh=1.0, bg_thresh=0.0):
    '''
    :param scores: (N), float, 0~1
    :return: segmented_scores: (N), float 0~1, >fg_thresh: 1, <bg_thresh: 0, mid: linear
    '''
    fg_mask = scores > fg_thresh
    bg_mask = scores < bg_thresh
    interval_mask = (fg_mask == 0) & (bg_mask == 0)

    segmented_scores = (fg_mask > 0).float()
    k = 1 / (fg_thresh - bg_thresh)
    b = bg_thresh / (bg_thresh - fg_thresh)
    segmented_scores[interval_mask] = scores[interval_mask] * k + b

    return segmented_scores
