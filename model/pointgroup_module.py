'''
PointGroup
Written by Li Jiang

Modified by Yifeng Dong and Daniel-Jordi Regenbrecht
'''

import torch
import torch.nn as nn
import spconv.pytorch as spconv
from spconv.pytorch.modules import SparseModule
import functools
from collections import OrderedDict
import sys
from util import pg_utils

sys.path.append('../../')

from lib.pointgroup_ops.functions import pointgroup_ops

class ResidualBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(
                nn.Identity()
            )
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False)
            )

        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
            norm_fn(out_channels),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)

        output = self.conv_branch(input)
        output = output.replace_feature(output.features + self.i_branch(identity).features)

        return output


class VGGBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        self.conv_layers = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        return self.conv_layers(input)


class UBlock(nn.Module):
    def __init__(self, nPlanes, norm_fn, block_reps, block, indice_key_id=1):

        super().__init__()

        self.nPlanes = nPlanes

        if len(self.nPlanes) % 2 != 1:
            raise ValueError("Has to be odd number of UBlock layers")

        blocks = {'block{}'.format(i): block(nPlanes[0], nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id)) for i in range(block_reps)}
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(nPlanes) > 2:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]),
                nn.ReLU(),
                spconv.SparseConv3d(nPlanes[0], nPlanes[1], kernel_size=2, stride=2, bias=False, indice_key='spconv{}'.format(indice_key_id))
            )

            self.u = UBlock(nPlanes[1:-1], norm_fn, block_reps, block, indice_key_id=indice_key_id+1)

            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[-2]),
                nn.ReLU(),
                spconv.SparseInverseConv3d(nPlanes[-2], nPlanes[-1], kernel_size=2, bias=False, indice_key='spconv{}'.format(indice_key_id))
            )

            blocks_tail = {}
            for i in range(block_reps):
                if i == 0:
                    blocks_tail['block{}'.format(i)] = block(nPlanes[0] + nPlanes[-1], nPlanes[-1], norm_fn, indice_key='subm{}'.format(indice_key_id))
                else:
                    blocks_tail['block{}'.format(i)] = block(nPlanes[-1], nPlanes[-1], norm_fn, indice_key='subm{}'.format(indice_key_id))
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, input):
        output = self.blocks(input)  # nPlanes[0] -> nPlanes[0]
        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape, output.batch_size)  # nPlanes[0]

        if len(self.nPlanes) > 2:
            output_decoder = self.conv(output)  # nPlanes[0] -> nPlanes[1]
            output_decoder = self.u(output_decoder)  # nPlanes[1] -> nPlanes[-2]
            output_decoder = self.deconv(output_decoder)  # nPlanes[-2] -> nPlanes[-1]

            output = output.replace_feature(torch.cat((identity.features, output_decoder.features), dim=1))  # nPlanes[0] + nPlanes[-1]

            output = self.blocks_tail(output)  # nPlanes[0] + nPlanes[-1] -> nPlanes[-1]

        return output


class PointGroupModule(nn.Module):
    r"""
        PointGroup detection module for isntance segmantation and cluster feature extraction.
    """
    def __init__(self, input_c=3,
                 m=32,
                 ublock_layers=(1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 3, 3),
                 classes=20,
                 cluster_npoint_thre=50,
                 cluster_radius=0.03,
                 cluster_meanActive=50,
                 cluster_shift_meanActive=300,
                 score_scale=50,
                 score_fullscale=14,
                 score_mode=4,
                 prepare_epochs=128,
                 block_reps=2,
                 block_residual=True,
                 pretrain_path=None,
                 pretrain_module=(),
                 fix_module=()):
        super().__init__()

        input_c = input_c
        m = m
        classes = classes
        block_reps = block_reps
        block_residual = block_residual

        self.cluster_radius = cluster_radius
        self.cluster_meanActive = cluster_meanActive
        self.cluster_shift_meanActive = cluster_shift_meanActive
        self.cluster_npoint_thre = cluster_npoint_thre

        self.score_scale = score_scale
        self.score_fullscale = score_fullscale
        self.mode = score_mode

        self.prepare_epochs = prepare_epochs

        self.pretrain_path = pretrain_path
        self.pretrain_module = pretrain_module
        self.fix_module = fix_module
        self.ublock_layers_conf = ublock_layers

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        if block_residual:
            block = ResidualBlock
        else:
            block = VGGBlock

        #backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(input_c, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
        )

        self.unet = UBlock([k * m for k in ublock_layers], norm_fn, block_reps, block, indice_key_id=1)

        self.output_layer = spconv.SparseSequential(
            norm_fn(ublock_layers[-1]*m),
            nn.ReLU()
        )

        #semantic segmentation
        self.linear = nn.Linear(ublock_layers[-1]*m, classes) # bias(default): True

        #offset
        self.offset = nn.Sequential(
            nn.Linear(ublock_layers[-1]*m, ublock_layers[-1]*m, bias=True),
            norm_fn(ublock_layers[-1]*m),
            nn.ReLU()
        )
        self.offset_linear = nn.Linear(ublock_layers[-1]*m, 3, bias=True)

        #### score branch
        self.score_unet = UBlock([ublock_layers[-1]*m, 2*ublock_layers[-1]*m, ublock_layers[-1]*m], norm_fn, 2, block, indice_key_id=1)
        self.score_outputlayer = spconv.SparseSequential(
            norm_fn(ublock_layers[-1]*m),
            nn.ReLU()
        )
        self.score_linear = nn.Linear(ublock_layers[-1]*m, 1)

        self.apply(self.set_bn_init)

        #### fix parameters
        module_map = {'input_conv': self.input_conv, 'unet': self.unet, 'output_layer': self.output_layer,
                      'linear': self.linear, 'offset': self.offset, 'offset_linear': self.offset_linear,
                      'score_unet': self.score_unet, 'score_outputlayer': self.score_outputlayer,
                      'score_linear': self.score_linear}

        for m in self.fix_module:
            mod = module_map[m]
            for param in mod.parameters():
                param.requires_grad = False

        ### load pretrain weights
        if self.pretrain_path is not None:
            pretrain_dict = torch.load(self.pretrain_path)
            for m in self.pretrain_module:
                print(
                    "Load pretrained " + m + ": %d/%d" % pg_utils.load_model_param(module_map[m], pretrain_dict, prefix=m))

    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    def forward(self, data_dict):
        r"""
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """

        coords = data_dict['locs']  # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = data_dict['voxel_locs']  # (M, 1 + 3), long, cuda
        p2v_map = data_dict['p2v_map']  # (N), int, cuda
        v2p_map = data_dict['v2p_map']  # (M, 1 + maxActive), int, cuda

        coords_float = data_dict['locs_float']  # (N, 3), float32, cuda
        feats = data_dict['feats']  # (N, C), float32, cuda
        # labels = data_dict['labels'].cuda()  # (N), long, cuda
        # instance_labels = data_dict['instance_labels'].cuda()  # (N), long, cuda, 0~total_nInst, -100
        #
        # instance_info = data_dict['instance_info'].cuda()  # (N, 9), float32, cuda, (meanxyz, minxyz, maxxyz)
        # instance_pointnum = data_dict['instance_pointnum'].cuda()  # (total_nInst), int, cuda

        batch_offsets = data_dict['batch_offsets']  # (B + 1), int, cuda

        spatial_shape = data_dict['spatial_shape']

        # if cfg.use_coords:
        #     feats = torch.cat((feats, coords_float), 1)
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map, 4)  # (M, C), float, cuda

        input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size=len(batch_offsets)-1)

        # ret = model(input_, p2v_map, coords_float, coords[:, 0].int(), batch_offsets, epoch)

        input = input_
        input_map = p2v_map
        batch_idxs = coords[:, 0].int()
        coords = coords_float

        batch_offsets = batch_offsets
        epoch = data_dict['epoch']

        #pass through backbone
        output = self.input_conv(input)
        output = self.unet(output)
        output = self.output_layer(output)
        output_feats = output.features[input_map]

        #semantic segmentation
        semantic_scores = self.linear(output_feats)
        semantic_preds = semantic_scores.max(dim=1)[1]

        data_dict['semantic_scores'] = semantic_scores
        data_dict['semantic_preds'] = semantic_preds
        
        #offset
        pt_offset_feats = self.offset(output_feats)
        pt_offsets = self.offset_linear(pt_offset_feats)

        data_dict['pt_offsets'] = pt_offsets

        if epoch > self.prepare_epochs:
            #clustering
            object_idxs = torch.nonzero(semantic_preds > 1).view(-1)

            batch_idxs_ = batch_idxs[object_idxs]
            batch_offsets_ = pg_utils.get_batch_offsets(batch_idxs_, input.batch_size)
            coords_ = coords[object_idxs]
            pt_offsets_ = pt_offsets[object_idxs]
            semantic_preds_cpu = semantic_preds[object_idxs].int().cpu()

            idx_shift, start_len_shift = pointgroup_ops.ballquery_batch_p(coords_ + pt_offsets_, batch_idxs_, batch_offsets_, self.cluster_radius, self.cluster_shift_meanActive)
            proposals_idx_shift, proposals_offset_shift = pointgroup_ops.bfs_cluster(semantic_preds_cpu, idx_shift.cpu(), start_len_shift.cpu(), self.cluster_npoint_thre)
            proposals_idx_shift[:, 1] = object_idxs[proposals_idx_shift[:, 1].long()].int()

            idx, start_len = pointgroup_ops.ballquery_batch_p(coords_, batch_idxs_, batch_offsets_, self.cluster_radius, self.cluster_meanActive)
            proposals_idx, proposals_offset = pointgroup_ops.bfs_cluster(semantic_preds_cpu, idx.cpu(), start_len.cpu(), self.cluster_npoint_thre)
            proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()

            proposals_idx_shift[:, 0] += (proposals_offset.size(0) - 1)
            proposals_offset_shift += proposals_offset[-1]
            proposals_idx = torch.cat((proposals_idx, proposals_idx_shift), dim=0)
            proposals_offset = torch.cat((proposals_offset, proposals_offset_shift[1:]))

            #proposal voxelization
            input_feats, inp_map = self.clusters_voxelization(proposals_idx, proposals_offset, output_feats, coords, self.score_fullscale, self.score_scale, self.mode)

            #### score
            score = self.score_unet(input_feats)
            score = self.score_outputlayer(score)
            score_feats = score.features[inp_map.long()] # (sumNPoint, C)
            score_feats = pointgroup_ops.roipool(score_feats, proposals_offset.cuda())  # (nProposal, C)
            scores = self.score_linear(score_feats)  # (nProposal, 1)

            proposal_feats = input_feats.features[inp_map.long()]
            proposal_feats = pointgroup_ops.roipool(proposal_feats, proposals_offset.cuda())

            data_dict['proposal_feats'] = proposal_feats  # (nProposal, C)
            data_dict['proposal_idxs'] = proposals_idx  # (sumNPoint, 2) [:, 0] is id of cluster, [:, 1] is id of point
            data_dict['proposal_offsets'] = proposals_offset  # (nProposal,) Where in proposals_idxs the next cluster begins
            data_dict['proposal_scores'] = scores  # (nProposal, 1)

        return data_dict

    def clusters_voxelization(self, clusters_idx, clusters_offset, feats, coords, fullscale, scale, mode):
        '''
        :param clusters_idx: (SumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N, cpu
        :param clusters_offset: (nCluster + 1), int, cpu
        :param feats: (N, C), float, cuda
        :param coords: (N, 3), float, cuda
        :return:
        '''
        c_idxs = clusters_idx[:, 1].cuda()
        clusters_feats = feats[c_idxs.long()]
        clusters_coords = coords[c_idxs.long()]

        clusters_coords_mean = pointgroup_ops.sec_mean(clusters_coords,
                                                       clusters_offset.cuda())  # (nCluster, 3), float
        clusters_coords_mean = torch.index_select(clusters_coords_mean, 0,
                                                  clusters_idx[:, 0].cuda().long())  # (sumNPoint, 3), float
        clusters_coords -= clusters_coords_mean

        clusters_coords_min = pointgroup_ops.sec_min(clusters_coords,
                                                     clusters_offset.cuda())  # (nCluster, 3), float
        clusters_coords_max = pointgroup_ops.sec_max(clusters_coords,
                                                     clusters_offset.cuda())  # (nCluster, 3), float

        clusters_scale = 1 / ((clusters_coords_max - clusters_coords_min) / fullscale).max(1)[
            0] - 0.01  # (nCluster), float
        clusters_scale = torch.clamp(clusters_scale, min=None, max=scale)

        min_xyz = clusters_coords_min * clusters_scale.unsqueeze(-1)  # (nCluster, 3), float
        max_xyz = clusters_coords_max * clusters_scale.unsqueeze(-1)

        clusters_scale = torch.index_select(clusters_scale, 0, clusters_idx[:, 0].cuda().long())

        clusters_coords = clusters_coords * clusters_scale.unsqueeze(-1)

        range = max_xyz - min_xyz
        offset = - min_xyz + torch.clamp(fullscale - range - 0.001, min=0) * torch.rand(3).cuda() + torch.clamp(
            fullscale - range + 0.001, max=0) * torch.rand(3).cuda()
        offset = torch.index_select(offset, 0, clusters_idx[:, 0].cuda().long())
        clusters_coords += offset
        assert clusters_coords.shape.numel() == ((clusters_coords >= 0) * (clusters_coords < fullscale)).sum()

        clusters_coords = clusters_coords.long()
        clusters_coords = torch.cat([clusters_idx[:, 0].view(-1, 1).long(), clusters_coords.cpu()],
                                    1)  # (sumNPoint, 1 + 3)

        out_coords, inp_map, out_map = pointgroup_ops.voxelization_idx(clusters_coords,
                                                                       int(clusters_idx[-1, 0]) + 1, mode)
        # output_coords: M * (1 + 3) long
        # input_map: sumNPoint int
        # output_map: M * (maxActive + 1) int

        out_feats = pointgroup_ops.voxelization(clusters_feats, out_map.cuda(), mode)  # (M, C), float, cuda

        spatial_shape = [fullscale] * 3
        voxelization_feats = spconv.SparseConvTensor(out_feats, out_coords.int().cuda(), spatial_shape,
                                                     int(clusters_idx[-1, 0]) + 1)

        return voxelization_feats, inp_map


def model_fn_decorator(test=False, ignore_label = -100, use_coords = True, mode = 4, prepare_epochs = 128, loss_weight = [1.0, 1.0, 1.0, 1.0], batch_size = 4, fg_thresh = 0.75, bg_thresh = 0.25):
    #### criterion

    semantic_criterion = nn.CrossEntropyLoss(ignore_index=ignore_label).cuda()
    score_criterion = nn.BCELoss(reduction='none').cuda()

    def test_model_fn(data_dict, model, epoch):
        coords = data_dict['locs'].cuda()              # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = data_dict['voxel_locs'].cuda()  # (M, 1 + 3), long, cuda
        p2v_map = data_dict['p2v_map'].cuda()          # (N), int, cuda
        v2p_map = data_dict['v2p_map'].cuda()          # (M, 1 + maxActive), int, cuda

        coords_float = data_dict['locs_float'].cuda()  # (N, 3), float32, cuda
        feats = data_dict['feats'].cuda()              # (N, C), float32, cuda

        batch_offsets = data_dict['batch_offsets'].cuda()    # (B + 1), int, cuda

        spatial_shape = data_dict['spatial_shape']

        if use_coords:
            feats = torch.cat((feats, coords_float), 1)
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map, mode)  # (M, C), float, cuda

        input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)

        ret = model(input_, p2v_map, coords_float, coords[:, 0].int(), batch_offsets, epoch)
        semantic_scores = ret['semantic_scores']  # (N, nClass) float32, cuda
        pt_offsets = ret['pt_offsets']            # (N, 3), float32, cuda
        if (epoch > prepare_epochs):
            score_feats = data_dict['proposal_feats']
            proposals_idx = data_dict['proposal_idxs']
            proposals_offset = data_dict['proposal_offsets']
            scores = data_dict['proposal_scores']
            # scores, proposals_idx, proposals_offset = ret['proposal_scores']

        ##### preds
        with torch.no_grad():
            preds = {}
            preds['semantic'] = semantic_scores
            preds['pt_offsets'] = pt_offsets
            if (epoch > prepare_epochs):
                preds['score'] = scores
                preds['proposals'] = (proposals_idx, proposals_offset)

        return preds


    def model_fn(data_dict, model: PointGroupModule, epoch):
        ##### prepare input and forward
        # data_dict {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
        # 'locs_float': locs_float, 'feats': feats, 'labels': labels, 'instance_labels': instance_labels,
        # 'instance_info': instance_infos, 'instance_pointnum': instance_pointnum,
        # 'id': tbl, 'offsets': batch_offsets, 'spatial_shape': spatial_shape}
        coords = data_dict['locs'].cuda()                          # (N, 1 + 3), long, cuda, dimension 0 for batch_idx

        coords_float = data_dict['locs_float'].cuda()              # (N, 3), float32, cuda
        labels = data_dict['pg_semantic_labels'].cuda()                        # (N), long, cuda
        instance_labels = data_dict['pg_instance_labels'].cuda()      # (N), long, cuda, 0~total_nInst, -100

        instance_info = data_dict['instance_info'].cuda()          # (N, 9), float32, cuda, (meanxyz, minxyz, maxxyz)
        instance_pointnum = data_dict['instance_pointnum'].cuda()  # (total_nInst), int, cuda

        ret = model(data_dict)
        semantic_scores = ret['semantic_scores'] # (N, nClass) float32, cuda
        pt_offsets = ret['pt_offsets']           # (N, 3), float32, cuda
        if epoch > prepare_epochs:
            # scores, proposals_idx, proposals_offset = ret['proposal_scores']
            score_feats = data_dict['proposal_feats']
            proposals_idx = data_dict['proposal_idxs']
            proposals_offset = data_dict['proposal_offsets']
            scores = data_dict['proposal_scores']
            # scores: (nProposal, 1) float, cuda
            # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int, cpu

        loss_inp = {}
        loss_inp['semantic_scores'] = (semantic_scores, labels)
        loss_inp['pt_offsets'] = (pt_offsets, coords_float, instance_info, instance_labels)
        if epoch > prepare_epochs:
            loss_inp['proposal_scores'] = (scores, proposals_idx, proposals_offset, instance_pointnum)

        loss, loss_out, infos = loss_fn(loss_inp, epoch)

        ##### accuracy / visual_dict / meter_dict
        with torch.no_grad():
            preds = {}
            preds['semantic'] = semantic_scores
            preds['pt_offsets'] = pt_offsets
            if(epoch > prepare_epochs):
                preds['score'] = scores
                preds['proposals'] = (proposals_idx, proposals_offset)

            visual_dict = {}
            visual_dict['loss'] = loss
            for k, v in loss_out.items():
                visual_dict[k] = v[0]

            meter_dict = {}
            meter_dict['loss'] = (loss.item(), coords.shape[0])
            for k, v in loss_out.items():
                meter_dict[k] = (float(v[0]), v[1])

        return loss, preds, visual_dict, meter_dict


    def loss_fn(loss_inp, epoch):

        loss_out = {}
        infos = {}

        '''semantic loss'''
        semantic_scores, semantic_labels = loss_inp['semantic_scores']
        # semantic_scores: (N, nClass), float32, cuda
        # semantic_labels: (N), long, cuda

        semantic_loss = semantic_criterion(semantic_scores, semantic_labels)
        loss_out['semantic_loss'] = (semantic_loss, semantic_scores.shape[0])

        '''offset loss'''
        pt_offsets, coords, instance_info, instance_labels = loss_inp['pt_offsets']
        # pt_offsets: (N, 3), float, cuda
        # coords: (N, 3), float32
        # instance_info: (N, 9), float32 tensor (meanxyz, minxyz, maxxyz)
        # instance_labels: (N), long

        gt_offsets = instance_info[:, 0:3] - coords   # (N, 3)
        pt_diff = pt_offsets - gt_offsets   # (N, 3)
        pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)   # (N)
        valid = (instance_labels != ignore_label).float()
        offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)

        gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)   # (N), float
        gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
        pt_offsets_norm = torch.norm(pt_offsets, p=2, dim=1)
        pt_offsets_ = pt_offsets / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
        direction_diff = - (gt_offsets_ * pt_offsets_).sum(-1)   # (N)
        offset_dir_loss = torch.sum(direction_diff * valid) / (torch.sum(valid) + 1e-6)

        loss_out['offset_norm_loss'] = (offset_norm_loss, valid.sum())
        loss_out['offset_dir_loss'] = (offset_dir_loss, valid.sum())

        if (epoch > prepare_epochs):
            '''score loss'''
            scores, proposals_idx, proposals_offset, instance_pointnum = loss_inp['proposal_scores']
            # scores: (nProposal, 1), float32
            # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int, cpu
            # instance_pointnum: (total_nInst), int

            ious = pointgroup_ops.get_iou(proposals_idx[:, 1].cuda(), proposals_offset.cuda(), instance_labels, instance_pointnum) # (nProposal, nInstance), float
            gt_ious, gt_instance_idxs = ious.max(1)  # (nProposal) float, long
            gt_scores = get_segmented_scores(gt_ious, fg_thresh, bg_thresh)

            score_loss = score_criterion(torch.sigmoid(scores.view(-1)), gt_scores)
            score_loss = score_loss.mean()

            loss_out['score_loss'] = (score_loss, gt_ious.shape[0])

        '''total loss'''
        loss = loss_weight[0] * semantic_loss + loss_weight[1] * offset_norm_loss + loss_weight[2] * offset_dir_loss
        if(epoch > prepare_epochs):
            loss += (loss_weight[3] * score_loss)

        return loss, loss_out, infos


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


    if test:
        fn = test_model_fn
    else:
        fn = model_fn
    return fn