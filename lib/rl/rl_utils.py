# HACK ignore warnings
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import json
import torch
import numpy as np

from tqdm import tqdm
from copy import deepcopy
from shutil import copyfile
from tensorboardX import SummaryWriter

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder

import lib.capeval.bleu.bleu as capblue
import lib.capeval.cider.cider as capcider
import lib.capeval.rouge.rouge as caprouge
import lib.capeval.spice.spice as capspice

from data.scannet.model_util_scannet import ScannetDatasetConfig
from config.scan2cap_config import CONF
from lib.ap_helper import parse_predictions
from lib.eval_helper import prepare_corpus, decode_caption
from util.box_util import box3d_iou_batch_tensor



SCANNET_MESH = os.path.join(CONF.PATH.AXIS_ALIGNED_MESH, "{}", "axis_aligned_scene.ply")
SCANNET_AGGR = os.path.join(CONF.PATH.SCANNET_SCANS, "{}/{}_vh_clean.aggregation.json") # scene_id, scene_id

SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.SCANREFER, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.SCANREFER, "ScanRefer_filtered_val.json")))
SCANREFER_ORGANIZED = json.load(open(os.path.join(CONF.PATH.SCANREFER, "ScanRefer_filtered_organized.json")))

DC = ScannetDatasetConfig()


class ReinforceHelper(object):
    def __init__(self, dataset, raw_data, max_len, organized, decay_rate=0.9, score_weights=[0,0,0,0,0,1.0,1.0]):
        self.dataset = dataset
        self.avg_cider = -1
        self.avg_spice = -1
        self.decay_rate = decay_rate
        self.cider = capcider.Cider()
        self.spice = capspice.Spice()
        self.bleu = capblue.Bleu(4)
        self.rouge = caprouge.Rouge()
        self.score_weights = score_weights # BLEU_1, BLEU_2, BLEU_3, BLEU_4,, ROUGE, CIDER, SPICE
        self.corpus = prepare_corpus(raw_data, max_len)
        self.organized = organized



    def _compute_scores(self, corpus, candidates):

        _, bleu = self.bleu.compute_score(corpus, candidates)
        _, rouge = self.rouge.compute_score(corpus, candidates)
        _, cider = self.cider.compute_score(candidates, corpus)
        _, spice = self.spice.compute_score(corpus, candidates)
        
        return bleu, rouge, cider, spice
        
    
    def _update_running_means(self, new_cider, new_spice, return_cur=True):
        if self.avg_cider == -1:
            self.avg_cider = new_cider
            self.avg_spice = new_spice
        else:
            self.avg_cider = self.avg_cider * self.decay_rate + new_cider * (1 - self.decay_rate)
            self.avg_spice = self.avg_spice * self.decay_rate + new_spice * (1 - self.decay_rate)
        if return_cur == True:
            return self.avg_cider, self.avg_spice

    def get_scores(self, data_dict, conf):
        captions = data_dict["actions"]  # batch_size, num_proposals, max_len - 1
        scores = np.zeros((captions.shape[0], captions.shape[1]))
        dataset_ids = data_dict["dataset_idx"]
        batch_size, num_proposals, _ = captions.shape

        # post-process
        # config
        POST_DICT = {
            "remove_empty_box": False,
            "use_3d_nms": True,
            "nms_iou": 0.25,
            "use_old_type_nms": False,
            "cls_nms": True,
            "per_class_proposal": True,
            "conf_thresh": 0.05,
            "dataset_config": DC
        }

        # nms mask
        _ = parse_predictions(data_dict, POST_DICT)
        nms_masks = torch.LongTensor(data_dict["pred_mask"]).cuda()

        # objectness mask
        obj_masks = data_dict["bbox_mask"]

        # final mask
        nms_masks = nms_masks * obj_masks

        # pick out object ids of detected objects
        detected_object_ids = torch.gather(data_dict["scene_object_ids"], 1, data_dict["object_assignment"])

        # bbox corners
        assigned_target_bbox_corners = torch.gather(
            data_dict["gt_box_corner_label"],
            1,
            data_dict["object_assignment"].view(batch_size, num_proposals, 1, 1).repeat(1, 1, 8, 3)
        ) # batch_size, num_proposals, 8, 3
        detected_bbox_corners = data_dict["bbox_corner"]  # batch_size, num_proposals, 8, 3
        # detected_bbox_centers = data_dict["center"] # batch_size, num_proposals, 3

        num_gt_objects = data_dict["gt_box_corner_label"].size()[1]
        repeated_detected_bbox = detected_bbox_corners.unsqueeze(2).expand(-1, -1, num_gt_objects, -1,
                                                                            -1)  # batch_size, num_proposals, num_gt_objects, 8, 3
        repeated_gt_bbox = data_dict["gt_box_corner_label"].unsqueeze(1).expand(-1, num_proposals, -1, -1,
                                                                                -1)  # batch_size, num_proposals, num_gt_objects, 8, 3

        # compute IoU between each detected box and each ground truth box
        ious = box3d_iou_batch_tensor(
            assigned_target_bbox_corners.view(-1, 8, 3), # batch_size * num_proposals, 8, 3
            detected_bbox_corners.view(-1, 8, 3) # batch_size * num_proposals, 8, 3
        ).view(batch_size, num_proposals)

        # find good boxes (IoU > threshold)
        good_bbox_masks = ious > conf.TRAIN.MIN_IOU_THRESHOLD  # batch_size, num_proposals

        candidates = {}
        corpus = {}
        candidate_ids = []
        running_id = 0

        # dump generated captions
        object_attn_masks = {}
        for batch_id in range(batch_size):
            dataset_idx = dataset_ids[batch_id].item()
            scene_id = self.dataset.scanrefer[dataset_idx]["scene_id"]
            object_attn_masks[scene_id] = np.zeros((num_proposals, num_proposals))
            for prop_id in range(num_proposals):
                if nms_masks[batch_id, prop_id] == 1 and good_bbox_masks[batch_id, prop_id] == 1:
                    object_id = str(detected_object_ids[batch_id, prop_id].item())
                    caption_decoded = decode_caption(captions[batch_id, prop_id], self.dataset.vocabulary["idx2word"])

                    try:
                        ann_list = list(self.organized[scene_id][object_id].keys())
                        object_name = self.organized[scene_id][object_id][ann_list[0]]["object_name"]
                        key = "{}|{}|{}".format(scene_id, object_id, object_name)
                        
                        if key in candidates.keys():
                            continue
                        else:
                            candidates[key] = [caption_decoded]
                            corpus[key] = self.corpus[key]

                            candidate_ids.append((running_id, batch_id, prop_id))
                            running_id+=1

                            assert len(candidates) == len(candidate_ids)
                        
                    except KeyError:
                        continue


        bleu, rouge, cider, spice = self._compute_scores(corpus, candidates)
        bleu_1, bleu_2, bleu_3, bleu_4 = bleu

        bleu_1 = np.asarray(bleu_1)
        bleu_2 = np.asarray(bleu_2)
        bleu_3 = np.asarray(bleu_3)
        bleu_4 = np.asarray(bleu_4)

        rouge = np.asarray(rouge)
        cider = np.asarray(cider)
        spice = np.asarray(spice)

        #substract running mean
        cider_mean, spice_mean = self._update_running_means(cider.mean(), spice.mean())
        cider_base = (cider - cider_mean)
        spice_base = (spice - spice_mean)

        all_scores = np.stack((bleu_1, bleu_2, bleu_3, bleu_4, rouge, cider_base, spice_base), axis=0)

        for i, batch_id, prop_id in candidate_ids:
            scores[batch_id, prop_id] = (self.score_weights*all_scores[:,i]).sum()

        scorelog=[bleu_1.mean(), bleu_2.mean(), bleu_3.mean(), bleu_4.mean(), rouge.mean(), cider.mean(), spice.mean()]
        return scores, scorelog, corpus, candidates


    # def get_sample_score(self, data_dict, conf):
    #     actions = data_dict["actions"].detach().int().cpu().numpy()
    #     batch_size, num_actions = actions.shape
    #     scores = torch.zeros((batch_size, num_actions))

    #     target_caps = data_dict["lang_ids"][:, 1:conf.TRAIN.MAX_DES_LEN]

    #     for i in range(batch_size):
    #         decoded_pred = decode_caption(actions[i], self.dataset.vocabulary["idx2word"])
    #         decoded_gt = decode_caption(target_caps[i], self.dataset.vocabulary["idx2word"])
    #         candidate = {
    #             "k": [decoded_pred]
    #         }
    #         gt = {
    #             "k": [decoded_gt]
    #         }
    #         bleu, _ = BLEU.compute_score(gt, candidate)
    #         bleu = np.mean(bleu)
    #         rouge, _ = ROUGE.compute_score(gt, candidate)
    #         cider, _ = CIDER.compute_score(candidate, gt)
    #         score = (bleu + rouge + cider) / 3
    #         if score <= 0.2:
    #             score = 0
    #         sentence_score = torch.ones((num_actions)) * score
    #         scores[i] = sentence_score

    #     return scores

    
