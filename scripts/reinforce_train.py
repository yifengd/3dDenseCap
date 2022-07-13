# HACK ignore warnings
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import json
import torch
import argparse
from easydict import EasyDict
import datetime

from tqdm import tqdm
from copy import deepcopy
from tensorboardX import SummaryWriter

sys.path.append(os.path.join(os.getcwd()))  # HACK add the root folder

import lib.rl.rl_utils as rl_utils

from data.scannet.model_util_scannet import ScannetDatasetConfig
from dataset.dataset import ScannetReferenceDataset
from config.scan2cap_config import CONF
from lib.ap_helper import parse_predictions
from lib.loss_helper import get_scene_cap_loss
from model.capnet import CapNet
from dataset.dataloader import get_dataloader
from model.caption_module import TopDownSceneCaptionModule

SCANNET_MESH = os.path.join(CONF.PATH.AXIS_ALIGNED_MESH, "{}", "axis_aligned_scene.ply")
SCANNET_AGGR = os.path.join(CONF.PATH.SCANNET_SCANS, "{}/{}_vh_clean.aggregation.json")  # scene_id, scene_id

SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.SCANREFER, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.SCANREFER, "ScanRefer_filtered_val.json")))
SCANREFER_ORGANIZED = json.load(open(os.path.join(CONF.PATH.SCANREFER, "ScanRefer_filtered_organized.json")))

DC = ScannetDatasetConfig()


def get_data(conf, args, scanrefer, all_scene_list, split, augment, scan2cad_rotation=None):
    dataset = ScannetReferenceDataset(
        scanrefer=scanrefer,
        scanrefer_all_scene=all_scene_list,
        split=split,
        name="ScanRefer",
        use_height=(not conf.NO_HEIGHT),
        use_color=conf.USE_COLOR,
        use_normal=conf.USE_NORMAL,
        augment=augment,
        scan2cad_rotation=scan2cad_rotation
    )
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    dataloader = get_dataloader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    return dataset, dataloader


def get_policy(conf, dataset):
    policy = TopDownSceneCaptionModule(dataset.vocabulary,
                                       dataset.glove, 300, conf.PG.UBLOCK_LAYERS[-1] * conf.PG.M,
                                       512, 256, conf.NUM_LOCALS, conf.QUERY_MODE,
                                       conf.USE_RELATION)
    policy.cuda()

    return policy


def get_pretrained_policy(capnet):
    return capnet.caption


def get_model(conf, args, dataset, root=CONF.PATH.OUTPUT):
    # initiate model
    input_channels = 0
    if conf.USE_COLOR:
        input_channels += 3
    if conf.USE_NORMAL:
        input_channels += 3
    if not conf.NO_HEIGHT:
        input_channels += 1

    model = CapNet(
        num_class=41,
        vocabulary=dataset.vocabulary,
        embeddings=dataset.glove,
        num_locals=conf.NUM_LOCALS,
        num_proposal=conf.NUM_PROPOSALS,
        no_caption=conf.NO_CAPTION,
        use_topdown=conf.USE_TOPDOWN,
        query_mode=conf.QUERY_MODE,
        graph_mode=conf.GRAPH_MODE,
        num_graph_steps=conf.NUM_GRAPH_STEPS,
        use_relation=conf.USE_RELATION,
        use_orientation=conf.USE_ORIENTATION,
        use_distance=conf.USE_DISTANCE,
        pg_prepare_epochs=conf.PG.PREPARE_EPOCHS,
        pg_input_c=input_channels,
        pg_m=conf.PG.M,
        # pg_ublocks=conf.PG.UBLOCK_LAYERS,
        pg_cluster_npoint_thre=conf.PG.CLUSTER_NPOINT_THRE,
        pg_cluster_radius=conf.PG.CLUSTER_RADIUS,
        pg_meanActive=conf.PG.MEAN_ACTIVE,
        pg_shift_meanActive=conf.PG.SHIFT_MEAN_ACTIVE,
        pg_score_scale=conf.PG.SCORE_SCALE,
        pg_score_fullscale=conf.PG.SCORE_FULLSCALE,
        pg_score_mode=conf.PG.SCORE_MODE,
        pg_block_reps=conf.PG.BLOCK_REPS,
        pg_block_residual=conf.PG.BLOCK_RESIDUAL,
        pg_fix_module=conf.PG.FIX_MODULE,
        proposal_score_thre=conf.PROPOSAL_SCORE_THRE,
    )

    # load
    model_name = "model_last.pth" if args.use_last else "model.pth"
    model_path = os.path.join(root, args.folder, model_name)
    model.load_state_dict(torch.load(model_path), strict=False)
    # model.load_state_dict(torch.load(model_path))

    # to device
    model.cuda()

    # set mode
    model.eval()

    return model


def get_scannet_scene_list(split):
    scene_list = sorted(
        [line.rstrip() for line in open(os.path.join(CONF.PATH.SCANREFER, "ScanRefer_filtered_{}.txt".format(split)))])

    return scene_list


def get_scanrefer_scenewise_train(args):
    train_scene_list = get_scannet_scene_list("train") if args.scene_id == "-1" else [args.scene_id]
    scanrefer_train = []
    for scene_id in train_scene_list:
        data = deepcopy(SCANREFER_VAL[0])
        data["scene_id"] = scene_id
        scanrefer_train.append(data)

    print("train on {} samples".format(len(scanrefer_train)))

    return scanrefer_train, train_scene_list


def get_conf(args):
    with open(os.path.join(CONF.PATH.OUTPUT, args.folder, "config.json")) as f:
        dict = json.load(f)
        conf = EasyDict(dict)
    return conf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help="Folder containing the model")
    parser.add_argument("--rl_policy", type=str, default=None, help="Start from a policy model checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--batch_size", type=int, help="batch size", default=1)
    parser.add_argument("--scene_id", type=str, help="Train with only one scene", default="-1")
    parser.add_argument("--use_gt_scores", action="store_true",
                        help="Use the ground truth objectness scores instead of predicted scores")
    parser.add_argument("--use_last", action="store_true", help="Use the last model instead of the best one")
    parser.add_argument("--cold_start", action="store_true", help="Train a new policy model from scratch")

    args = parser.parse_args()
    # setting
    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.gpu)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    if args.cold_start and args.rl_policy is not None:
        raise ValueError("Configuration Error: --cold_start conflicts with --rl_policy")

    conf = get_conf(args)
    raw_data = json.load(open(os.path.join(CONF.PATH.SCANREFER, "ScanRefer_filtered_train.json")))
    max_len = conf.TRAIN.MAX_DES_LEN

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # get training data
    scanrefer_train, train_scene_list = get_scanrefer_scenewise_train(args)

    # get dataloader
    dataset, dataloader = get_data(conf, args, scanrefer_train, train_scene_list, "train", False)

    # get model
    conf.NUM_PROPOSALS = 256
    model = get_model(conf, args, dataset)

    if args.cold_start:
        policy = get_policy(conf, dataset)
    else:
        if args.rl_policy is not None:
            policy_path = os.path.join(CONF.PATH.OUTPUT, args.folder, args.rl_policy)
            model.caption.load_state_dict(torch.load(policy_path), strict=False)

        policy = get_pretrained_policy(model)

    stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    writer = SummaryWriter(os.path.join(conf.PATH.OUTPUT, args.folder, "rl_logs", stamp), flush_secs=10)

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    helper = rl_utils.ReinforceHelper(dataset=dataset, raw_data=raw_data, max_len=max_len,
                                      organized=SCANREFER_ORGANIZED, score_weights=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])

    object_id_to_object_name = {}
    for scene_id in train_scene_list:
        object_id_to_object_name[scene_id] = {}

        aggr_file = json.load(open(SCANNET_AGGR.format(scene_id, scene_id)))
        for entry in aggr_file["segGroups"]:
            object_id = str(entry["objectId"])
            object_name = entry["label"]
            if len(object_name.split(" ")) > 1: object_name = "_".join(object_name.split(" "))

            object_id_to_object_name[scene_id][object_id] = object_name

    # forward
    global_idx = 0

    for epoch in range(10):
        for data_dict in tqdm(dataloader):
            data_dict['epoch'] = torch.tensor(500)
            # move to cuda
            for key in data_dict:
                data_dict[key] = data_dict[key].cuda()

            with torch.no_grad():
                # Run in eval mode
                data_dict = model(data_dict, use_tf=False, is_eval=True)
                data_dict = get_scene_cap_loss(data_dict, device, DC, weights=dataset.weights, detection=True,
                                               caption=False, prepare_epochs=0)

            POST_DICT = {
                "remove_empty_box": False,
                "use_3d_nms": True,
                "nms_iou": 0.25,
                "use_old_type_nms": False,
                "cls_nms": True,  # Can also be False
                "per_class_proposal": True,
                "conf_thresh": 0.05,
                "dataset_config": DC
            }

            # nms mask
            _ = parse_predictions(data_dict, POST_DICT, use_gt_scores=args.use_gt_scores)
            nms_masks = torch.LongTensor(data_dict["pred_mask"]).cuda()

            # objectness mask

            obj_masks = data_dict["bbox_mask"]  # torch.argmax(, 2).long()

            # final mask
            nms_masks = nms_masks * obj_masks

            data_dict["nms_masks"] = nms_masks

            for episode in range(10):
                for key in data_dict:
                    if isinstance(data_dict[key], torch.Tensor):
                        data_dict[key] = data_dict[key].detach()
                # generate descriptions for batch
                data_dict = policy(data_dict, is_eval=True, stochastic=True)  # run in eval mode for scene-wise predictions

                # unpack
                scores, scorelogs, corpus, candidates = helper.get_scores(data_dict, conf)
                bleu1, bleu2, bleu3, bleu4, rouge, cider, spice = scorelogs
                log_probs = data_dict["action_probs"]
                log_probs = torch.log(log_probs)
                rewards = torch.tensor(scores).cuda()

                mask = data_dict["nms_masks"]  # batch_size, num_proposals
                mask = mask.unsqueeze(2).to(dtype=torch.int).detach()  # 1, batch_size, num_proposals

                rewards = rewards.unsqueeze(2).repeat(1, 1, log_probs.shape[-1])
                rewards = rewards * mask
                log_probs = log_probs * mask
                probs = data_dict["action_probs"] * mask

                # compute loss

                policy_loss = (-rewards * log_probs).mean()

                writer.add_scalar("rl/policy_loss", policy_loss, global_idx)
                writer.add_scalar("rl/bleu1", bleu1, global_idx)
                writer.add_scalar("rl/bleu2", bleu2, global_idx)
                writer.add_scalar("rl/bleu3", bleu3, global_idx)
                writer.add_scalar("rl/bleu4", bleu4, global_idx)
                writer.add_scalar("rl/rouge", rouge, global_idx)
                writer.add_scalar("rl/cider", cider, global_idx)
                writer.add_scalar("rl/spice", spice, global_idx)
                writer.add_scalar("rl/action_probs", probs.mean(), global_idx)

                if episode == 9:
                    # save model
                    model_path = os.path.join(conf.PATH.OUTPUT, args.folder, "rl_logs", stamp,
                                              "policy_iter_{}.pth".format(global_idx))
                    torch.save(policy.state_dict(), model_path)

                    corpus_path = os.path.join(CONF.PATH.OUTPUT, args.folder, "rl_logs", stamp, "corpus_RL_train.json")
                    pred_path = os.path.join(CONF.PATH.OUTPUT, args.folder, "rl_logs", stamp, "pred_RL_train.json")

                    with open(corpus_path, "w") as f:
                        json.dump(corpus, f, indent=4)

                    with open(pred_path, "w") as f:
                        json.dump(candidates, f, indent=4)

                policy.zero_grad()
                policy_loss.backward()
                optimizer.step()

                global_idx += 1
