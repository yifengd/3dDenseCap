# HACK ignore warnings
import itertools
import warnings

from easydict import EasyDict

from config.scan2cap_config import CONF
from dataset.dataloader import get_dataloader
from dataset.dataset import ScannetReferenceDataset
from model.capnet import CapNet

warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import json
import torch
import argparse

import numpy as np

from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder

import lib.capeval.bleu.bleu as capblue
import lib.capeval.cider.cider as capcider
import lib.capeval.rouge.rouge as caprouge

from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.ap_helper import APCalculator, parse_predictions, parse_groundtruths
from lib.loss_helper import get_scene_cap_loss
from lib.eval_helper import eval_cap

# SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
# SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))
# SCANREFER_DUMMY = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_dummy.json")))

# constants
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

    dataloader = get_dataloader(dataset, batch_size=min(args.batch_size, len(dataset)), shuffle=True, num_workers=2)

    return dataset, dataloader

def get_conf(args):
    with open(os.path.join("outputs", args.folder, "config.json")) as f:
        dict = json.load(f)
        conf = EasyDict(dict)
    return conf

def get_model(args, dataset, device, conf, root=CONF.PATH.OUTPUT, eval_pretrained=False):
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
        pg_ublocks=conf.PG.UBLOCK_LAYERS,
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

    if args.rl_policy is not None:
        policy_path = os.path.join(root, args.folder, args.rl_policy)
        model.caption.load_state_dict(torch.load(policy_path), strict=False)
    
    # to device
    model.to(device)

    # set mode
    model.eval()

    return model

def get_scannet_scene_list(data):
    # scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_{}.txt".format(split)))])
    scene_list = sorted(list(set([d["scene_id"] for d in data])))

    return scene_list

def get_eval_data(args):
    scanrefer_train = json.load(open(os.path.join(CONF.PATH.SCANREFER, "ScanRefer_filtered_train.json")))
    scanrefer_val = json.load(open(os.path.join(CONF.PATH.SCANREFER, "ScanRefer_filtered_val.json")))

    if args.scene_id != None:
        eval_scene_list = [args.scene_id]
    else:
        eval_scene_list = get_scannet_scene_list(scanrefer_train) if args.use_train else get_scannet_scene_list(scanrefer_val)

    scanrefer_eval = []
    for scene_id in eval_scene_list:
        data = deepcopy(scanrefer_train[0]) if args.use_train else deepcopy(scanrefer_val[0])
        data["scene_id"] = scene_id
        scanrefer_eval.append(data)

    print("eval on {} samples".format(len(scanrefer_eval)))

    return scanrefer_eval, eval_scene_list

def eval_caption(args):
    print("initializing...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    conf = get_conf(args)

    # get eval data
    scanrefer_eval, eval_scene_list = get_eval_data(args)

    # get dataloader
    dataset, dataloader = get_data(conf, args, scanrefer_eval, eval_scene_list, "train" if args.use_train else "val", augment=False)

    # get model
    model = get_model(args, dataset, device, conf)

    # evaluate
    bleu, cider, rouge, meteor, log = eval_cap(model, device, dataset, dataloader, "train" if args.use_train else "val", args.folder, 0,
                                            force=True, save_interm=args.save_interm, min_iou=args.min_iou)

    # report
    print("\n----------------------Evaluation-----------------------")
    print("[BLEU-1] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[0][0], max(bleu[1][0]), min(bleu[1][0])))
    print("[BLEU-2] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[0][1], max(bleu[1][1]), min(bleu[1][1])))
    print("[BLEU-3] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[0][2], max(bleu[1][2]), min(bleu[1][2])))
    print("[BLEU-4] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[0][3], max(bleu[1][3]), min(bleu[1][3])))
    print("[CIDEr] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(cider[0], max(cider[1]), min(cider[1])))
    print("[ROUGE-L] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(rouge[0], max(rouge[1]), min(rouge[1])))
    print("[METEOR] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(meteor[0], max(meteor[1]), min(meteor[1]))) # max(meteor[1]), min(meteor[1])))
    print()

def eval_detection(args):
    print("evaluate detection...")
    # constant
    DC = ScannetDatasetConfig()

    conf = get_conf(args)
    
    # init training dataset
    print("preparing data...")
    # get eval data
    scanrefer_eval, eval_scene_list = get_eval_data(args)

    # get dataloader
    dataset, dataloader = get_data(conf, args, scanrefer_eval, eval_scene_list, "train" if args.use_train else "val", augment=False)


    # model
    print("initializing...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # root = CONF.PATH.PRETRAINED if args.eval_pretrained else CONF.PATH.OUTPUT
    model = get_model(args, dataset, device, conf, eval_pretrained=args.eval_pretrained)

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
    AP_IOU_THRESHOLDS = [0.25, 0.5]
    AP_CALCULATOR_LIST = [APCalculator(iou_thresh, DC.class2type) for iou_thresh in AP_IOU_THRESHOLDS]

    sem_acc = []
    for data in tqdm(dataloader):
        data['epoch'] = torch.tensor(0)
        for key in data:
            data[key] = data[key].cuda()

        # feed
        with torch.no_grad():
            data = model(data, False, True)
            data = get_scene_cap_loss(data, device, DC, weights=dataset.weights, detection=True, caption=False, prepare_epochs=0)

        batch_pred_map_cls = parse_predictions(data, POST_DICT) 
        batch_gt_map_cls = parse_groundtruths(data, POST_DICT)

        batch_pred_map_cls_nyu40id2class_filtered = []
        for i in range(len(batch_pred_map_cls)):
            batch_pred_map_cls_nyu40id2class_filtered.append([])
            for j in range(len(batch_pred_map_cls[i])):
                pred_sem_cls, box_params, box_score = batch_pred_map_cls[i][j]
                if int(pred_sem_cls) in DC.nyu40id2class: # filter
                    batch_pred_map_cls_nyu40id2class_filtered[i].append((DC.nyu40id2class[int(pred_sem_cls)], box_params, box_score))

        # for i in range(len(batch_gt_map_cls)):
        #     for j in range(len(batch_gt_map_cls[i])):
        #         pred_sem_cls, box_params = batch_gt_map_cls[i][j]
        #         batch_gt_map_cls[i][j] = (0, box_params,)

        for ap_calculator in AP_CALCULATOR_LIST:
            ap_calculator.step(batch_pred_map_cls_nyu40id2class_filtered, batch_gt_map_cls)

    # aggregate object detection results and report
    for i, ap_calculator in enumerate(AP_CALCULATOR_LIST):
        print()
        print("-"*10, "iou_thresh: %f"%(AP_IOU_THRESHOLDS[i]), "-"*10)
        metrics_dict = ap_calculator.compute_metrics()
        for key in metrics_dict:
            print("eval %s: %f"%(key, metrics_dict[key]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help="Folder containing the model")
    parser.add_argument("--batch_size", type=int, help="batch size", default=8)
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")

    parser.add_argument("--min_iou", type=float, default=0.25, help="Min IoU threshold for evaluation")

    parser.add_argument("--use_train", action="store_true", help="Use train split in evaluation.")
    parser.add_argument("--use_last", action="store_true", help="Use the last model")

    parser.add_argument("--eval_caption", action="store_true", help="evaluate the reference localization results")
    parser.add_argument("--eval_detection", action="store_true", help="evaluate the object detection results")

    parser.add_argument("--save_interm", action="store_true", help="Save the intermediate results")

    parser.add_argument("--rl_policy", type=str, default=None, help="Use a RL policy for caption generation")
    parser.add_argument("--scene_id", type=str, default=None, help="Only eval on one scene")
    args = parser.parse_args()

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.gpu)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # evaluate
    if args.eval_caption: eval_caption(args)
    if args.eval_detection: eval_detection(args)

