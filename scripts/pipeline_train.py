# HACK ignore warnings
from re import M
import warnings

from dataset.dataloader import get_dataloader

warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import json
import torch
import torch.optim as optim
import numpy as np

from datetime import datetime
from copy import deepcopy

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from data.scannet.model_util_scannet import ScannetDatasetConfig
from dataset.dataset import ScannetReferenceDataset
from lib.solver import Solver
from config.scan2cap_config import CONF
from model.capnet import CapNet


# SCANREFER_DUMMY = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_dummy.json")))
SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.SCANREFER, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.SCANREFER, "ScanRefer_filtered_val.json")))

# extracted ScanNet object rotations from Scan2CAD
# NOTE some scenes are missing in this annotation!!!
SCAN2CAD_ROTATION = json.load(open(os.path.join(CONF.PATH.SCAN2CAD, "scannet_instance_rotations.json")))

# constants
DC = ScannetDatasetConfig()


def get_data(scanrefer, all_scene_list, split, config, augment, scan2cad_rotation=None):
    dataset = ScannetReferenceDataset(
        scanrefer=scanrefer[:CONF.NUM_SCENES],
        scanrefer_all_scene=all_scene_list,
        split=split,
        name=CONF.DATASET,
        use_height=(not CONF.NO_HEIGHT),
        use_color=CONF.USE_COLOR,
        use_normal=CONF.USE_NORMAL,
        augment=augment,
        scan2cad_rotation=scan2cad_rotation
    )
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    dataloader = get_dataloader(dataset, batch_size=CONF.BATCH_SIZE, shuffle=True, num_workers=CONF.NUM_WORKERS)

    return dataset, dataloader


def get_model(dataset, device):
    # initiate model
    input_channels = 0
    if CONF.USE_COLOR:
        input_channels += 3
    if CONF.USE_NORMAL:
        input_channels += 3
    if not CONF.NO_HEIGHT:
        input_channels += 1


    if CONF.PG.PRETRAIN != "":
        pg_prepare_epochs = -1
    else:
        pg_prepare_epochs = CONF.PG.PREPARE_EPOCHS

    model = CapNet(
        num_class=41,
        vocabulary=dataset.vocabulary,
        embeddings=dataset.glove,
        num_locals=CONF.NUM_LOCALS,
        num_proposal=CONF.NUM_PROPOSALS,
        no_caption=CONF.NO_CAPTION,
        use_topdown=CONF.USE_TOPDOWN,
        query_mode=CONF.QUERY_MODE,
        graph_mode=CONF.GRAPH_MODE,
        num_graph_steps=CONF.NUM_GRAPH_STEPS,
        use_relation=CONF.USE_RELATION,
        use_orientation=CONF.USE_ORIENTATION,
        use_distance=CONF.USE_DISTANCE,
        pg_prepare_epochs=pg_prepare_epochs,
        pg_input_c=input_channels,
        pg_m = CONF.PG.M,
        pg_ublocks=CONF.PG.UBLOCK_LAYERS,
        pg_cluster_npoint_thre=CONF.PG.CLUSTER_NPOINT_THRE,
        pg_cluster_radius=CONF.PG.CLUSTER_RADIUS,
        pg_meanActive=CONF.PG.MEAN_ACTIVE,
        pg_shift_meanActive=CONF.PG.SHIFT_MEAN_ACTIVE,
        pg_score_scale=CONF.PG.SCORE_SCALE,
        pg_score_fullscale=CONF.PG.SCORE_FULLSCALE,
        pg_score_mode=CONF.PG.SCORE_MODE,
        pg_block_reps=CONF.PG.BLOCK_REPS,
        pg_block_residual=CONF.PG.BLOCK_RESIDUAL,
        pg_fix_module=CONF.PG.FIX_MODULE,
        proposal_score_thre=CONF.PROPOSAL_SCORE_THRE,
    )

    if CONF.PG.PRETRAIN != "":
        raise("Not implemented")

    
    # to device
    model.to(device)

    return model

def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params

def get_solver(dataset, dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(dataset["train"], device)
    optimizer = optim.Adam(model.parameters(), lr=CONF.LR, weight_decay=CONF.WD)

    checkpoint_best = None

    if CONF.USE_CHECKPOINT:
        print("loading checkpoint {}...".format(CONF.USE_CHECKPOINT))
        stamp = CONF.USE_CHECKPOINT + "--" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        root = os.path.join(CONF.PATH.OUTPUT, stamp)

        # model_state = torch.load(os.path.join(CONF.PATH.OUTPUT, CONF.USE_CHECKPOINT, "model.pth"))
        # model.load_state_dict(model_state)
        # checkpoint_best = None

        checkpoint = torch.load(os.path.join(CONF.PATH.OUTPUT, CONF.USE_CHECKPOINT, "checkpoint.tar"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        checkpoint_best = checkpoint["best"]

        for group in optimizer.param_groups:
            group['lr'] = CONF.LR
            group['weight_decay'] = CONF.WD
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        os.makedirs(root, exist_ok=True)

    # scheduler parameters for training solely the detection pipeline
    LR_DECAY_STEP = None # [80, 120, 160] if args.no_caption else None
    LR_DECAY_RATE = None #  0.1 if args.no_caption else None
    BN_DECAY_STEP = None # 20 if args.no_caption else None
    BN_DECAY_RATE = None # 0.5 if args.no_caption else None

    solver = Solver(
        model=model,
        device=device,
        config=DC,
        dataset=dataset,
        dataloader=dataloader,
        optimizer=optimizer,
        stamp=stamp,
        val_step=CONF.VAL_STEP,
        detection=not CONF.NO_DETECTION,
        caption=not CONF.NO_CAPTION,
        orientation=CONF.USE_ORIENTATION,
        distance=CONF.USE_DISTANCE,
        use_tf=CONF.USE_TF,
        lr_decay_step=LR_DECAY_STEP,
        lr_decay_rate=LR_DECAY_RATE,
        bn_decay_step=BN_DECAY_STEP,
        bn_decay_rate=BN_DECAY_RATE,
        criterion=CONF.CRITERION,
        checkpoint_best=checkpoint_best,
        train_eval_limit=CONF.TRAIN_EVAL_LIMIT
    )
    num_params = get_num_params(model)

    return solver, num_params, root

def save_info(root, num_params, dataset):

    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump(CONF, f, indent=4)

    info = {}

    info["num_train"] = len(dataset["train"])
    info["num_eval_train"] = len(dataset["eval"]["train"])
    info["num_eval_val"] = len(dataset["eval"]["val"])
    info["num_train_scenes"] = len(dataset["train"].scene_list)
    info["num_eval_train_scenes"] = len(dataset["eval"]["train"].scene_list)
    info["num_eval_val_scenes"] = len(dataset["eval"]["val"].scene_list)
    info["num_params"] = num_params

    with open(os.path.join(root, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

    return scene_list

def get_scanrefer():
    if CONF.DATASET == "ScanRefer":
        scanrefer_train = json.load(open(os.path.join(CONF.PATH.SCANREFER, "ScanRefer_filtered_train.json")))
        scanrefer_eval_train = json.load(open(os.path.join(CONF.PATH.SCANREFER, "ScanRefer_filtered_train.json")))
        scanrefer_eval_val = json.load(open(os.path.join(CONF.PATH.SCANREFER, "ScanRefer_filtered_val.json")))
    # elif args.dataset == "ReferIt3D":
    #     scanrefer_train = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d_train.json")))
    #     scanrefer_eval_train = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d_train.json")))
    #     scanrefer_eval_val = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d_val.json")))
    else:
        raise ValueError("Invalid dataset.")

    if CONF.DEBUG:
        scanrefer_train = [SCANREFER_TRAIN[0]]
        scanrefer_eval_train = [SCANREFER_TRAIN[0]]
        scanrefer_eval_val = [SCANREFER_TRAIN[0]]

    if CONF.NO_CAPTION:
        train_scene_list = get_scannet_scene_list("train")
        val_scene_list = get_scannet_scene_list("val")

        new_scanrefer_train = []
        for scene_id in train_scene_list:
            data = deepcopy(SCANREFER_TRAIN[0])
            data["scene_id"] = scene_id
            new_scanrefer_train.append(data)

        new_scanrefer_eval_train = []
        for scene_id in train_scene_list:
            data = deepcopy(SCANREFER_TRAIN[0])
            data["scene_id"] = scene_id
            new_scanrefer_eval_train.append(data)

        new_scanrefer_eval_val = []
        for scene_id in val_scene_list:
            data = deepcopy(SCANREFER_TRAIN[0])
            data["scene_id"] = scene_id
            new_scanrefer_eval_val.append(data)
    else:
        # get initial scene list
        train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
        val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_eval_val])))

        # filter data in chosen scenes
        new_scanrefer_train = []
        for data in scanrefer_train:
            if data["scene_id"] in train_scene_list:
                new_scanrefer_train.append(data)

        # eval on train
        new_scanrefer_eval_train = []
        for scene_id in train_scene_list:
            data = deepcopy(SCANREFER_TRAIN[0])
            data["scene_id"] = scene_id
            new_scanrefer_eval_train.append(data)

        new_scanrefer_eval_val = []
        for scene_id in val_scene_list:
            data = deepcopy(SCANREFER_TRAIN[0])
            data["scene_id"] = scene_id
            new_scanrefer_eval_val.append(data)

    # all scanrefer scene
    all_scene_list = train_scene_list + val_scene_list

    print("using {} dataset".format(CONF.DATASET))
    print("train on {} samples from {} scenes".format(len(new_scanrefer_train), len(train_scene_list)))
    print("eval on {} scenes from train and {} scenes from val".format(len(new_scanrefer_eval_train), len(new_scanrefer_eval_val)))

    return new_scanrefer_train, new_scanrefer_eval_train, new_scanrefer_eval_val, all_scene_list


def train():
    # init training dataset
    print("preparing data...")
    scanrefer_train, scanrefer_eval_train, scanrefer_eval_val, all_scene_list = get_scanrefer()

    # dataloader
    train_dataset, train_dataloader = get_data(scanrefer_train, all_scene_list, "train", DC, CONF.AUGMENT_DATA, SCAN2CAD_ROTATION)
    eval_train_dataset, eval_train_dataloader = get_data(scanrefer_eval_train, all_scene_list, "val", DC, False)
    eval_val_dataset, eval_val_dataloader = get_data(scanrefer_eval_val, all_scene_list, "val", DC, False)
    dataset = {
        "train": train_dataset,
        "eval": {
            "train": eval_train_dataset,
            "val": eval_val_dataset
        }
    }
    dataloader = {
        "train": train_dataloader,
        "eval": {
            "train": eval_train_dataloader,
            "val": eval_val_dataloader
        }
    }

    print("initializing...")
    solver, num_params, root = get_solver(dataset, dataloader)

    print("Start training...\n")
    save_info(root, num_params, dataset)
    solver(CONF.NUM_EPOCH, CONF.VERBOSE)

if __name__ == "__main__":


    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = CONF.GPU
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # reproducibility
    torch.manual_seed(CONF.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(CONF.SEED)

    train()