import argparse
import json
import os

import torch
from tqdm import tqdm

from config.scan2cap_config import CONF
from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.loss_helper import get_scene_cap_loss
from scripts.reinforce_train import get_data, get_model
from scripts.pipeline_train import get_scanrefer

SCANNET_MESH = os.path.join(CONF.PATH.AXIS_ALIGNED_MESH, "{}", "axis_aligned_scene.ply")
SCANNET_AGGR = os.path.join(CONF.PATH.SCANNET_SCANS, "{}/{}_vh_clean.aggregation.json") # scene_id, scene_id

SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.SCANREFER, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.SCANREFER, "ScanRefer_filtered_val.json")))
SCANREFER_ORGANIZED = json.load(open(os.path.join(CONF.PATH.SCANREFER, "ScanRefer_filtered_organized.json")))
SCAN2CAD_ROTATION = json.load(open(os.path.join(CONF.PATH.SCAN2CAD, "scannet_instance_rotations.json")))

DC = ScannetDatasetConfig()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help="Folder containing the model")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--batch_size", type=int, help="batch size", default=1)
    parser.add_argument("--use_last", action="store_true", help="Use the last model instead of the best one")

    args = parser.parse_args()
    # setting
    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.gpu)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    gamma = 0.999

    conf = CONF # get_conf(args)
    conf.NUM_SCENES = 100

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # get eval data
    scanrefer_train, scanrefer_eval_train, scanrefer_eval_val, all_scene_list = get_scanrefer()

    # get dataloader
    eval_train_dataset, eval_train_dataloader = get_data(conf, args, scanrefer_eval_train[:5], all_scene_list, "val", DC, augment=False, scan2cad_rotation=SCAN2CAD_ROTATION)
    eval_val_dataset, eval_val_dataloader = get_data(conf, args, scanrefer_eval_val[:5], all_scene_list, "val", DC, augment=False, scan2cad_rotation=SCAN2CAD_ROTATION)

    # get model
    model, policy = get_model(conf, args, eval_train_dataset)

    # object_id_to_object_name = {}
    # for scene_id in eval_scene_list:
    #     object_id_to_object_name[scene_id] = {}
    #
    #     aggr_file = json.load(open(SCANNET_AGGR.format(scene_id, scene_id)))
    #     for entry in aggr_file["segGroups"]:
    #         object_id = str(entry["objectId"])
    #         object_name = entry["label"]
    #         if len(object_name.split(" ")) > 1: object_name = "_".join(object_name.split(" "))
    #
    #         object_id_to_object_name[scene_id][object_id] = object_name

    data_dicts = {}
    # forward
    for data_dict in tqdm(eval_train_dataloader):
        data_dict['epoch'] = torch.tensor(500)
        # move to cuda
        for key in data_dict:
            data_dict[key] = data_dict[key].cuda()

        with torch.no_grad():
            data_dict = model(data_dict, use_tf=False, is_eval=True)
            data_dict = get_scene_cap_loss(data_dict, device, DC, weights=eval_train_dataset.weights, detection=True, caption=False, prepare_epochs=0)

        data_dicts[scanrefer_eval_train[data_dict["dataset_idx"]]['scene_id']] = data_dict

    torch.save(data_dicts, "data/detection_outputs_train.pth")

    data_dicts = {}
    for data_dict in tqdm(eval_val_dataloader):
        data_dict['epoch'] = torch.tensor(500)
        # move to cuda
        for key in data_dict:
            data_dict[key] = data_dict[key].cuda()

        with torch.no_grad():
            data_dict = model(data_dict, use_tf=False, is_eval=True)
            data_dict = get_scene_cap_loss(data_dict, device, DC, weights=eval_train_dataset.weights, detection=True, caption=False, prepare_epochs=0)

        data_dicts[scanrefer_eval_val[data_dict["dataset_idx"]]['scene_id']] = data_dict

    torch.save(data_dicts, "data/detection_outputs_val.pth")
