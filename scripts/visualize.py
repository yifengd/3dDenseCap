# HACK ignore warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import json
import torch
import argparse
from easydict import EasyDict

import numpy as np

from tqdm import tqdm
from copy import deepcopy
from shutil import copyfile

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder

from data.scannet.model_util_scannet import ScannetDatasetConfig
from dataset.dataset import ScannetReferenceDataset
from config.scan2cap_config import CONF
from lib.ap_helper import parse_predictions, gt_scores
from lib.loss_helper import get_scene_cap_loss
from model.capnet import CapNet
from lib.colors import COLORS
from dataset.dataloader import get_dataloader

SCANNET_MESH = os.path.join(CONF.PATH.AXIS_ALIGNED_MESH, "{}", "axis_aligned_scene.ply")
SCANNET_AGGR = os.path.join(CONF.PATH.SCANNET_SCANS, "{}/{}_vh_clean.aggregation.json") # scene_id, scene_id

SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.SCANREFER, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.SCANREFER, "ScanRefer_filtered_val.json")))
SCANREFER_ORGANIZED = json.load(open(os.path.join(CONF.PATH.SCANREFER, "ScanRefer_filtered_organized.json")))

# constants
DC = ScannetDatasetConfig()

def get_data(conf, args, scanrefer, all_scene_list, split, config, augment, scan2cad_rotation=None):
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

    # to device
    model.cuda()

    # set mode
    model.eval()

    return model

def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_{}.txt".format(split)))])

    return scene_list

def get_eval_data(args):
    eval_scene_list = get_scannet_scene_list("val") if args.scene_id == "-1" else [args.scene_id]
    scanrefer_eval = []
    for scene_id in eval_scene_list:
        data = deepcopy(SCANREFER_VAL[0])
        data["scene_id"] = scene_id
        scanrefer_eval.append(data)

    print("eval on {} samples".format(len(scanrefer_eval)))

    return scanrefer_eval, eval_scene_list

def decode_caption(raw_caption, idx2word):
    decoded = ["sos"]
    for token_idx in raw_caption:
        token_idx = token_idx.item()
        token = idx2word[str(token_idx)]
        decoded.append(token)
        if token == "eos": break

    if "eos" not in decoded: decoded.append("eos")
    decoded = " ".join(decoded)

    return decoded

def write_ply(verts, colors, indices, output_file):
    if colors is None:
        colors = np.zeros_like(verts)
    if indices is None:
        indices = []

    file = open(output_file, 'w')
    file.write('ply \n')
    file.write('format ascii 1.0\n')
    file.write('element vertex {:d}\n'.format(len(verts)))
    file.write('property float x\n')
    file.write('property float y\n')
    file.write('property float z\n')
    file.write('property uchar red\n')
    file.write('property uchar green\n')
    file.write('property uchar blue\n')
    file.write('element face {:d}\n'.format(len(indices)))
    file.write('property list uchar uint vertex_indices\n')
    file.write('end_header\n')
    for vert, color in zip(verts, colors):
        file.write("{:f} {:f} {:f} {:d} {:d} {:d}\n".format(vert[0], vert[1], vert[2] , int(color[0]*255), int(color[1]*255), int(color[2]*255)))
    for ind in indices:
        file.write('3 {:d} {:d} {:d}\n'.format(ind[0], ind[1], ind[2]))
    file.close()

def write_bbox(corners, color, output_file):
    """
    bbox: (cx, cy, cz, lx, ly, lz, r), center and length in three axis, the last is the rotation
    output_file: string
    """
    def create_cylinder_mesh(radius, p0, p1, stacks=10, slices=10):
        
        import math

        def compute_length_vec3(vec3):
            return math.sqrt(vec3[0]*vec3[0] + vec3[1]*vec3[1] + vec3[2]*vec3[2])
        
        def rotation(axis, angle):
            rot = np.eye(4)
            c = np.cos(-angle)
            s = np.sin(-angle)
            t = 1.0 - c
            axis /= compute_length_vec3(axis)
            x = axis[0]
            y = axis[1]
            z = axis[2]
            rot[0,0] = 1 + t*(x*x-1)
            rot[0,1] = z*s+t*x*y
            rot[0,2] = -y*s+t*x*z
            rot[1,0] = -z*s+t*x*y
            rot[1,1] = 1+t*(y*y-1)
            rot[1,2] = x*s+t*y*z
            rot[2,0] = y*s+t*x*z
            rot[2,1] = -x*s+t*y*z
            rot[2,2] = 1+t*(z*z-1)
            return rot


        verts = []
        indices = []
        diff = (p1 - p0).astype(np.float32)
        height = compute_length_vec3(diff)
        for i in range(stacks+1):
            for i2 in range(slices):
                theta = i2 * 2.0 * math.pi / slices
                pos = np.array([radius*math.cos(theta), radius*math.sin(theta), height*i/stacks])
                verts.append(pos)
        for i in range(stacks):
            for i2 in range(slices):
                i2p1 = math.fmod(i2 + 1, slices)
                indices.append( np.array([(i + 1)*slices + i2, i*slices + i2, i*slices + i2p1], dtype=np.uint32) )
                indices.append( np.array([(i + 1)*slices + i2, i*slices + i2p1, (i + 1)*slices + i2p1], dtype=np.uint32) )
        transform = np.eye(4)
        va = np.array([0, 0, 1], dtype=np.float32)
        vb = diff
        vb /= compute_length_vec3(vb)
        axis = np.cross(vb, va)
        angle = np.arccos(np.clip(np.dot(va, vb), -1, 1))
        if angle != 0:
            if compute_length_vec3(axis) == 0:
                dotx = va[0]
                if (math.fabs(dotx) != 1.0):
                    axis = np.array([1,0,0]) - dotx * va
                else:
                    axis = np.array([0,1,0]) - va[1] * va
                axis /= compute_length_vec3(axis)
            transform = rotation(axis, -angle)
        transform[:3,3] += p0
        verts = [np.dot(transform, np.array([v[0], v[1], v[2], 1.0])) for v in verts]
        verts = [np.array([v[0], v[1], v[2]]) / v[3] for v in verts]
            
        return verts, indices

    def get_bbox_edges(bbox_min, bbox_max):
        def get_bbox_verts(bbox_min, bbox_max):
            verts = [
                np.array([bbox_min[0], bbox_min[1], bbox_min[2]]),
                np.array([bbox_max[0], bbox_min[1], bbox_min[2]]),
                np.array([bbox_max[0], bbox_max[1], bbox_min[2]]),
                np.array([bbox_min[0], bbox_max[1], bbox_min[2]]),

                np.array([bbox_min[0], bbox_min[1], bbox_max[2]]),
                np.array([bbox_max[0], bbox_min[1], bbox_max[2]]),
                np.array([bbox_max[0], bbox_max[1], bbox_max[2]]),
                np.array([bbox_min[0], bbox_max[1], bbox_max[2]])
            ]
            return verts

        box_verts = get_bbox_verts(bbox_min, bbox_max)
        edges = [
            (box_verts[0], box_verts[1]),
            (box_verts[1], box_verts[2]),
            (box_verts[2], box_verts[3]),
            (box_verts[3], box_verts[0]),

            (box_verts[4], box_verts[5]),
            (box_verts[5], box_verts[6]),
            (box_verts[6], box_verts[7]),
            (box_verts[7], box_verts[4]),

            (box_verts[0], box_verts[4]),
            (box_verts[1], box_verts[5]),
            (box_verts[2], box_verts[6]),
            (box_verts[3], box_verts[7])
        ]
        return edges

    radius = 0.03
    offset = [0,0,0]
    verts = []
    indices = []
    colors = []

    box_min = np.min(corners, axis=0)
    box_max = np.max(corners, axis=0)
    edges = get_bbox_edges(box_min, box_max)
    for k in range(len(edges)):
        cyl_verts, cyl_ind = create_cylinder_mesh(radius, edges[k][0], edges[k][1])
        cur_num_verts = len(verts)
        cyl_color = [[c / 255 for c in color] for _ in cyl_verts]
        cyl_verts = [x + offset for x in cyl_verts]
        cyl_ind = [x + cur_num_verts for x in cyl_ind]
        verts.extend(cyl_verts)
        indices.extend(cyl_ind)
        colors.extend(cyl_color)

    write_ply(verts, colors, indices, output_file)

def visualize(conf, args):
    print("initializing...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # get eval data
    scanrefer_eval, eval_scene_list = get_eval_data(args)

    # get dataloader
    dataset, dataloader = get_data(conf, args, scanrefer_eval, eval_scene_list, "val", DC, augment=False)

    # get model
    model = get_model(conf, args, dataset)

    object_id_to_object_name = {}
    for scene_id in eval_scene_list:
        object_id_to_object_name[scene_id] = {}

        aggr_file = json.load(open(SCANNET_AGGR.format(scene_id, scene_id)))
        for entry in aggr_file["segGroups"]:
            object_id = str(entry["objectId"])
            object_name = entry["label"]
            if len(object_name.split(" ")) > 1: object_name = "_".join(object_name.split(" "))

            object_id_to_object_name[scene_id][object_id] = object_name

    # forward
    print("visualizing...")
    for data_dict in tqdm(dataloader):
        data_dict['epoch'] = torch.tensor(500)
        # move to cuda
        for key in data_dict:
            data_dict[key] = data_dict[key].cuda()

        with torch.no_grad():
            data_dict = model(data_dict, use_tf=False, is_eval=True)
            data_dict = get_scene_cap_loss(data_dict, device, DC, weights=dataset.weights, detection=True, caption=False, prepare_epochs=0)

        # unpack
        captions = data_dict["lang_cap"].argmax(-1) # batch_size, num_proposals, max_len - 1
        dataset_ids = data_dict["dataset_idx"]
        batch_size, num_proposals, _ = captions.shape

        # post-process
        # config
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
        if args.use_gt_scores:
            obj_masks = torch.where(gt_scores(data_dict, batch_size, num_proposals) > CONF.PROPOSAL_SCORE_THRE, 1, 0)
        else:
            obj_masks = data_dict["bbox_mask"]  # torch.argmax(, 2).long()

        # final mask
        nms_masks = nms_masks * obj_masks

        # pick out object ids of detected objects
        # detected_object_ids = torch.gather(data_dict["scene_object_ids"], 1, data_dict["object_assignment"])

        # bbox corners
        detected_bbox_corners = data_dict["bbox_corner"]  # batch_size, num_proposals, 8, 3
        detected_bbox_centers = data_dict['bbox_parameters'][:, :3]  # batch_size, num_proposals, 3

        for batch_id in range(batch_size):
            dataset_idx = dataset_ids[batch_id].item()
            scene_id = dataset.scanrefer[dataset_idx]["scene_id"]

            scene_root = os.path.join(CONF.PATH.OUTPUT, args.folder, "vis/{}".format(scene_id))
            os.makedirs(scene_root, exist_ok=True)
            mesh_path = os.path.join(scene_root, "{}.ply".format(scene_id))
            copyfile(SCANNET_MESH.format(scene_id), mesh_path)

            MEAN_XYZ = data_dict['mean_xyz']

            candidates = {}
            for prop_id in range(num_proposals):
                if nms_masks[batch_id, prop_id] > 0:
                    object_id = str(prop_id)
                    caption_decoded = decode_caption(captions[batch_id, prop_id], dataset.vocabulary["idx2word"])
                    # detected_bbox_corner = detected_bbox_corners[batch_id, prop_id].detach().cpu().numpy()

                    detected_bbox_corner = detected_bbox_corners[batch_id, prop_id] + MEAN_XYZ
                    detected_bbox_corner = detected_bbox_corner.detach().cpu().numpy()

                    # print(scene_id, object_id)
                    try:
                        # ann_list = list(SCANREFER_ORGANIZED[scene_id][object_id].keys())
                        # object_name = SCANREFER_ORGANIZED[scene_id][object_id][ann_list[0]]["object_name"]

                        object_name = DC.nyu40id2class_name[int(data_dict['bbox_sems'].squeeze()[prop_id])]
                        # store
                        candidates[object_id] = {
                            "object_name": object_name,
                            "description": caption_decoded
                        }

                        ply_name = "pred-{}-{}.ply".format(object_id, object_name)
                        ply_path = os.path.join(scene_root, ply_name)

                        palette_idx = int(object_id) % len(COLORS)
                        color = COLORS[palette_idx]
                        write_bbox(detected_bbox_corner, color, ply_path)
                        
                    except KeyError:
                        continue

            # save predictions for the scene
            pred_path = os.path.join(CONF.PATH.OUTPUT, args.folder, "vis/{}/predictions.json".format(scene_id))
            with open(pred_path, "w") as f:
                json.dump(candidates, f, indent=4)

            # gt_object_ids = VOTENET_DATABASE["0|{}_gt_ids".format(scene_id)]
            # gt_object_ids = np.array(gt_object_ids)

            # gt_bbox_corners = VOTENET_DATABASE["0|{}_gt_corners".format(scene_id)]
            # gt_bbox_corners = np.array(gt_bbox_corners)

            gt_bbox_corners = data_dict["gt_box_corner_label"]
            gt_object_ids = data_dict["gt_box_object_ids"]

            for i, object_id in enumerate(gt_object_ids.squeeze()):
                object_id = str(int(object_id))
                object_name = object_id_to_object_name[scene_id][object_id]

                ply_name = "gt-{}-{}.ply".format(object_id, object_name)
                ply_path = os.path.join(scene_root, ply_name)

                palette_idx = int(object_id) % len(COLORS)
                color = COLORS[palette_idx]

                gt_corners = gt_bbox_corners.squeeze()[i] + MEAN_XYZ
                gt_corners = gt_corners.cpu().detach().numpy()

                write_bbox(gt_corners, color, ply_path)

    print("done!")

def get_conf(args):
    with open(os.path.join("outputs", args.folder, "config.json")) as f:
        dict = json.load(f)
        conf = EasyDict(dict)
    return conf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help="Folder containing the model")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--batch_size", type=int, help="batch size", default=1)
    parser.add_argument("--scene_id", type=str, help="scene id", default="scene0000_00")
    parser.add_argument("--use_gt_scores", action="store_true", help="Use the ground truth objectness scores instead of predicted scores")
    parser.add_argument("--use_last", action="store_true", help="Use the last model instead of the best one")

    args = parser.parse_args()
    # setting
    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.gpu)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    conf = get_conf(args)

    visualize(conf, args)