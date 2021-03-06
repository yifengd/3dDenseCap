'''
File Created: Monday, 25th November 2019 1:35:30 pm
Author: Dave Zhenyu Chen (zhenyu.chen@tum.de)
'''

import os
import sys
import time
import json
import pickle
import numpy as np

from itertools import chain
from collections import Counter
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), "lib"))  # HACK add the lib folder
from config.scan2cap_config import CONF
from util.pc_utils import rotx, roty, rotz, random_sampling
from util.box_util import get_3d_box, get_3d_box_batch
from data.scannet.model_util_scannet import ScannetDatasetConfig, rotate_aligned_boxes_along_axis

# data setting
DC = ScannetDatasetConfig()
MAX_NUM_OBJ = 128
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
OBJ_CLASS_IDS = np.array(
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
     34, 35, 36, 37, 38, 39, 40])  # exclude wall (1), floor (2), ceiling (22)

# data path
SCANNET_V2_TSV = os.path.join(CONF.PATH.SCANNET_META, "scannetv2-labels.combined.tsv")
# SCANREFER_VOCAB = os.path.join(CONF.PATH.DATA, "ScanRefer_vocabulary.json")
VOCAB = os.path.join(CONF.PATH.DATA, "{}_vocabulary.json")  # dataset_name
# SCANREFER_VOCAB_WEIGHTS = os.path.join(CONF.PATH.DATA, "ScanRefer_vocabulary_weights.json")
VOCAB_WEIGHTS = os.path.join(CONF.PATH.DATA, "{}_vocabulary_weights.json")  # dataset_name
# MULTIVIEW_DATA = os.path.join(CONF.PATH.SCANNET_DATA, "enet_feats.hdf5")
# MULTIVIEW_DATA = CONF.MULTIVIEW
GLOVE_PICKLE = os.path.join(CONF.PATH.DATA, "glove.p")


class ReferenceDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def _get_raw2label(self):
        # mapping
        scannet_labels = DC.type2class.keys()
        scannet2label = {label: i for i, label in enumerate(scannet_labels)}

        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2label = {}
        for i in range(len(lines)):
            label_classes_set = set(scannet_labels)
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = elements[7]
            if nyu40_name not in label_classes_set:
                raw2label[raw_name] = scannet2label['others']
            else:
                raw2label[raw_name] = scannet2label[nyu40_name]

        return raw2label

    def _get_unique_multiple_lookup(self):
        all_sem_labels = {}
        cache = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]

            if scene_id not in all_sem_labels:
                all_sem_labels[scene_id] = []

            if scene_id not in cache:
                cache[scene_id] = {}

            if object_id not in cache[scene_id]:
                cache[scene_id][object_id] = {}
                try:
                    all_sem_labels[scene_id].append(self.raw2label[object_name])
                except KeyError:
                    all_sem_labels[scene_id].append(17)

        # convert to numpy array
        all_sem_labels = {scene_id: np.array(all_sem_labels[scene_id]) for scene_id in all_sem_labels.keys()}

        unique_multiple_lookup = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]

            try:
                sem_label = self.raw2label[object_name]
            except KeyError:
                sem_label = 17

            unique_multiple = 0 if (all_sem_labels[scene_id] == sem_label).sum() == 1 else 1

            # store
            if scene_id not in unique_multiple_lookup:
                unique_multiple_lookup[scene_id] = {}

            if object_id not in unique_multiple_lookup[scene_id]:
                unique_multiple_lookup[scene_id][object_id] = {}

            if ann_id not in unique_multiple_lookup[scene_id][object_id]:
                unique_multiple_lookup[scene_id][object_id][ann_id] = None

            unique_multiple_lookup[scene_id][object_id][ann_id] = unique_multiple

        return unique_multiple_lookup

    def _tranform_des(self):
        lang = {}
        label = {}
        for data in tqdm(self.scanrefer):
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            ann_id = data["ann_id"]

            if scene_id not in lang:
                lang[scene_id] = {}
                label[scene_id] = {}

            if object_id not in lang[scene_id]:
                lang[scene_id][object_id] = {}
                label[scene_id][object_id] = {}

            if ann_id not in lang[scene_id][object_id]:
                lang[scene_id][object_id][ann_id] = {}
                label[scene_id][object_id][ann_id] = {}

            # trim long descriptions
            tokens = data["token"][:CONF.TRAIN.MAX_DES_LEN]

            # tokenize the description
            tokens = ["sos"] + tokens + ["eos"]
            embeddings = np.zeros((CONF.TRAIN.MAX_DES_LEN + 2, 300))
            labels = np.zeros((CONF.TRAIN.MAX_DES_LEN + 2))  # start and end

            # load
            for token_id in range(len(tokens)):
                token = tokens[token_id]
                try:
                    embeddings[token_id] = self.glove[token]
                    labels[token_id] = self.vocabulary["word2idx"][token]
                except KeyError:
                    embeddings[token_id] = self.glove["unk"]
                    labels[token_id] = self.vocabulary["word2idx"]["unk"]

            # store
            lang[scene_id][object_id][ann_id] = embeddings
            label[scene_id][object_id][ann_id] = labels

        return lang, label

    def _build_vocabulary(self, dataset_name):
        vocab_path = VOCAB.format(dataset_name)
        if os.path.exists(vocab_path):
            self.vocabulary = json.load(open(vocab_path))
        else:
            if self.split == "train":
                all_words = chain(*[data["token"][:CONF.TRAIN.MAX_DES_LEN] for data in self.scanrefer])
                word_counter = Counter(all_words)
                word_counter = sorted([(k, v) for k, v in word_counter.items() if k in self.glove], key=lambda x: x[1],
                                      reverse=True)
                word_list = [k for k, _ in word_counter]

                # build vocabulary
                word2idx, idx2word = {}, {}
                spw = ["pad_", "unk", "sos", "eos"]  # NOTE distinguish padding token "pad_" and the actual word "pad"
                for i, w in enumerate(word_list):
                    shifted_i = i + len(spw)
                    word2idx[w] = shifted_i
                    idx2word[shifted_i] = w

                # add special words into vocabulary
                for i, w in enumerate(spw):
                    word2idx[w] = i
                    idx2word[i] = w

                vocab = {
                    "word2idx": word2idx,
                    "idx2word": idx2word
                }
                json.dump(vocab, open(vocab_path, "w"), indent=4)

                self.vocabulary = vocab

    def _build_frequency(self, dataset_name):
        vocab_weights_path = VOCAB_WEIGHTS.format(dataset_name)
        if os.path.exists(vocab_weights_path):
            with open(vocab_weights_path) as f:
                weights = json.load(f)
                self.weights = np.array([v for _, v in weights.items()])
        else:
            all_tokens = []
            for scene_id in self.lang_ids.keys():
                for object_id in self.lang_ids[scene_id].keys():
                    for ann_id in self.lang_ids[scene_id][object_id].keys():
                        all_tokens += self.lang_ids[scene_id][object_id][ann_id].astype(int).tolist()

            word_count = Counter(all_tokens)
            word_count = sorted([(k, v) for k, v in word_count.items()], key=lambda x: x[0])

            # frequencies = [c for _, c in word_count]
            # weights = np.array(frequencies).astype(float)
            # weights = weights / np.sum(weights)
            # weights = 1 / np.log(1.05 + weights)

            weights = np.ones((len(word_count)))

            self.weights = weights

            with open(vocab_weights_path, "w") as f:
                weights = {k: v for k, v in enumerate(weights)}
                json.dump(weights, f, indent=4)

    def _load_data(self, dataset_name):
        print("loading data...")
        # load language features
        self.glove = pickle.load(open(GLOVE_PICKLE, "rb"))
        self._build_vocabulary(dataset_name)
        self.num_vocabs = len(self.vocabulary["word2idx"].keys())
        self.lang, self.lang_ids = self._tranform_des()
        self._build_frequency(dataset_name)

        # add scannet data
        self.scene_list = sorted(list(set([data["scene_id"] for data in self.scanrefer])))

        # load scene data
        self.scene_data = {}
        for scene_id in self.scene_list:
            self.scene_data[scene_id] = {}
            # self.scene_data[scene_id]["mesh_vertices"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_vert.npy")
            self.scene_data[scene_id]["mesh_vertices"] = np.load(
                os.path.join(CONF.PATH.SCANNET_DATA, scene_id) + "_aligned_vert.npy")  # axis-aligned
            self.scene_data[scene_id]["instance_labels"] = np.load(
                os.path.join(CONF.PATH.SCANNET_DATA, scene_id) + "_ins_label.npy")
            self.scene_data[scene_id]["semantic_labels"] = np.load(
                os.path.join(CONF.PATH.SCANNET_DATA, scene_id) + "_sem_label.npy")
            # self.scene_data[scene_id]["instance_bboxes"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_bbox.npy")
            self.scene_data[scene_id]["instance_bboxes"] = np.load(
                os.path.join(CONF.PATH.SCANNET_DATA, scene_id) + "_aligned_bbox.npy")

        # prepare class mapping
        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2nyuid = {}
        for i in range(len(lines)):
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = int(elements[4])
            raw2nyuid[raw_name] = nyu40_name

        # store
        self.raw2nyuid = raw2nyuid
        self.raw2label = self._get_raw2label()
        self.unique_multiple_lookup = self._get_unique_multiple_lookup()

    def _translate(self, point_set, bbox):
        # unpack
        coords = point_set[:, :3]

        # translation factors
        x_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        y_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        z_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        factor = [x_factor, y_factor, z_factor]

        # dump
        coords += factor
        point_set[:, :3] = coords
        bbox[:, :3] += factor

        return point_set, bbox


class ScannetReferenceDataset(ReferenceDataset):
    def __init__(self, scanrefer, scanrefer_all_scene, split="train", name="ScanRefer", use_height=False,
                 use_color=True, use_normal=False, augment=False, scan2cad_rotation=None):

        # NOTE only feed the scan2cad_rotation when on the training mode and train split

        super().__init__()
        self.scanrefer = scanrefer
        self.scanrefer_all_scene = scanrefer_all_scene  # all scene_ids in scanrefer
        self.split = split
        self.name = name
        # self.num_points = num_points
        self.use_color = use_color
        self.use_height = use_height
        self.use_normal = use_normal
        # self.use_multiview = use_multiview
        self.augment = augment
        self.scan2cad_rotation = scan2cad_rotation

        # load data
        self._load_data(name)
        self.multiview_data = {}

    def __len__(self):
        return len(self.scanrefer)

    def __getitem__(self, idx):
        start = time.time()
        scene_id = self.scanrefer[idx]["scene_id"]
        object_id = int(self.scanrefer[idx]["object_id"])
        object_name = " ".join(self.scanrefer[idx]["object_name"].split("_"))
        ann_id = self.scanrefer[idx]["ann_id"]

        # get language features
        lang_feat = self.lang[scene_id][str(object_id)][ann_id]
        lang_len = len(self.scanrefer[idx]["token"]) + 2
        lang_len = lang_len if lang_len <= CONF.TRAIN.MAX_DES_LEN + 2 else CONF.TRAIN.MAX_DES_LEN + 2

        # get pc
        mesh_vertices = self.scene_data[scene_id]["mesh_vertices"]
        instance_labels = self.scene_data[scene_id]["instance_labels"]
        semantic_labels = self.scene_data[scene_id]["semantic_labels"]
        instance_bboxes = self.scene_data[scene_id]["instance_bboxes"]

        num_points = mesh_vertices.shape[0]

        if not self.use_color:
            point_cloud = mesh_vertices[:, 0:3]  # do not use color for now
            pcl_color = mesh_vertices[:, 3:6]
        else:
            point_cloud = mesh_vertices[:, 0:6]
            point_cloud[:, 3:6] = (point_cloud[:, 3:6] - MEAN_COLOR_RGB) / 256.0
            pcl_color = point_cloud[:, 3:6]

        MEAN_XYZ = point_cloud[:, :3].mean(0)

        point_cloud[:, :3] = point_cloud[:, :3] - MEAN_XYZ
        instance_bboxes[:, :3] = instance_bboxes[:, :3] - MEAN_XYZ

        if self.use_normal:
            normals = mesh_vertices[:, 6:9]
            point_cloud = np.concatenate([point_cloud, normals], 1)

        # if self.use_multiview:
        #     # load multiview database
        #     pid = mp.current_process().pid
        #     if pid not in self.multiview_data:
        #         self.multiview_data[pid] = h5py.File(MULTIVIEW_DATA, "r", libver="latest")
        #
        #     multiview = self.multiview_data[pid][scene_id]
        #     point_cloud = np.concatenate([point_cloud, multiview], 1)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)

        point_cloud, choices = random_sampling(point_cloud, 100000, return_choices=True)
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]
        pcl_color = pcl_color[choices]

        num_points = 100000

        # ------------------------------- LABELS ------------------------------
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))  # like instance bboxes but only (center, size)
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))  # mask of which bboxes are valid
        angle_classes = np.zeros((MAX_NUM_OBJ,))  # here, always zeros ("class" of the rotation of the object)
        angle_residuals = np.zeros((MAX_NUM_OBJ,))  # here, always zeros
        size_classes = np.zeros((MAX_NUM_OBJ,))  # here, equal to the semantic class of each instance ("size class")
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))  # difference between mean size of the semantic class ("size class") and actual bbox size

        ref_box_label = np.zeros(MAX_NUM_OBJ)  # bbox label for reference target
        ref_center_label = np.zeros(3)  # bbox center for reference target
        ref_heading_class_label = 0  # always zero
        ref_heading_residual_label = 0  # always zero
        ref_size_class_label = 0  # equal to the semantic class, see "size_classes" above
        ref_size_residual_label = np.zeros(3)  # bbox size residual for reference target
        ref_box_corner_label = np.zeros((8, 3))  # corners of the bbox

        num_bbox = 1
        point_votes = np.zeros([num_points, 3])
        point_votes_mask = np.zeros(num_points)

        # Calculate bboxes:
        num_bbox = instance_bboxes.shape[0] if instance_bboxes.shape[0] < MAX_NUM_OBJ else MAX_NUM_OBJ
        target_bboxes_mask[0:num_bbox] = 1
        target_bboxes[0:num_bbox, :] = instance_bboxes[:MAX_NUM_OBJ, 0:6]

        # ------------------------------- DATA AUGMENTATION ------------------------------
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:, 0] = -1 * point_cloud[:, 0]
                target_bboxes[:, 0] = -1 * target_bboxes[:, 0]

            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                point_cloud[:, 1] = -1 * point_cloud[:, 1]
                target_bboxes[:, 1] = -1 * target_bboxes[:, 1]

                # Rotation along X-axis
            rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
            rot_mat = rotx(rot_angle)
            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "x")

            # Rotation along Y-axis
            rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
            rot_mat = roty(rot_angle)
            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "y")

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
            rot_mat = rotz(rot_angle)
            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "z")

            # Translation
            # point_cloud, target_bboxes = self._translate(point_cloud, target_bboxes)

        # compute votes *AFTER* augmentation
        # generate votes
        # Note: since there's no map between bbox instance labels and
        # pc instance_labels (it had been filtered
        # in the data preparation step) we'll compute the instance bbox
        # from the points sharing the same instance label.
        for i_instance in np.unique(instance_labels):
            # find all points belong to that instance
            ind = np.where(instance_labels == i_instance)[0]
            # find the semantic label
            if semantic_labels[ind[0]] in DC.nyu40ids:
                x = point_cloud[ind, :3]
                center = 0.5 * (x.min(0) + x.max(0))
                point_votes[ind, :] = center - x
                point_votes_mask[ind] = 1.0
        point_votes = np.tile(point_votes, (1, 3))  # make 3 votes identical

        class_ind = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:num_bbox, -2]]
        # NOTE: set size class as semantic class. Consider use size2class.
        size_classes[0:num_bbox] = class_ind
        size_residuals[0:num_bbox, :] = target_bboxes[0:num_bbox, 3:6] - DC.mean_size_arr[class_ind, :]

        # construct the reference target label for each bbox
        for i, gt_id in enumerate(instance_bboxes[:num_bbox, -1]):
            if gt_id == object_id:
                ref_box_label[i] = 1
                ref_center_label = target_bboxes[i, 0:3]
                ref_heading_class_label = angle_classes[i]
                ref_heading_residual_label = angle_residuals[i]
                ref_size_class_label = size_classes[i]
                ref_size_residual_label = size_residuals[i]

                # construct ground truth box corner coordinates
                ref_obb = DC.param2obb(ref_center_label, ref_heading_class_label, ref_heading_residual_label,
                                       ref_size_class_label, ref_size_residual_label)
                ref_box_corner_label = get_3d_box(ref_obb[3:6], ref_obb[6], ref_obb[0:3])

        # construct all GT bbox corners
        all_obb = DC.param2obb_batch(target_bboxes[:num_bbox, 0:3], angle_classes[:num_bbox].astype(np.int64),
                                     angle_residuals[:num_bbox],
                                     size_classes[:num_bbox].astype(np.int64), size_residuals[:num_bbox])
        all_box_corner_label = get_3d_box_batch(all_obb[:, 3:6], all_obb[:, 6], all_obb[:, 0:3])

        # store
        gt_box_corner_label = np.zeros((MAX_NUM_OBJ, 8, 3))
        gt_box_masks = np.zeros((MAX_NUM_OBJ,))
        gt_box_object_ids = np.zeros((MAX_NUM_OBJ,))

        gt_box_corner_label[:num_bbox] = all_box_corner_label
        gt_box_masks[:num_bbox] = 1
        gt_box_object_ids[:num_bbox] = instance_bboxes[:, -1]

        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        target_object_ids = np.zeros((MAX_NUM_OBJ,))  # object ids of all objects
        try:
            target_bboxes_semcls[0:num_bbox] = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:, -2][0:num_bbox]]
            target_object_ids[0:num_bbox] = instance_bboxes[:, -1][0:num_bbox]
        except KeyError:
            pass

        object_cat = self.raw2label[object_name] if object_name in self.raw2label else 17

        # object rotations
        scene_object_rotations = np.zeros((MAX_NUM_OBJ, 3, 3))
        scene_object_rotation_masks = np.zeros((MAX_NUM_OBJ,))  # NOTE this is not object mask!!!
        # if scene is not in scan2cad annotations, skip
        # if the instance is not in scan2cad annotations, skip
        if self.scan2cad_rotation and scene_id in self.scan2cad_rotation:
            for i, instance_id in enumerate(instance_bboxes[:num_bbox, -1].astype(int)):
                try:
                    rotation = np.array(self.scan2cad_rotation[scene_id][str(instance_id)])

                    scene_object_rotations[i] = rotation
                    scene_object_rotation_masks[i] = 1
                except KeyError:
                    pass

        data_dict = {}

        # point cloud data including features: (NUM_POINTS, 3/6/9/10) depending on whether colors/normals/height are used.
        # (x,y,z, r,g,b, nx,ny,nz, h), RGB values are normalized here but not in pcl_color
        data_dict["point_clouds"] = point_cloud.astype(np.float32)
        data_dict["pcl_color"] = pcl_color  # non-normalized RGB: (NUM_POINTS, 3)

        data_dict['mean_xyz'] = MEAN_XYZ

        data_dict['semantic_labels'] = semantic_labels.astype(np.int64)
        data_dict['instance_labels'] = instance_labels.astype(np.int64)

        data_dict["lang_feat"] = lang_feat.astype(np.float32)  # description tokens' glove embeddings: (LEN_DESCRIPTION, 300)
        data_dict["lang_len"] = np.array(lang_len).astype(np.int64)  # length of each description: int
        data_dict["lang_ids"] = np.array(self.lang_ids[scene_id][str(object_id)][ann_id]).astype(np.int64)  # ID of every token in description: (LEN_DESCRIPTION,)

        data_dict["center_label"] = target_bboxes.astype(np.float32)[:, 0:3]  # (MAX_NUM_OBJ, 3) ground truth box center XYZ

        # === These are always zero for us ===
        data_dict["heading_class_label"] = angle_classes.astype(np.int64)  # (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
        data_dict["heading_residual_label"] = angle_residuals.astype(np.float32)  # (MAX_NUM_OBJ,)
        # ======

        data_dict["size_class_label"] = size_classes.astype(np.int64)  # (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER. See declaration above.
        data_dict["size_residual_label"] = size_residuals.astype(np.float32)  # (MAX_NUM_OBJ, 3). See declaration above.

        data_dict["num_bbox"] = np.array(num_bbox).astype(np.int64)
        data_dict["sem_cls_label"] = target_bboxes_semcls.astype(np.int64)  # (MAX_NUM_OBJ,) semantic class index

        data_dict["scene_object_ids"] = target_object_ids.astype(np.int64)  # (MAX_NUM_OBJ,) object ids of all objects

        data_dict["scene_object_rotations"] = scene_object_rotations.astype(np.float32)  # (MAX_NUM_OBJ, 3, 3)
        data_dict["scene_object_rotation_masks"] = scene_object_rotation_masks.astype(np.int64)  # (MAX_NUM_OBJ)

        data_dict["box_label_mask"] = target_bboxes_mask.astype(
            np.float32)  # (MAX_NUM_OBJ)  0/1 with 1 indicating a valid object's box

        data_dict["vote_label"] = point_votes.astype(np.float32)
        data_dict["vote_label_mask"] = point_votes_mask.astype(np.int64)

        data_dict["dataset_idx"] = np.array(idx).astype(np.int64)

        data_dict["ref_box_label"] = ref_box_label.astype(np.int64)  # mask: 1 for the index of the instance_bbox which is the ref object
        data_dict["ref_center_label"] = ref_center_label.astype(np.float32)  # center of the ref object: (3,)
        data_dict["ref_heading_class_label"] = np.array(int(ref_heading_class_label)).astype(np.int64)  # int
        data_dict["ref_heading_residual_label"] = np.array(int(ref_heading_residual_label)).astype(np.int64)  # int
        data_dict["ref_size_class_label"] = np.array(int(ref_size_class_label)).astype(np.int64)  # int
        data_dict["ref_size_residual_label"] = ref_size_residual_label.astype(np.float32)  # (3,)
        data_dict["ref_box_corner_label"] = ref_box_corner_label.astype(np.float64)  # target box corners, NOTE type must be double

        data_dict["gt_box_corner_label"] = gt_box_corner_label.astype(np.float64)  # all ground truth box corners, NOTE type must be double, (MAX_NUM_OBJ, 8, 3)
        data_dict["gt_box_masks"] = gt_box_masks.astype(np.int64)  # valid bbox masks, (MAX_NUM_OBJ,)
        data_dict["gt_box_object_ids"] = gt_box_object_ids.astype(np.int64)  # valid bbox object ids, (MAX_NUM_OBJ,)

        data_dict["object_id"] = np.array(int(object_id)).astype(np.int64)  # ID of the ref object
        data_dict["ann_id"] = np.array(int(ann_id)).astype(np.int64)  # annotation id of this sample
        data_dict["object_cat"] = np.array(object_cat).astype(np.int64)  # ref object category
        data_dict["unique_multiple"] = np.array(self.unique_multiple_lookup[scene_id][str(object_id)][ann_id]).astype(
            np.int64)

        data_dict["load_time"] = time.time() - start

        return data_dict


class ScannetReferenceTestDataset():

    def __init__(self, scanrefer_all_scene,
                 num_points=40000,
                 use_height=False,
                 use_color=False,
                 use_normal=False,
                 use_multiview=False):

        self.scanrefer_all_scene = scanrefer_all_scene  # all scene_ids in scanrefer
        self.num_points = num_points
        self.use_color = use_color
        self.use_height = use_height
        self.use_normal = use_normal
        self.use_multiview = use_multiview

        # load data
        self.scene_data = self._load_data()
        self.glove = pickle.load(open(GLOVE_PICKLE, "rb"))
        # self.vocabulary = json.load(open(SCANREFER_VOCAB))
        self.multiview_data = {}

    def __len__(self):
        return len(self.scanrefer_all_scene)

    def __getitem__(self, idx):
        start = time.time()

        scene_id = self.scanrefer_all_scene[idx]

        # get pc
        mesh_vertices = self.scene_data[scene_id]["mesh_vertices"]

        if not self.use_color:
            point_cloud = mesh_vertices[:, 0:3]  # do not use color for now
            pcl_color = mesh_vertices[:, 3:6]
        else:
            point_cloud = mesh_vertices[:, 0:6]
            point_cloud[:, 3:6] = (point_cloud[:, 3:6] - MEAN_COLOR_RGB) / 256.0
            pcl_color = point_cloud[:, 3:6]

        if self.use_normal:
            normals = mesh_vertices[:, 6:9]
            point_cloud = np.concatenate([point_cloud, normals], 1)

        # if self.use_multiview:
        #     # load multiview database
        #     pid = mp.current_process().pid
        #     if pid not in self.multiview_data:
        #         self.multiview_data[pid] = h5py.File(MULTIVIEW_DATA, "r", libver="latest")
        #
        #     multiview = self.multiview_data[pid][scene_id]
        #     point_cloud = np.concatenate([point_cloud, multiview], 1)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)

        # point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)

        data_dict = {}
        data_dict["point_clouds"] = point_cloud.astype(np.float32)  # point cloud data including features
        data_dict["dataset_idx"] = idx
        data_dict["lang_feat"] = self.glove["sos"].astype(np.float32)  # GloVE embedding for sos token
        data_dict["load_time"] = time.time() - start

        return data_dict

    def _load_data(self):
        scene_data = {}
        for scene_id in self.scanrefer_all_scene:
            scene_data[scene_id] = {}
            scene_data[scene_id]["mesh_vertices"] = np.load(
                os.path.join(CONF.PATH.SCANNET_DATA, scene_id) + "_aligned_vert.npy")  # axis-aligned

        return scene_data
