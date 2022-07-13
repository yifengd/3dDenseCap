# import glob
# import os
#
# import torch
# from torch.utils.data import Dataset
#
# from dataset.dataset_config import get_scanrefer
#
#
# class ScanNetDataset(Dataset):
#     def __init__(self):
#         self.data_root = "data/"
#         self.dataset = "scannetv2"
#         self.split = "train"
#         self.filename_suffix = ".pth"
#
#         self.train_file_names = sorted(
#             glob.glob(os.path.join(self.data_root, self.dataset, self.split, '*' + self.filename_suffix)))
#
#     def __len__(self):
#         return len(self.train_file_names)
#
#     def __getitem__(self, idx):
#         coords, colors, sem_labels, instance_labels = torch.load(self.train_file_names[idx])
#         return {
#             "coords": coords,
#             "colors": colors,
#             "sem_labels": sem_labels,
#             "instance_labels": instance_labels
#         }
#
#     def get_scene_by_id(self, scene_id):
#         coords, colors, sem_labels, instance_labels = torch.load(
#             os.path.join(self.data_root, self.dataset, self.split, scene_id + "_inst_nostuff" + self.filename_suffix))
#
#         return {
#             "coords": coords,
#             "colors": colors,
#             "sem_labels": sem_labels,
#             "instance_labels": instance_labels
#         }
#
#
# class DenseCapDataset(Dataset):
#     def __init__(self, scannet: ScanNetDataset):
#         self.scannet = scannet
#         self.scanrefer = get_scanrefer()[0]  # train
#
#     def __len__(self):
#         return len(self.scanrefer)
#
#     def __getitem__(self, idx):
#         # ==== ScanRefer ====
#
#         scr = self.scanrefer[idx]
#
#         data_dict = {
#             "scene_id": scr["scene_id"],
#             "object_id": scr["object_id"],
#             "object_name": scr["object_name"],
#             "ann_id": scr["ann_id"],
#             "description": scr["description"],
#             "token": scr["token"]
#         }
#
#         # ==== ScanNet ====
#
#         scnnet = self.scannet.get_scene_by_id(data_dict['scene_id'])
#
#         data_dict['coords'] = scnnet['coords']
#         data_dict['colors'] = scnnet['colors']
#         data_dict['sem_labels'] = scnnet['sem_labels']
#         data_dict['instance_labels'] = scnnet['instance_labels']
#
#         # ==== Language features ====
#
#         return data_dict
#
#
