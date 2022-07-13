import torch
from torch.utils.data import Dataset

from model.caption_module import select_target


class RLScanCapSceneDataset(Dataset):
    def __init__(self, split):
        self.split = split

        if split == "train":
            self.data = torch.load("data/detection_outputs_train.pth", map_location="cpu")
        elif split == "val":
            self.data = torch.load("data/detection_outputs_val.pth", map_location="cpu")
        else:
            raise ValueError("Unknown split")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, scene_id):
        return self.data[scene_id]


class RLScanCapDataset(Dataset):
    def __init__(self, scanrefer, split):
        self.scanrefer = scanrefer
        self.prepared_scene_data = RLScanCapSceneDataset(split)

    def __getitem__(self, idx):
        scene_id = self.scanrefer[idx]["scene_id"]
        object_id = int(self.scanrefer[idx]["object_id"])
        object_name = " ".join(self.scanrefer[idx]["object_name"].split("_"))
        ann_id = self.scanrefer[idx]["ann_id"]

        data_dict = self.prepared_scene_data[scene_id]

        ref_box_mask = data_dict["gt_box_object_ids"] == object_id
        index_of_ref_obj = ref_box_mask.squeeze().nonzero().item()

        data_dict["ref_box_label"] = ref_box_mask # mask: 1 for the index of the instance_bbox which is the ref object
        data_dict["ref_box_corner_label"] = data_dict['gt_box_corner_label'][0, index_of_ref_obj].unsqueeze(0)  # target box corners, NOTE type must be double

        gt_description = self.scanrefer[idx]["description"]
        feat_size = data_dict["bbox_feature"].shape[2]

        target_id, target_iou = select_target(data_dict)  # 1
        target_feats = torch.gather(data_dict["bbox_feature"], 1,
                                    target_id.view(1, 1, 1).repeat(1, 1, feat_size)).squeeze(1)  # batch_size, emb_size

        all_gt_descriptions = [sr["description"] for sr in self.scanrefer if sr["scene_id"] == scene_id and int(sr["object_id"]) == object_id]
        return data_dict, target_feats, all_gt_descriptions

    def __len__(self):
        return len(self.scanrefer)


