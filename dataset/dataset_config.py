import json
import os

DATA_PATH = "data/"


def get_simple_scanrefer() -> (list, list):
    scanrefer_train = json.load(open(os.path.join(DATA_PATH, "scanrefer", "ScanRefer_filtered_train.json")))
    scanrefer_eval_val = json.load(open(os.path.join(DATA_PATH, "scanrefer", "ScanRefer_filtered_val.json")))

    return scanrefer_train, scanrefer_eval_val
