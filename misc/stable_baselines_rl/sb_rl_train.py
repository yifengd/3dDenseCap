import json
import os
import pickle
from copy import deepcopy

import gym
import stable_baselines3
import torch
from easydict import EasyDict
from gym import spaces

from config.scan2cap_config import CONF
from data.scannet.model_util_scannet import ScannetDatasetConfig
from misc.stable_baselines_rl.rl_dataset import RLScanCapDataset
import lib.capeval.meteor.meteor as capmeteor


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

class ScanCapEnv(gym.Env):
    def __init__(self, scanrefer, features_size, embedding_size, vocab_size, vocabulary, glove):
        self.data = RLScanCapDataset(scanrefer, "train")
        self.sample_idx = -1
        self.vocabulary = vocabulary
        self.glove = glove

        self.reward_average = None
        self.reward_variance = None

        self.meteor = capmeteor.Meteor()

        self.observation_space = spaces.Dict({
            "input_features": spaces.Box(shape=[1, features_size], high=torch.inf, low=-torch.inf),
            # "bbox_feature": spaces.Box(shape=[1, num_proposals, embedding_size], high=torch.inf, low=-torch.inf),
            # "bbox_corner": spaces.Box(shape=[1, num_proposals, 8, 3], high=torch.inf, low=-torch.inf),
            # "ref_box_corner_label": spaces.Box(shape=[1, 8, 3], high=torch.inf, low=-torch.inf),
            
            # "hidden_state": spaces.Box(shape=[hidden_size], high=torch.inf, low=-torch.inf),
            # "previous_word": spaces.Discrete(vocab_size),
            "previous_word_embedding": spaces.Box(shape=[300], high=torch.inf, low=-torch.inf)
        })

        self.action_space = spaces.Discrete(vocab_size)

        # self.action_space = spaces.Dict({
        #     "next_word": spaces.Discrete(vocab_size),
        #     # "hidden_state": spaces.Box(shape=[hidden_size], high=torch.inf, low=-torch.inf),
        # })

    def reset(self, seed=None, return_info=False, options=None):
        self.sample_idx += 1
        self.sample_data, self.input_features, self.current_gt_descriptions = self.data[self.sample_idx % 1] # len(self.data)]
        self.current_pred_description = []

        return {
            "input_features": self.input_features.cpu(),

            # "hidden_state": hidden_state,
            # "previous_word": 2,  # sos
            "previous_word_embedding": self.glove['sos']
        }

    def step(self, action: int):
        next_word = action

        self.current_pred_description.append(next_word)

        done = next_word == 3 or len(self.current_pred_description) > 64  # eos

        if done:
            decoded_pred = decode_caption(torch.tensor(self.current_pred_description), self.vocabulary["idx2word"])
            decoded_gt = ["sos " + sent + " eos" for sent in self.current_gt_descriptions]
            candidate = {
                "key": [decoded_pred]
            }
            gt = {
                "key": decoded_gt
            }
            # print(decoded_pred)
            # cider, _ = capcider.Cider().compute_score(gt, candidate)
            # rouge, _ = caprouge.Rouge().compute_score(gt, candidate)
            # bleu = capbleu.Bleu(4).compute_score(gt, candidate)[0][3]
            meteor, _ = self.meteor.compute_score(gt, candidate)

            reward = meteor

            # if self.reward_average is None:
            #     self.reward_average = reward
            # else:
            #     self.reward_average = 0.1 * reward + 0.9 * self.reward_average
            #
            # if self.reward_variance is None:
            #     self.reward_variance = (reward - self.reward_average) ** 2
            # else:
            #     self.reward_variance = 0.1 * ((reward - self.reward_average) ** 2) + 0.9 * self.reward_variance
            #
            # reward = reward - self.reward_average
            # reward = reward / np.sqrt(self.reward_variance + 1e-3)

            # if not done:
            #     reward = reward / 64
            # else:
            #     reward = (64 - len(self.current_pred_description)) * (reward / 64)

            if self.sample_idx % 1000 == 10:
                print(decoded_pred)
                print(decoded_gt)
                # print(rouge, bleu, cider)
            # print(reward)
        else:
            reward = 0.

        obs = {
            "input_features": self.input_features.cpu(),

            # "hidden_state": hidden_state,
            # "previous_word": next_word,
            "previous_word_embedding": self.glove[self.vocabulary["idx2word"][str(next_word)]],
        }

        return obs, reward, done, {}

    def render(self, mode="human"):
        pass

    def close(self):
        pass

SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.SCANREFER, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.SCANREFER, "ScanRefer_filtered_val.json")))

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


args = EasyDict()
args.batch_size = 1

DC = ScannetDatasetConfig()

scanrefer_train, scanrefer_eval_train, scanrefer_eval_val, all_scene_list = get_scanrefer()

VOCAB = os.path.join(CONF.PATH.DATA, "{}_vocabulary.json")  # dataset_name
vocab_path = VOCAB.format("ScanRefer")
vocabulary = json.load(open(vocab_path))
num_vocabs = len(vocabulary["word2idx"].keys())

glove = pickle.load(open("data/glove.p", "rb"))

# Parallel environments
env = ScanCapEnv(scanrefer_train[:300], 16, 300, num_vocabs, vocabulary, glove)

model = stable_baselines3.PPO("MultiInputPolicy", env, verbose=1, clip_range=0.4)
model.learn(total_timesteps=2500000)
model.save("RPPO_Policy")

del model # remove to demonstrate saving and loading

# model = A2C.load("a2c_policy")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
