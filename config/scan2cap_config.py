import os
import sys
from easydict import EasyDict

CONF = EasyDict()

# path
CONF.PATH = EasyDict()
CONF.PATH.BASE = ""
CONF.PATH.DATA = "data/"
CONF.PATH.SCANNET = os.path.join(CONF.PATH.BASE, CONF.PATH.DATA, "scannet")
CONF.PATH.SCANREFER = os.path.join(CONF.PATH.BASE, CONF.PATH.DATA, "scanrefer")

# append to syspath
for _, path in CONF.PATH.items():
    sys.path.append(path)

# scannet data
CONF.PATH.SCANNET_SCANS = os.path.join(CONF.PATH.SCANNET, "scans")
CONF.PATH.SCANNET_META = os.path.join(CONF.PATH.SCANNET, "meta_data")
CONF.PATH.SCANNET_DATA = os.path.join(CONF.PATH.SCANNET, "scannet_data")

# Scan2CAD
CONF.PATH.SCAN2CAD = os.path.join(CONF.PATH.DATA, "Scan2CAD_dataset")

# output
CONF.PATH.OUTPUT = os.path.join(CONF.PATH.BASE, "outputs")
CONF.PATH.AXIS_ALIGNED_MESH = os.path.join(CONF.PATH.OUTPUT, "ScanNet_axis_aligned_mesh")

# pretrained
CONF.PATH.PRETRAINED = os.path.join(CONF.PATH.BASE, "pretrained")



# eval
CONF.EVAL = EasyDict()
CONF.EVAL.MIN_IOU_THRESHOLD = 0.5


##pointgroup config

CONF.PG = EasyDict()

CONF.PG.SCALE = 50
CONF.PG.FULL_SCALE = [128, 512]
CONF.PG.MODE = 4 #4=mean
CONF.PG.PREPARE_EPOCHS = -1

CONF.PG.FG_THRESH = 0.75
CONF.PG.BG_THRESH = 0.25

CONF.PG.LOSS_WEIGHTS = [1.0, 1.0, 1.0, 1.0] # semantic_loss, offset_norm_loss, offset_dir_loss, score_loss

CONF.PG.PRETRAIN = ""

CONF.PG.INPUT_C = 3

CONF.PG.M = 32
CONF.PG.UBLOCK_LAYERS = [1, 2, 3, 4, 5, 6, 5, 4, 3, 3, 3]  # Multiples of PG.M
CONF.PG.CLUSTER_NPOINT_THRE = 50
CONF.PG.CLUSTER_RADIUS = 0.03
CONF.PG.MEAN_ACTIVE = 50
CONF.PG.SHIFT_MEAN_ACTIVE = 300
CONF.PG.SCORE_SCALE = 50
CONF.PG.SCORE_FULLSCALE = 14
CONF.PG.SCORE_MODE = 4
CONF.PG.BLOCK_REPS = 2
CONF.PG.BLOCK_RESIDUAL = True
CONF.PG.FIX_MODULE = []

#training config
CONF.TRAIN = EasyDict()
CONF.TRAIN.MAX_DES_LEN = 30
CONF.TRAIN.SEED = 42
CONF.TRAIN.OVERLAID_THRESHOLD = 0.5
CONF.TRAIN.MIN_IOU_THRESHOLD = 0.25
CONF.TRAIN.NUM_BINS = 6

CONF.MAX_N_POINT = 100000
CONF.PROPOSAL_SCORE_THRE = 0.05

CONF.TASK = 'train'

CONF.NUM_WORKERS = 4

CONF.DATASET = 'ScanRefer'
CONF.GPU = "0"
CONF.SEED = 42
CONF.BATCH_SIZE = 5
CONF.NUM_EPOCH = 64
CONF.VERBOSE = 10 #print every 10 epoch
CONF.VAL_STEP = 1000 #validate every 2000 epoch
CONF.LR = 5e-4
CONF.WD = 1e-5
CONF.NUM_PROPOSALS = 512
CONF.NUM_LOCALS = 10
CONF.NUM_GRAPH_STEPS = 2
CONF.CRITERION = 'cider' #bleu-1, bleu-2, bleu-3, bleu-4, cider, rouge, meteor, sum
CONF.QUERY_MODE = 'center' #center, corner
CONF.GRAPH_MODE = 'edge_conv' #graph_conv, edge_conv
CONF.GRAPH_AGGR = 'add' #add, mean, max
CONF.NO_HEIGHT = False
CONF.NO_AUGMENT = False
CONF.NO_DETECTION = False
CONF.NO_CAPTION = False
CONF.USE_TF = False
CONF.USE_COLOR = True
CONF.USE_NORMAL = True
CONF.USE_MULTIIEW = False
CONF.USE_TOPDOWN = True
CONF.USE_RELATION = True
CONF.USE_NEW = False
CONF.USE_ORIENTATION = True
CONF.USE_DISTANCE = False
CONF.USE_PRETRAINED = None
CONF.USE_CHECKPOINT = ""
CONF.DEBUG = False
CONF.AUGMENT_DATA = False

CONF.TRAIN_EVAL_LIMIT = 10

CONF.NUM_SCENES = None #set None to train on whole dataset

CONF.REINFORCE_STEPS = 200
CONF.REINFORCE_GAMMA = 0.99