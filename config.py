import os
import yaml
from yacs.config import CfgNode as CN


_C = CN()

_C.SEED = 2021
_C.OUTPUT_DIR = "./output"
_C.TAG = 'default_30_random'
_C.NUM_FOLDS = 10
_C.HYPER = True
_C.HYPER_COUNT = 1
_C.NUM_ITERS = 30

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()

_C.DATA.BATCH_SIZE = 64

_C.DATA.DATA_PATH_GNN = './dataset/raw/clintox.pkl'

_C.DATA.DATASET_NAME = 'chembl'

_C.DATA.VOCAB_SIZE = 4747

_C.DATA.MAX_LEN = 40

_C.DATA.TASK_TYPE = None

_C.DATA.METRIC = None

_C.DATA.SPLIT_TYPE = 'random'

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()

_C.MODEL.NUM_LAYER = 6

_C.MODEL.NUM_HEAD = 8

_C.MODEL.D_MODEL = 256

_C.MODEL.D_FF = 512

_C.MODEL.DROPOUT = 0.0

_C.MODEL.NUM_TASK = 23

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()

_C.TRAIN.NUM_EPOCH = 21

# Optimizer
_C.TRAIN.OPTIMIZER = CN()

_C.TRAIN.OPTIMIZER.TYPE = 'adam'

_C.TRAIN.OPTIMIZER.BASE_LR = 1e-4

_C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 1e-4

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.TYPE = 'reduce'
# NoamLR parameters
_C.TRAIN.LR_SCHEDULER.WARMUP_EPOCHS = 2.0
_C.TRAIN.LR_SCHEDULER.INIT_LR = 1e-4
_C.TRAIN.LR_SCHEDULER.MAX_LR = 1e-2
_C.TRAIN.LR_SCHEDULER.FINAL_LR = 1e-4
# ReduceLRonPlateau
_C.TRAIN.LR_SCHEDULER.FACTOR = 0.7
_C.TRAIN.LR_SCHEDULER.PATIENCE = 10
_C.TRAIN.LR_SCHEDULER.MIN_LR = 1e-5

def _update_config_from_file(config, cfg_file):
    config.defrost()
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(cfg, args):
    _update_config_from_file(cfg, args.cfg)

def get_config(args=None):
    cfg = _C.clone()
    if args:
        update_config(cfg,args)

    return cfg