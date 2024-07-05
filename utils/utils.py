import random
import os
import numpy as np
import torch
import math
import time
from collections import defaultdict
from rdkit import Chem
from termcolor import colored
import logging
from rdkit.Chem.Scaffolds import MurckoScaffold
from itertools import compress
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve, roc_auc_score


def seed_set(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def create_logger(cfg):
    dataset_name = cfg.DATA.DATASET_NAME
    tag_name = cfg.TAG
    time_str = time.strftime("%Y-%m-%d")
    log_name = "{}_{}_{}.log".format(dataset_name, tag_name, time_str)

    # log dir
    log_dir = os.path.join(cfg.OUTPUT_DIR, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # create logger
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = \
        colored('[%(asctime)s]', 'green') + \
        colored('(%(filename)s %(lineno)d): ', 'yellow') + \
        colored('%(levelname)-5s', 'magenta') + ' %(message)s'

    # create console handlers for master process
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(log_dir, log_name))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger

def build_optimizer(cfg,model):
    params = model.parameters()
    opt_lower = cfg.TRAIN.OPTIMIZER.TYPE.lower()
    optimizer = None

    if opt_lower == 'sgd':
        optimizer = torch.optim.SGD(
            params,
            lr=cfg.TRAIN.OPTIMIZER.BASE_LR,
            momentum=cfg.TRAIN.OPTIMIZER.MOMENTUM,
            weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY,
            nesterov=True
        )

    elif opt_lower == 'adam':
        optimizer = torch.optim.Adam(
            params,
            lr=cfg.TRAIN.OPTIMIZER.BASE_LR,
            weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY
        )
    return optimizer

def build_scheduler(cfg, optimizer, steps_per_epoch):
    if cfg.TRAIN.LR_SCHEDULER.TYPE == "reduce":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=cfg.TRAIN.LR_SCHEDULER.FACTOR,
            patience=cfg.TRAIN.LR_SCHEDULER.PATIENCE,
            min_lr=cfg.TRAIN.LR_SCHEDULER.MIN_LR
        )
    elif cfg.TRAIN.LR_SCHEDULER.TYPE == "noam":
        scheduler = NoamLR(
            optimizer,
            warmup_epochs=[cfg.TRAIN.LR_SCHEDULER.WARMUP_EPOCHS],
            total_epochs=[cfg.TRAIN.MAX_EPOCHS],
            steps_per_epoch=steps_per_epoch,
            init_lr=[cfg.TRAIN.LR_SCHEDULER.INIT_LR],
            max_lr=[cfg.TRAIN.LR_SCHEDULER.MAX_LR],
            final_lr=[cfg.TRAIN.LR_SCHEDULER.FINAL_LR]
        )
    else:
        raise NotImplementedError("Unsupported LR Scheduler: {}".format(cfg.TRAIN.LR_SCHEDULER.TYPE))

    return scheduler

class NoamLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, steps_per_epoch,
                 init_lr, max_lr, final_lr):

        assert len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs) == len(init_lr) == \
               len(max_lr) == len(final_lr)

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self):

        return list(self.lr)

    def step(self, current_step=None):

        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i]

def generate_scaffold(mol, include_chirality=False):
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

    return scaffold

#def scaffold_to_smiles(smiles, use_indices=False):
    #scaffolds = defaultdict(set)
    #for i, smi in enumerate(smiles):
        #scaffold = generate_scaffold(smi)
        #if use_indices:
            #scaffolds[scaffold].add(i)
        #else:
            #scaffolds[scaffold].add(smi)

    #return scaffolds

#def scaffold_split(smiles,train_size,val_size,test_size,balanced=True):
    #train_ids, val_ids, test_ids = [], [], []
    #train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    #scaffold_to_indices = scaffold_to_smiles(smiles, use_indices=True)

    #if balanced:
        #index_sets = list(scaffold_to_indices.values())
        #big_index_sets = []
        #small_index_sets = []
        #for index_set in index_sets:
            #if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                #big_index_sets.append(index_set)
            #else:
                #small_index_sets.append(index_set)
        #random.shuffle(big_index_sets)
        #random.shuffle(small_index_sets)
        #index_sets = big_index_sets + small_index_sets
    #else:
        #index_sets = sorted(list(scaffold_to_indices.values()),
                            #key=lambda index_set: len(index_set),
                            #reverse=True)
    #for index_set in index_sets:
        #if len(train_ids) + len(index_set) <= train_size:
            #train_ids += index_set
            #train_scaffold_count += 1
        #elif len(val_ids) + len(index_set) <= val_size:
            #val_ids += index_set
            #val_scaffold_count += 1
        #else:
            #test_ids += index_set
            #test_scaffold_count += 1

    #return train_ids, val_ids, test_ids

def scaffold_split(dataset, smiles_list, task_idx=None, null_value=0,
                   frac_train=0.8, frac_valid=0.1, frac_test=0.1,
                   return_smiles=False):
    """
    Adapted from  https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
    Split dataset by Bemis-Murcko scaffolds
    This function can also ignore examples containing null values for a
    selected task when splitting. Deterministic split
    :param dataset: pytorch geometric dataset obj
    :param smiles_list: list of smiles corresponding to the dataset obj
    :param task_idx: column idx of the data.y tensor. Will filter out
    examples with null value in specified task column of the data.y tensor
    prior to splitting. If None, then no filtering
    :param null_value: float that specifies null value in data.y to filter if
    task_idx is provided
    :param frac_train:
    :param frac_valid:
    :param frac_test:
    :param return_smiles:
    :return: train, valid, test slices of the input dataset obj. If
    return_smiles = True, also returns ([train_smiles_list],
    [valid_smiles_list], [test_smiles_list])
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if task_idx != None:
        # filter based on null values in task_idx
        # get task array
        y_task = np.array([data.y[task_idx].item() for data in dataset])
        # boolean array that correspond to non null values
        non_null = y_task != null_value
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    else:
        non_null = np.ones(len(dataset)) == 1
        smiles_list = list(compress(enumerate(smiles_list), non_null))

    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}
    for i, smiles in smiles_list:
        scaffold = generate_scaffold(smiles, include_chirality=True)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]

    # get train, valid test indices
    train_cutoff = frac_train * len(smiles_list)
    valid_cutoff = (frac_train + frac_valid) * len(smiles_list)
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    #train_dataset = dataset.iloc[train_idx]
    #valid_dataset = dataset.iloc[valid_idx]
    #test_dataset = dataset.iloc[test_idx]
    train_dataset = dataset[train_idx]
    valid_dataset = dataset[valid_idx]
    test_dataset = dataset[test_idx]

    if not return_smiles:
        return train_dataset, valid_dataset, test_dataset
    else:
        train_smiles = [smiles_list[i][1] for i in train_idx]
        valid_smiles = [smiles_list[i][1] for i in valid_idx]
        test_smiles = [smiles_list[i][1] for i in test_idx]
        return train_dataset, valid_dataset, test_dataset, (train_smiles,
                                                            valid_smiles,
                                                            test_smiles)

def prc_auc_score(targets,preds):
    precision,recall,_ = precision_recall_curve(targets,preds)
    return auc(recall,precision)

def rmse(targets,preds):
    return math.sqrt(mean_squared_error(targets,preds))

def get_metric_func(metric):
    if metric == 'auc':
        return roc_auc_score

    if metric == 'prc':
        return prc_auc_score

    if metric == 'mae':
        return mean_absolute_error

    if metric == 'mse':
        return mean_squared_error

    if metric == 'rmse':
        return rmse

    raise ValueError(f'Metric "{metric}" not supported.')