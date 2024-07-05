import os
import torch
import yaml
import numpy as np
from copy import deepcopy
from hyperopt import fmin, hp, tpe

from model.model import Encoder_GNN
from model.model import PredictModel
from train import main, parse_args
from utils.utils import create_logger

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def cross_validate(cfg,logger,dataset_name):
    init_seed = cfg.SEED
    out_dir = cfg.OUTPUT_DIR
    all_scores = []
    for fold_num in range(cfg.NUM_FOLDS):
        cfg.defrost()
        cfg.SEED = init_seed + fold_num
        cfg.OUTPUT_DIR = os.path.join(out_dir, f'fold_{fold_num}')
        cfg.freeze()
        model_scores = main(cfg,logger,dataset_name)
        all_scores.append(model_scores)
    all_scores = np.array(all_scores)

    cfg.defrost()
    cfg.OUTPUT_DIR = out_dir
    cfg.freeze()

    #avg_scores = np.nanmean(all_scores)
    mean_score, std_score = np.nanmean(all_scores), np.nanstd(all_scores)

    return mean_score, std_score

SPACE = {
    'DATA.BATCH_SIZE': hp.choice('bs', [32,64]),
    'MODEL.NUM_LAYER': hp.choice('num_layer', [6,8,12]),
    'MODEL.DROPOUT': hp.quniform('dropout', low=0.0, high=0.5, q=0.1),
    'TRAIN.OPTIMIZER.BASE_LR': hp.loguniform('lr', np.log(1e-4), np.log(1e-2)),
    'TRAIN.OPTIMIZER.WEIGHT_DECAY': hp.choice('l2', [1e-4, 1e-5, 1e-6]),
}

INT_KEYS = ['DATA.BATCH_SIZE', 'MODEL.NUM_LAYER']

def hyperopt(cfg,logger,dataset_name):
    yaml_name = "best_{}_{}.yaml".format(dataset_name, cfg.TAG)
    cfg_save_path = os.path.join(cfg.OUTPUT_DIR, yaml_name)

    results = []
    def objective(hyperparams):
        for key in INT_KEYS:
            hyperparams[key] = int(hyperparams[key])
        hyper_cfg = deepcopy(cfg)
        if hyper_cfg.OUTPUT_DIR is not None:
            folder_name = f'round_{hyper_cfg.HYPER_COUNT}'
            hyper_cfg.defrost()
            hyper_cfg.OUTPUT_DIR = os.path.join(hyper_cfg.OUTPUT_DIR, folder_name)
            hyper_cfg.freeze()

        hyper_cfg.defrost()
        opts = list()
        for key, value in hyperparams.items():
            opts.append(key)
            opts.append(value)
        hyper_cfg.merge_from_list(opts)
        hyper_cfg.freeze()

        cfg.defrost()
        cfg.HYPER_COUNT += 1
        cfg.freeze()

        mean_score, std_score = cross_validate(hyper_cfg,logger,dataset_name)

        result_name = 'mean_score.txt'
        mean_result_path = hyper_cfg.OUTPUT_DIR + result_name
        with open(mean_result_path,'a') as f:
            f.writelines([dataset_name+':', str(mean_score)])
        encoder = Encoder_GNN(cfg)
        temp_model = PredictModel(cfg,encoder=encoder)
        num_params = sum(param.numel() for param in temp_model.parameters() if param.requires_grad)

        results.append({
            'mean_score': mean_score,
            'std_score': std_score,
            'hyperparams': hyperparams,
            'num_params': num_params
        })

        return (-1 if hyper_cfg.DATA.TASK_TYPE == 'classification' else 1) * mean_score

    fmin(objective, SPACE, algo=tpe.suggest, max_evals=cfg.NUM_ITERS, verbose=False)

    results = [result for result in results if not np.isnan(result['mean_score'])]
    best_result = \
        min(results, key=lambda result: (-1 if cfg.DATA.TASK_TYPE == 'classification' else 1) * result['mean_score'])

    with open(cfg_save_path, 'w') as f:
        yaml.dump(best_result['hyperparams'], f, indent=4, sort_keys=True)

if __name__ == '__main__':
    dataset_list = ['clintox','sider','tox21','toxcast']
    for dataset_name in dataset_list:
    	_, cfg = parse_args(dataset_name)

    	logger = create_logger(cfg)
    	# print config
    	logger.info(cfg.dump())
    	# print device mode
    	if torch.cuda.is_available():
        	logger.info('GPU mode...')
    	else:
        	logger.info('CPU mode...')
    	if cfg.HYPER:
        	hyperopt(cfg,logger,dataset_name)

