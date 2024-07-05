import os
import random
import numpy as np
import pandas as pd
import time
import datetime
import pickle
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse

from dataset.dataset_clr import My_Dataset
from model.model import Encoder_GNN
from model.model import PredictModel
from config.config import get_config
from utils.utils import seed_set,build_optimizer,get_metric_func,create_logger,scaffold_split

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def parse_args(dataset):
    parser = argparse.ArgumentParser(description="codes for HM-BERT")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="./config/freesolv.yaml",
        type=str,
    )

    args = parser.parse_args()
    args.cfg = "./config/" + dataset + ".yaml"
    cfg = get_config(args)

    return args, cfg

def train(cfg,train_loader,model,optimizer,loss_fcn):
    total_loss = []
    model.train()
    for data in train_loader:
        loss = 0
        x1,adj,y = data['bert_input'].to(device),data['bert_adj'].to(device),data['bert_label'].to(device)
        mask = data['bert_mask'].to(device)
        #x2 = data['bert_input_hm'].to(device)
        #y = y.float()
        preds = model(x1,mask,adj)
        
        #if cfg.MODEL.NUM_TASK == 1:
            #preds = preds.view(-1)
            #if cfg.DATA.TASK_TYPE == 'classification':
                #preds = nn.Sigmoid()(preds)
            #loss += loss_fcn(preds, y)
        #else:
            #preds = nn.Sigmoid()(preds)
            #for i in range(cfg.MODEL.NUM_TASK):
                #label = y[:, i].squeeze()
                #pred = preds[:, i]
                #validId = np.where((label.cpu().numpy() == 0) | (label.cpu().numpy() == 1))[0]
                #if len(validId) == 0:
                    #continue
                #if label.dim() == 0:
                    #label = label.unsqueeze(0)
                #y_pred = pred[torch.tensor(validId).to(device)]
                #y_label = label[torch.tensor(validId).to(device)]
                #loss += loss_fcn(y_pred, y_label)

        if cfg.DATA.TASK_TYPE == 'classification':
            if not isinstance(y,torch.cuda.LongTensor):
                y = y.long()
            for i in range(cfg.MODEL.NUM_TASK):
                if cfg.MODEL.NUM_TASK == 1:
                    label = y
                    pred = preds[:, i * 2:(i + 1) * 2]
                else:
                    label = y[:, i].squeeze()
                    pred = preds[:, i * 2:(i + 1) * 2]
                validId = np.where((label.cpu().numpy() == 0) | (label.cpu().numpy() == 1))[0]
                if len(validId) == 0:
                    continue
                if label.dim() == 0:
                    label = label.unsqueeze(0)
                y_pred = pred[torch.tensor(validId).to(device)]
                y_label = label[torch.tensor(validId).to(device)]
                loss += loss_fcn(y_pred, y_label)

        else:
            y = y.float()
            preds = preds.view(-1)
            loss += loss_fcn(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
    return sum(total_loss)/len(train_loader)

def eval(cfg,test_loader,model,metric_func):
    model.eval()
    
    y_pred_list = {}
    y_label_list = {}
    for n in range(cfg.MODEL.NUM_TASK):
        y_pred_list[n] = torch.Tensor()
        y_label_list[n] = torch.Tensor()
    with torch.no_grad():
        for data in test_loader:
            x1,adj,y = data['bert_input'].to(device),data['bert_adj'].to(device),data['bert_label']
            mask = data['bert_mask'].to(device)
            #x2 = data['bert_input_hm'].to(device)
            preds = model(x1,mask,adj)
            #preds = preds.cpu()
            #if cfg.MODEL.NUM_TASK == 1:
                #if cfg.DATA.TASK_TYPE == 'classification':
                    #preds = nn.Sigmoid()(preds)
                #y_label_list[0] = torch.cat((y_label_list[0], y), 0)
                #y_pred_list[0] = torch.cat((y_pred_list[0], preds), 0)
            #else:
                #preds = nn.Sigmoid()(preds)
                #for i in range(cfg.MODEL.NUM_TASK):
                    #label = y[:, i].squeeze()
                    #pred = preds[:, i]
                    #validId = np.where((label.numpy() == 0) | (label.numpy() == 1))[0]
                    #if len(validId) == 0:
                        #continue
                    #if label.dim() == 0:
                        #label = label.unsqueeze(0)
                    #y_pred = pred[torch.tensor(validId)]
                    #y_label = label[torch.tensor(validId)]
                    #y_label_list[i] = torch.cat((y_label_list[i], y_label), 0)
                    #y_pred_list[i] = torch.cat((y_pred_list[i], y_pred), 0)
            if cfg.DATA.TASK_TYPE == 'classification':
                for i in range(cfg.MODEL.NUM_TASK):
                    if cfg.MODEL.NUM_TASK == 1:
                        label = y
                        pred = preds[:, i * 2:(i + 1) * 2]
                    else:
                        label = y[:, i].squeeze()
                        pred = preds[:, i * 2:(i + 1) * 2]
                    validId = np.where((label.cpu().numpy() == 0) | (label.cpu().numpy() == 1))[0]
                    if len(validId) == 0:
                        continue
                    if label.dim() == 0:
                        label = label.unsqueeze(0)
                    y_pred = pred[torch.tensor(validId)]
                    y_label = label[torch.tensor(validId)]
                    y_pred = F.softmax(y_pred.detach().cpu(), dim=-1)[:, 1].view(-1)
                    y_label_list[i] = torch.cat((y_label_list[i], y_label), 0)
                    y_pred_list[i] = torch.cat((y_pred_list[i], y_pred), 0)
            else:
                preds = preds.detach().cpu()
                y_label_list[0] = torch.cat((y_label_list[0], y), 0)
                y_pred_list[0] = torch.cat((y_pred_list[0], preds), 0)
        val_results = []
        for i in range(cfg.MODEL.NUM_TASK):
            if cfg.DATA.TASK_TYPE == 'classification':
                nan = False
                labels = y_label_list[i].numpy()
                preds = y_pred_list[i].numpy()
                if all(target == 0 for target in labels) or all(target == 1 for target in labels):
                    nan = True
                if nan:
                    val_results.append(float('nan'))
                    continue
            else:
                labels = y_label_list[i].numpy()
                preds = y_pred_list[i].numpy()
            if len(labels) == 0:
                continue
            score = metric_func(labels,preds)
            val_results.append(score)
        return np.nanmean(val_results)

def main(cfg,logger,dataset_name):
    best_score = 0
    stopping_monitor = 0
    model_file_name = './output/model_clr/'+dataset_name+'.pkl'
    logger.info(model_file_name)
    seed_set(cfg.SEED)

    dataset = My_Dataset(cfg)
    dataset = np.array(dataset)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    if cfg.DATA.SPLIT_TYPE == 'random':
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        trn_id, val_id, test_id = indices[:train_size],indices[train_size:(train_size + val_size)],indices[(train_size + val_size):]
        trn, val, test = dataset[trn_id],dataset[val_id],dataset[test_id]
    else:
        path = cfg.DATA.DATA_PATH
        with open(path, 'rb') as save_file:
            lists = pickle.load(save_file)
        smiles = lists[3]
        trn, val, test = scaffold_split(dataset,smiles)
        
    train_loader = DataLoader(trn, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True)

    encoder = Encoder_GNN(cfg)
    encoder.load_state_dict(torch.load('./output/model_pre/20_model_encoder_bert.pkl'),strict=False)
    model = PredictModel(cfg,encoder=encoder).to(device)
    optimizer = build_optimizer(cfg, model)
    metric_func = get_metric_func(cfg.DATA.METRIC)
    if cfg.DATA.TASK_TYPE == 'classification':
        #loss_fcn = nn.BCELoss()
        loss_fcn = nn.CrossEntropyLoss()
    else:
        loss_fcn = nn.MSELoss()

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(cfg.TRAIN.NUM_EPOCH):

        loss = train(cfg,train_loader,model,optimizer,loss_fcn)
        val_score = eval(cfg,val_loader,model,metric_func)
        
        if cfg.DATA.TASK_TYPE == 'classification':
            if val_score > best_score:
                best_score = val_score
                stopping_monitor = 0
                torch.save(model.state_dict(),model_file_name)
            else:
                stopping_monitor += 1
        elif cfg.DATA.TASK_TYPE == 'regression':
            if epoch == 0:
                best_score = val_score
            else:
                if val_score < best_score:
                    best_score = val_score
                    stopping_monitor = 0
                    torch.save(model.state_dict(),model_file_name)
                else:
                    stopping_monitor += 1
        if stopping_monitor > 20:
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f'Training time {total_time_str}')

    model.load_state_dict(torch.load(model_file_name))
    score = eval(cfg,test_loader,model,metric_func)
    return score

#if __name__ == '__main__':
    #dataset_list = ['freesolv','ESOL','lipo','bace','bace_scaffold','bbbp','bbbp_scaffold','clintox','sider','tox21']
    #all_scores = []
    #for dataset_name in dataset_list:
        #_, cfg = parse_args(dataset_name)

        #logger = create_logger(cfg)
        # print config
        #logger.info(cfg.dump())
        # print device mode
        #if torch.cuda.is_available():
            #logger.info('GPU mode...')
        #else:
            #logger.info('CPU mode...')
        #scores = []
        #for i in range(10):
            #cfg.defrost()
            #cfg.SEED = 2021+i
            #cfg.freeze()
            #score = main(cfg,logger,dataset_name)
            #scores.append(score)
        #all_scores.append(scores)
    #df = pd.DataFrame(all_scores,index=dataset_list)
    #df.to_csv('/share/home/liuchong/zyy/output/result.csv')
        
