import os
import time
import datetime
import torch
from model import Encoder_GNN
from model import BertModel
import torch.nn as nn
from dataset_pre import My_Pre_Dataset
from torch.utils.data import DataLoader
import argparse
from nt_xent import NT_Xent
from config import get_config
from utils import seed_set,build_optimizer,create_logger

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="codes for HM-BERT")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="./config/freesolv.yaml",
        type=str,
    )

    args = parser.parse_args()
    args.cfg = './config/pretrain.yaml'
    cfg = get_config(args)

    return args, cfg

def train(train_loader,model,optimizer):
    train_loss = 0
    model.train()
    for data in train_loader:
        x1,x2,adj = data['bert_augment_n'].to(device),data['bert_augment_s'].to(device),data['bert_adj'].to(device)
        fp = data['bert_fp'].to(device)
        y = data['bert_label'].to(device)
        mask = data['bert_mask'].to(device)
        #y = y.float()
        out1,out2 = model(x1,x2,mask,adj)
        criterion = NT_Xent(out1.shape[0], 0.1, 1)
        loss_fcn = nn.NLLLoss(ignore_index=0)
        loss = criterion(out1, fp) + loss_fcn(out2.transpose(1, 2),y)
      
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss/len(train_loader)

def main(cfg,logger):
    #seed_set(cfg.SEED)
    dataset = My_Pre_Dataset(cfg)
    
    encoder = Encoder_GNN(cfg)
    model = BertModel(cfg,encoder=encoder).to(device)
    optimizer = build_optimizer(cfg,model)

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(cfg.TRAIN.NUM_EPOCH):
        
        train_loader = DataLoader(dataset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True)
        loss = train(train_loader,model,optimizer)
        print(loss)

        if epoch in list(range(0,21,5)):
            torch.save(encoder.state_dict(),'./output/model_pre/'+str(epoch)+'_model_encoder_bert'+'.pkl')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f'Training time {total_time_str}')

if __name__ == '__main__':
    _, cfg = parse_args()

    logger = create_logger(cfg)
    # print config
    logger.info(cfg.dump())
    # print device mode
    if torch.cuda.is_available():
        logger.info('GPU mode...')
    else:
        logger.info('CPU mode...')
    main(cfg,logger)