import torch
from torch.utils.data import Dataset
import numpy as np
import pickle

class My_Dataset(Dataset):
    def __init__(self,cfg):
        path = cfg.DATA.DATA_PATH
        with open(path, 'rb') as save_file:
            lists = pickle.load(save_file)
        all_nodes,all_adj,labels = lists[0],lists[1],lists[2]

        self.all_nodes = all_nodes
        self.all_adj = all_adj
        self.y = labels
        self.max_len = cfg.DATA.MAX_LEN


    def __len__(self):
        return len(self.all_nodes)

    def __getitem__(self, item):

        motifs_attr = self.all_nodes[item]
        mask = np.zeros(len(motifs_attr)+1)
        mask = list(mask)
        padding = [1] * (self.max_len - len(motifs_attr))
        adj = self.all_adj[item]
        y = self.y[item]
        temp = np.ones((len(motifs_attr)+1, len(motifs_attr)+1))
        temp[1:, 1:] = adj
        adj = temp
        adj = np.pad(adj,((0,self.max_len - len(motifs_attr)),(0,self.max_len - len(motifs_attr))),'constant')
        adj = (1 - adj) * (-1e9)
        #adj = adj[:100,:100]
        motifs_attr = np.pad(motifs_attr,((1,self.max_len-len(motifs_attr)),(0,1)),'constant')
        motifs_attr[0,56] = 1.0
        #motifs_attr = motifs_attr[:100]
        mask.extend(padding)

        output = {
            'bert_input':motifs_attr,
            'bert_adj': adj,
            'bert_label':y,
            'bert_mask':mask
                     

        }
        return {key: torch.tensor(value) for key, value in output.items()}

