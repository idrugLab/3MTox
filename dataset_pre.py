import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import networkx as nx
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem
import random
import math
import copy

from graph import atom_attr,bond_attr

class My_Pre_Dataset(Dataset):
    def __init__(self,cfg):
        path_gnn = cfg.DATA.DATA_PATH_GNN
        lists_gnn = self.gen_lists(path_gnn)
        augment_motifs_n,fp_list,augment_motifs_s, all_adj = lists_gnn[0],lists_gnn[1],lists_gnn[2],lists_gnn[3]
        path_hm = cfg.DATA.DATA_PATH_HM
        with open(path_hm, 'rb') as save_file:
            lists_hm = pickle.load(save_file)
        all_nodes = lists_hm[0]

        #self.all_motifs = all_motifs
        self.augment_motifs_n = augment_motifs_n
        self.fp_list = fp_list
        self.augment_motifs_s = augment_motifs_s
        self.all_adj = all_adj
        #self.augment_motifs_l = augment_motifs_l
        self.all_nodes = all_nodes
        self.max_len = cfg.DATA.MAX_LEN


    def __len__(self):
        return len(self.all_nodes)

    def __getitem__(self, item):
        nodes = self.all_nodes[item]
        X = [4750] + nodes
        y = copy.deepcopy(X)
        mask = np.zeros(len(y))
        mask = list(mask)
        #X = self.random_word(X)
        #motifs_attr = self.all_motifs[item]
        fp = self.fp_list[item]
        augment_attr_s = self.augment_motifs_s[item]
        adj = self.all_adj[item]
        augment_attr_n = self.augment_motifs_n[item]
        temp = np.ones((len(augment_attr_s)+1, len(augment_attr_s)+1))
        temp[1:, 1:] = adj
        adj = temp
        adj = np.pad(adj,((0,self.max_len - len(augment_attr_s)),(0,self.max_len - len(augment_attr_s))),'constant')
        adj = (1 - adj) * (-1e9)
        #motifs_attr = np.pad(motifs_attr,((1,self.max_len-len(motifs_attr)),(0,1)),'constant')
        #motifs_attr[0,56] = 1.0
        augment_attr_s = np.pad(augment_attr_s,((1,self.max_len-len(augment_attr_s)),(0,1)),'constant')
        augment_attr_s[0, 56] = 1.0
        augment_attr_n = np.pad(augment_attr_n,((1,self.max_len-len(augment_attr_n)),(0,1)),'constant')
        augment_attr_n[0, 56] = 1.0
        padding = [0] * (self.max_len - len(nodes))
        padding_mask = [1] * (self.max_len - len(nodes))
        #X.extend(padding)
        y.extend(padding)
        mask.extend(padding_mask)

        output = {
            'bert_augment_s':augment_attr_s,
            'bert_adj': adj,
            'bert_augment_n':augment_attr_n,
            'bert_fp':fp,
            'bert_label':y,
            'bert_mask':mask

        }
        return {key: torch.tensor(value) for key, value in output.items()}

    def gen_lists(self,path):
        with open(path, 'rb') as save_file:
            smiles = pickle.load(save_file)
        smi_len = len(smiles)
        #print(smi_len)
        #all_motifs = []
        all_adj = []
        augment_motifs_s = []
        augment_motifs_n = []
        fp_list = []
        #augment_motifs_l = []
        for g in range(smi_len):
            smi = smiles[g]
            mol = Chem.MolFromSmiles(smi)

            clique_list = []
            augment_list_s = []
            #augment_list_l = []
            if mol is not None:
                atoms_attr = atom_attr(mol)
                nx_graph = nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol))
                mcb = nx.cycle_basis(nx_graph)

                edges = []
                for e in nx_graph.edges():
                    count = 0
                    for c in mcb:
                        if e[0] in set(c) and e[1] in set(c):
                            count += 1
                            break
                    if count == 0:
                        edges.append(e)

                for e in edges:
                    motif_attr = 0
                    for i in e:
                        motif_attr += atoms_attr[i, :]
                    edge_attr = bond_attr(mol, e).flatten()
                    motif_attr = np.concatenate((motif_attr, edge_attr))
                    clique_list.append(motif_attr)

                for r in mcb:
                    motif_attr = 0
                    for i in r:
                        motif_attr += atoms_attr[i, :]
                    edge_attr = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
                    edge_attr = np.array(edge_attr)
                    motif_attr = np.concatenate((motif_attr, edge_attr))
                    clique_list.append(motif_attr)

                if len(clique_list) > 0:
                    fp = MACCSkeys.GenMACCSKeys(mol)
                    #fp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)
                    N = len(clique_list)
                    num_mask_motifs_s = max([1, math.floor(0.15 * N)])
                    #num_mask_motifs_l = max([1, math.floor(0.30 * N)])
                    mask_motifs_s = random.sample(list(range(N)), num_mask_motifs_s)
                    #mask_motifs_l = random.sample(list(range(N)), num_mask_motifs_l)
                    for id, data in enumerate(clique_list):
                        if id in mask_motifs_s:
                            motif_attr = np.zeros(56)
                            augment_list_s.append(motif_attr)
                        else:
                            augment_list_s.append(data)
                    #for id, data in enumerate(clique_list):
                        #if id in mask_motifs_l:
                            #motif_attr = np.zeros(56)
                            #augment_list_l.append(motif_attr)
                        #else:
                            #augment_list_l.append(data)
                    augment_motifs_n.append(clique_list)
                    fp_list.append(list(fp))
                    augment_motifs_s.append(augment_list_s)
                    #augment_motifs_l.append(augment_list_l)

                    motifs = mcb + edges
                    adj = np.eye(len(clique_list))
                    for m in range(len(motifs)):
                        for i in range(m + 1, len(motifs)):
                            for j in motifs[m]:
                                if j in motifs[i]:
                                    adj[m, i] = 1.0
                                    adj[i, m] = 1.0
                    all_adj.append(adj)
        lists = [augment_motifs_n,fp_list,augment_motifs_s, all_adj]
        return lists

