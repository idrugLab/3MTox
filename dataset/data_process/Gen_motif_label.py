import operator
import math
import networkx as nx
import numpy as np
import pandas as pd
from rdkit import Chem
import copy
import os
from collections import defaultdict
import pickle

path = 'data/raw/all_datasets_select.pkl'
with open(path, 'rb') as save_file:
    dataset = pickle.load(save_file)
smi_length = len(dataset)
print(smi_length)
vocab = {}
all_nodes = []
all_adj = []

def add_to_vocab(clique):
    c = copy.deepcopy(clique[0])
    weight = copy.deepcopy(clique[1])
    for i in range(len(c)):
        if (c,weight) in vocab:
            return vocab[(c,weight)]
        else:
            c = shift_right(c)
            weight = shift_right(weight)
    vocab[(c,weight)] = len(list(vocab.keys())) + 1
    return vocab[(c,weight)]

def shift_right(l):
    if type(l) == int:
        return l
    elif type(l) == tuple:
        l = list(l)
        l = tuple([l[-1]] + l[:-1])
        return l
    elif type(l) == list:
        l = tuple([l[-1]] + l[:-1])
        return l
    else:
        print('ERROR')

def find_ring_weight(ring,g):
    weight_list = []
    for i in range(len(ring)-1):
        weight = g.get_edge_data(ring[i],ring[i+1])['label']
        weight_list.append(weight)
    weight = g.get_edge_data(ring[-1],ring[0])['label']
    weight_list.append(weight)
    return weight_list

for g in range(smi_length):

    smi = dataset[g]
    clique_list = []
    atom_list = []
    BOND_TYPES = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                  Chem.rdchem.BondType.AROMATIC]
    mol = Chem.MolFromSmiles(smi)
    #mol = Chem.AddHs(mol)
    atoms = mol.GetAtoms()
    for atom in atoms:
        atom_list.append(atom.GetSymbol())
    nx_graph = nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol))
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        btype = BOND_TYPES.index(bond.GetBondType())
        nx_graph[a1][a2]['label'] = btype
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
        weight = nx_graph.get_edge_data(e[0],e[1])['label']
        edge = ((atom_list[e[0]],atom_list[e[1]]),weight)
        clique_id = add_to_vocab(edge)
        clique_list.append(clique_id)

    for r in mcb:
        weights = tuple(find_ring_weight(r,nx_graph))
        ring = []
        for i in range(len(r)):
            ring.append(atom_list[r[i]])
        cycle = (tuple(ring),weights)
        cycle_id = add_to_vocab(cycle)
        clique_list.append(cycle_id)


    all_nodes.append(clique_list)

    motifs = mcb + edges
    adj = np.eye(len(clique_list))
    for m in range(len(motifs)):
        for i in range(m + 1, len(motifs)):
            for j in motifs[m]:
                if j in motifs[i]:
                    adj[m, i] = 1.0
                    adj[i, m] = 1.0
    all_adj.append(adj)

print(len(vocab))

len_list = []
for nodes in all_nodes:
    length = len(nodes)
    len_list.append(length)
max_len = max(len_list)
print(max_len)

lists = [all_nodes,all_adj]
with open('data/processed/pretrain.pkl', 'wb') as save_file:
    pickle.dump(lists, save_file)










