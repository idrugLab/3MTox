import pandas as pd
import networkx as nx
from rdkit import Chem
import numpy as np
import pickle

from utils.graph import atom_attr,bond_attr

path_raw = 'data/raw/clintox.csv'
df = pd.read_csv(path_raw)
smiles = df['smiles'].replace({np.nan:None})
smi_length = len(smiles)
all_nodes = []
all_adj = []
labels = []

reader = list(df.columns.values)
task_names = reader[1:]
num_tasks = len(task_names)
print(num_tasks)
smi_list = []

for g in range(smi_length):

    smi = smiles[g]

    clique_list = []
    atom_list = []
    BOND_TYPES = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                  Chem.rdchem.BondType.AROMATIC, Chem.rdchem.BondType.UNSPECIFIED]
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:

        atoms_attr = atom_attr(mol)
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
            motif_attr = 0
            for i in e:
                motif_attr += atoms_attr[i, :]
            edge_attr = bond_attr(mol,e).flatten()
            motif_attr = np.concatenate((motif_attr,edge_attr))
            clique_list.append(motif_attr)

        for r in mcb:
            motif_attr = 0
            for i in r:
                motif_attr += atoms_attr[i, :]
            edge_attr = [0,0,0,0,0,1,0,0,0,0]
            edge_attr = np.array(edge_attr)
            motif_attr = np.concatenate((motif_attr,edge_attr))
            clique_list.append(motif_attr)


        if len(clique_list) > 0:
            smi_list.append(smi)
            all_nodes.append(clique_list)

            motifs = mcb + edges
            adj = np.eye(len(clique_list))
            for m in range(len(motifs)):
                for i in range(m+1,len(motifs)):
                    for j in motifs[m]:
                        if j in motifs[i]:
                            adj[m,i] = 1.0
                            adj[i,m] = 1.0
            all_adj.append(adj)
            if num_tasks == 1:
                label = df[task_names[-1]][g]
                labels.append(label)
            else:
                label_list = []
                for task in task_names:
                    label_list.append(df[task][g])
                labels.append(label_list)

all_nodes_new = []
all_adj_new = []
labels_new = []
smi_list_new = []

for id,data in enumerate(all_nodes):
    if len(data) < 100:
        all_nodes_new.append(data)
        all_adj_new.append(all_adj[id])
        labels_new.append(labels[id])
        smi_list_new.append(smi_list[id])

len_list = []
for nodes in all_nodes_new:
    length = len(nodes)
    len_list.append(length)
max_len = max(len_list)
print(max_len)
print(len(all_nodes_new))
print(len(all_adj_new))
print(len(labels_new))
print(len(smi_list_new))
print(len(all_nodes_new[0][0]))

lists = [all_nodes_new,all_adj_new,labels_new,smi_list_new]
path_pro = 'data/processed/clintox.pkl'
with open(path_pro, 'wb') as save_file:
    pickle.dump(lists, save_file)