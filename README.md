## 3MTox
A motif-level graph-based multi-view chemical language model for toxicity identification with deep interpretation

## Introduction
In this study, we propose a motifs-level graph-based multi-view pretraining language model, called 3MTox, for toxicity identification. The 3MTox model uses Bidirectional Encoder Representations from Transformers (BERT) as the backbone framework, and a motif graph as input. The results of extensive experiments showed that our 3MTox model achieved state-of-the-art performance on toxicity benchmark datasets and outperformed the baseline models considered. In addition, the interpretability of the model ensures that the it can quickly and accurately identify toxicity sites in a given molecule, thereby contributing to the determination of the status of toxicity and associated analyses. We think that the 3MTox model is among the most promising tools that are currently available for toxicity identification.
![image](https://github.com/idrugLab/3MTox/blob/main/pngs/model.png)
Fig. 2. The framework of 3MTox

## Requirements
This project is developed using python 3.7.10, and mainly requires the following libraries.
```txt
rdkit==2023.03.2
scikit_learn==1.0.2
networkx==2.6.3
torch==1.13.0+cu116
```
To install [requirements](https://github.com/idrugLab/3MTox/blob/main/requirements.txt):
```txt
pip install -r requirements.txt
```

## Pre-training
### 1. preparing dataset for pre-training
We directly used a pretrained molecular corpus (~1.45 million) from our previous study-FG-BERT
```txt
path: https://github.com/idrugLab/3MTox/blob/main/data/raw/dataset_select_chembl.pkl
```
### 2. preprocessing pre-training dataset
We design a multi-view task in the framework as a pretraining strategy for the model, including a task of contrastive learning and one of masked motif prediction, where the goal of the prediction task is to predict the category of the motifs, so we have to manually assign a label to each motif in the motif vocabulary in advance.
This can be achieved with the following code:
```txt
path: https://github.com/idrugLab/3MTox/blob/main/dataset/data_process/Gen_motif_label.py
```
### 3. pre-training 
```txt
path: https://github.com/idrugLab/3MTox/blob/main/train_pre.py
```
## Fine-tuning
### 1. preprocessing fine-tuning dataset

### 2. fine-tuning model for one seed
```txt
path: 
```
### 3. fine-tuning model for ten seeds and optimization
```txt
path: 
```
## Results
As shown in Fig. 3a, it delivered the best performance on three of the four datasets.The overall predictive accuracy of the 3MTox model was higher by 8.5% to 13.6% compared with all baseline models when using the random data splitting method. The 3MTox model achieved the best performance on ToxCast (AUC = 0.675) and ClinTox (AUC = 0.900), while the pretrained FG-BERT model yielded the best results on Tox21 (AUC = 0.784) and SIDER (AUC = 0.640).In addition, it achieved the best overall performance, with the highest average AUC value of 0.748 (Fig. 3d).
![image](https://github.com/idrugLab/3MTox/blob/main/pngs/result.png)
Fig. 3. Performance of 3MTox compared with the baseline models on four benchmark toxicity datasets
