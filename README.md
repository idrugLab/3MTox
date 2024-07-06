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

## Usage
### Pre-training

### Fine-tuning

## Results
![image](https://github.com/idrugLab/3MTox/blob/main/pngs/result.png)
Fig. 3. Performance of 3MTox compared with the baseline models on four benchmark toxicity datasets
