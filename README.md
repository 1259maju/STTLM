## STTLM: Spatial-temporal Transformer Networks for Traffic Flow Forecasting Using a Pre-Trained Language Model [https://doi.org/10.3390/s24175502]
The code for "Spatial-temporal Transformer Networks for Traffic Flow Forecasting Using a Pre-Trained Language Model"

## Introduction
Traffic flow forecasting is a significant branch of spatial–temporal prediction. It involves analyzing historical traffic conditions, modeling the spatial–temporal dependencies in traffic flows, and using these data to estimate future traffic conditions at a specific location. The complexity of modeling spatial–temporal dependencies challenges traffic forecasting.
This study proposes a spatial–temporal transformer network (STTLM) incorporating a pre-trained language model (LM) to forecast traffic flow. The key contributions of this study are summarized as follows:
(1)	We have developed a framework to extract spatial–temporal features from traffic data using the Transformer’s self-attention mechanism and the design of embeddings to extract spatial–temporal dependencies.
(2)	Our approach involves using the temporal Transformer (TT) first to extract the features related to temporal information separately. Then, these features are input into the spatial Transformer (ST) together with the unique embedding associated with spatial data. This method realizes the fusion of spatial–temporal information, avoids the confusion of spatial-temporal details during the initial self-attention process, and maximizes the role of embedding in the model.
(3)	Additionally, we utilized pre-trained language models to improve sequence prediction performance without the need for complex temporal and linguistic data alignment.

## Prerequisites
Our code is based on Python3.8, a few depended libraries as as follows:
1. Torch ==2.0.0
2. Transformers ==4.39.0
3. NumPy ==1.22.3
4. SciPy ==1.8.0
5. Pandas ==1.4.2
6. Peft == 0.9.0
7. PyWavelets ==1.4.1
8. Sentencepiece ==0.2.0
9. Torchinfo ==1.8.0
10. PyYAML ==6.0.1

## Dataset
We chose two datasets from "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting"[DCRNN] (https://github.com/liyaguang/DCRNN), which are METRLA and PEMSBAY. Two datasets from "Spatial-Temporal Synchronous Graph Convolutional Networks"[STSGCN] (https://github.com/Davidham3/STSGCN), they are PEMS04 and PEMS08. The preprocessing operations on the datasets mainly refer to "STAEformer: Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting"[STAEformer](https://github.com/XDZhelheim/STAEformer).


## Training Commands
```bash
cd STTLM/
#Run our proposed STTLM network 
python3 train1.py -d <dataset> -g <gpu_id>    
```
`<dataset>`:
- METRLA
- PEMSBAY
- PEMS04
- PEMS08

## Citation
If you find this repository, e.g., the code and the datasets, useful in your research, please cite the following paper:

    @article{Ju2024spatial,title={Spatial-temporal Transformer Networks for Traffic Flow Forecasting Using a Pre-Trained Language Model},
    author={Ma, Ju; Zhao,Juan and Hou,Yao},
    journal={Sensors 2024, 24, 5502},
    year={2024}
    }   

## Acknowledgments

We would like to extend our sincere gratitude to the authors of the paper titled STAEformer for their invaluable contribution. This codebase is heavily inspired by and adapted from their original implementation. Without their work, this project would not have been possible.
