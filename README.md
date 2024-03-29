# HTGExplainer

This repo is the pytorch implementation of the paper [Heterogeneous Temporal Graph Neural Network Explainer](https://dl.acm.org/doi/10.1145/3583780.3614909) accepted in CIKM '23.

## Model Architecture

![model_figure](./htmodel.png)

## Setup

```python
conda create -n HTGExplainer python=3.10
conda activate HTGExplainer
conda install -r requirements.txt
```

## Datasets
Please download and unzip all the files from [link](https://drive.google.com/drive/folders/12w7E1utk4buXjCXw59KK532Q8Mf4Ig7C?usp=sharing) and put them under `data/` folder.

## Usage

todo

## Citation
Please consider citing our paper if you found it helpful! :)

```bibtex
@inproceedings{HTGExplainer,
author = {Li, Jiazheng and Zhang, Chunhui and Zhang, Chuxu},
title = {Heterogeneous Temporal Graph Neural Network Explainer},
year = {2023},
booktitle = {Proceedings of the 32nd ACM International Conference on Information and Knowledge Management (CIKM)},
}
```