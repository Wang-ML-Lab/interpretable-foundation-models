# Probabilistic Conceptual Explainers (PACE): <br> Trustworthy Conceptual Explanations for Vision Foundation Models
This repo contains the code and data for our ICML 2024 paper:

**Probabilistic Conceptual Explainers: Trustworthy Conceptual Explanations for Vision Foundation Models**<br>
Hengyi Wang, Shiwei Tan, Hao Wang<br>
[[Paper](wanghao.in/paper/ICML24_PACE.pdf)] [[Poster](https://icml.cc/virtual/2024/poster/34650)]

## Installation
conda env create -f environment_PACE.yml

conda activate PACE

cd src

## Generate the *Color* Dataset 

python generate_data.py

## Finetune ViT for the *Color* Dataset

python main.py --train  --task Color --name ViT-base --num_epochs 5 --lr 1e-3 --require_grad

## Train PACE for the *Color* Dataset

python main.py --train  --task Color --name ViT-PACE --num_epochs 1

## Test PACE for the *Color* Dataset

python main.py  --task Color --name ViT-PACE --num_epochs 1



## Reference

```bib
@inproceedings{PACE,
  title={Probabilistic Conceptual Explainers: Trustworthy Conceptual Explanations for Vision Foundation Models},
  author={Hengyi Wang and
          Shiwei Tan and
          Hao Wang},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```
