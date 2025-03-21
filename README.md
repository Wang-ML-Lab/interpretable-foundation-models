# Probabilistic Conceptual Explainers <br> for Foundation Models
This repo contains the code and data for our PACE (ICML 2024 paper):

**Probabilistic Conceptual Explainers: Trustworthy Conceptual Explanations for Vision Foundation Models**<br>
Hengyi Wang*, Shiwei Tan*, Hao Wang<br>
[[Paper](http://www.wanghao.in/paper/ICML24_PACE.pdf)] [[ICML Website](https://icml.cc/virtual/2024/poster/34650)]

and our VALC (EMNLP 2024 Findings paper):

**Variational Language Concepts for Interpreting Foundation Language Models**<br>
Hengyi Wang, Shiwei Tan, Zhiqing Hong, Desheng Zhang, Hao Wang<br>
[[Paper](http://www.wanghao.in/paper/EMNLP24_VALC.pdf)] [[ACL Website](https://aclanthology.org/2024.findings-emnlp.505/)]

## Brief Introduction for PACE
We propose five desiderata for explaining vision foundation models like ViTs - faithfulness, stability, sparsity, multi-level structure, and parsimony - and demonstrate the inadequacy of current methods in meeting these criteria comprehensively. Rather than using sparse autoencoders (SAEs), we introduce a variational Bayesian explanation framework, dubbed ProbAbilistic Concept Explainers (PACE), which models the distributions of patch embeddings to provide trustworthy post-hoc conceptual explanations. Our PACE can provide dataset-, image-, and patch-level explanations for ViTs and achieves all five desiderata (faithfulness, stability, sparsity, multi-level structure, and parsimony) in a unified framework. 

## Probabilistic Conceptual Explainers (PACE) for Vision Transformers (ViTs)
PACE is compatible with *arbitrary* vision transformers.

Below are some sample concepts automatically discovered by our PACE, *without the need for concept annotation during training*. 

![More_Random_Samples_Color](https://github.com/user-attachments/assets/f39aa0c6-3427-428e-ada9-aa9880d0ca09)

**Figure 1.** Above are some sample concepts discovered by PACE in the *COLOR* dataset. See Figure 3 of [our paper](http://wanghao.in/paper/ICML24_PACE.pdf) for details on the *COLOR* dataset.

![More_Random_Samples_Flower](https://github.com/user-attachments/assets/80bd9dcf-2514-49ca-a659-6b101d423044)

**Figure 2.** Above are some sample concepts discovered by PACE in the *Oxford Flower* dataset. 



### Installation
```bash
conda env create -f environment_PACE.yml
conda activate PACE
cd src
```

### Generate the *Color* Dataset 
```bash
python generate_data.py
```
### Finetune ViT for the *Color* Dataset
```bash
python main.py --train  --task Color --name ViT-base --num_epochs 5 --lr 1e-3 --require_grad
```

### Finetune ViT for the Real-World Datasets
```bash
python main.py --train --task flower102 --name bert-base --lr 3e-5 --train_batch_size 16 --require_grad
```

```bash
python main.py --train --task cub2011 --name bert-base --lr 1e-4 --require_grad --train_batch_size 16
```

```bash
python main.py --train --task cars --name bert-base --lr 1e-4 --require_grad
```


### Train PACE for Each Dataset
```bash
python main.py --train  --task $Dataset --name ViT-PACE --num_epochs 1
```
### Test PACE for Each Dataset
```bash
python main.py  --task $Dataset --name ViT-PACE --num_epochs 1
```


## Probabilistic Conceptual Explainers (VALC) for Pretrained Language Models

Coming Soon!

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

@inproceedings{VALC,
  title={Variational Language Concepts for Interpreting Foundation Language Models},
  author={Hengyi Wang and
          Shiwei Tan and
          Zhiqing Hong and 
          Desheng Zhang and
          Hao Wang},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2024},
  year={2024}
}
```
