# Probabilistic Conceptual Explainers <br> for Foundation Models
This repo contains the code and data for our PACE (ICML 2024 paper):

**Probabilistic Conceptual Explainers: Trustworthy Conceptual Explanations for Vision Foundation Models**<br>
Hengyi Wang, Shiwei Tan, Hao Wang<br>
[[Paper](http://www.wanghao.in/paper/ICML24_PACE.pdf)] [[ICML Website](https://icml.cc/virtual/2024/poster/34650)]

and our VALC (EMNLP 2024 Findings paper):

**Variational Language Concepts for Interpreting Foundation Language Models**<br>
[[Paper](http://www.wanghao.in/paper/EMNLP24_VALC.pdf)] [[ACL Website](https://aclanthology.org/2024.findings-emnlp.505/)]

## Probabilistic Conceptual Explainers (PACE) for Vision Transformers

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
### Train PACE for the *Color* Dataset
```bash
python main.py --train  --task Color --name ViT-PACE --num_epochs 1
```
### Test PACE for the *Color* Dataset
```bash
python main.py  --task Color --name ViT-PACE --num_epochs 1
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
