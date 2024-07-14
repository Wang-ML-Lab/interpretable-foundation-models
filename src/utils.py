from tkinter import image_types
import numpy as np
import pandas as pd
from numpy.linalg import *
from scipy.linalg import sqrtm
from scipy.special import gammaln, psi
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from datasets import load_metric
from config import parser
from transformers import ViTFeatureExtractor
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor)
from torch import FloatTensor, div
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

import logging
import os
import random
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
import pickle
from sklearn.cluster import KMeans
from torchvision.datasets import VisionDataset
import os
from PIL import Image


from PIL import Image, ImageOps


args = parser.parse_args() 

def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['img']]
    return examples

def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]
    return examples


def accuracy_score(labels, preds):
    acc = (preds==labels).astype(np.float).mean()
    return acc

def compute_metrics_acc(pred):
    #labels = pred.label_ids
    labels, preds = pred.predictions

    #print('labels',labels)
    #print('preds',preds)
    #precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        #'f1': f1,
        #'precision': precision,
        #'recall': recall
    }    

def compute_metrics(pred):
    #labels = pred.label_ids
    labels, preds = pred.predictions
    #metric = load_metric('glue', args.task)
    metric =load_metric('accuracy')
    return metric.compute(predictions=preds, references=labels)

def dirichlet_expectation(alpha):
    '''
    E[log(theta)|alpha], where theta ~ Dir(alpha).
    from blei/online LDA
    '''
    if len(alpha.shape) == 1: # 1D version
        return psi(alpha) - psi(np.sum(alpha))
    return psi(alpha) - psi(np.sum(alpha,1))[:, np.newaxis]


def read_tsv_file(file_path):
    df = pd.read_csv(file_path,sep='\t')
    seq = df['sentence']
    return seq



class DynamicCrop(object):
    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, img):
        w, h = img.size
        crop_size = min(w, h)
        
        left_margin = (w - crop_size) / 2
        top_margin = (h - crop_size) / 2

        # Random crop for training
        if self.is_train:
            left_margin = random.randint(0, w - crop_size)
            top_margin = random.randint(0, h - crop_size)
        
        img = img.crop((left_margin, top_margin, left_margin + crop_size, top_margin + crop_size))
        return img

def build_transform(output_size, is_train=True):
    """
    Get the appropriate image transformation based on the training/testing phase.
    
    Parameters:
    - output_size (int or tuple): Size for resizing the cropped image.
    - is_train (bool): If True, random crop and resize are performed. Otherwise, center crop and resize.
    
    Returns:
    - torchvision.transforms.Compose: A composition of transformations.
    """
    return transforms.Compose([
        DynamicCrop(is_train=is_train),
        transforms.Resize(output_size),
        transforms.ToTensor()
    ])




class MyImageDataset(Dataset):
    """Dataset class for Image"""
    def __init__(self, dataset, labels, transform=None, normalize=None):
        super(MyImageDataset, self).__init__()
        assert(len(dataset) == len(labels))
        self.dataset = dataset
        self.labels = labels
        self.transform = transform
        self.normalize = normalize

        #print(self.labels)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        
        if self.transform:
            data = self.transform(data)
            img_to_tensor = transforms.ToTensor()
            # if data is not tensor
            if not isinstance(data, torch.Tensor):
                data = img_to_tensor(data)
        if self.normalize:
            data = self.normalize(data)
        
        return {'encodings':data, 'labels':self.labels[idx]}



class Adam:
    def __init__(self, alpha=1e-4, beta1=0.9, beta2=0.9):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = 0
        self.v = 0
        self.t = 0
        self.eps = 1e-5
    def update(self, g):
        self.t += 1
        self.m = self.beta1 * self.m + (1-self.beta1) * g
        self.v = self.beta2 * self.v + (1-self.beta2) * g**2
        self.m = self.m / (1-self.beta1**self.t)
        self.v = self.v / (1-self.beta2**self.t)
        return self.alpha * self.m / (np.sqrt(self.v+self.eps)+self.eps)


def posterior_mu_sigma(mu, sigma, n, mu0, n0=500, lamda0=None):
    if lamda0 is None:
        lamda0 = np.eye(sigma.shape[-1])
        
    post_mu = (n0*mu0+n*mu)/(n+n0)
    lamda = (n0* lamda0  +  n* inv(sigma)) / (n+n0) #+ n*n0/(n0+n) * np.matmul((mu-mu0).T,mu-mu0)
    post_sigma = (n0*lamda0 + n*inv(lamda) )/ (n+n0)
    return post_mu, post_sigma



def softmax(x): # 2D 
    """Compute softmax values for each sets of scores in x."""
    # x (B,d)
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)
    
