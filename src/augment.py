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

import torch.nn.functional as F
import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
import pickle
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from utils import *




def cov_z(phi):
    ret = np.empty((phi.shape[0], phi.shape[0]))
    for i in range(phi.shape[0]):
        for j in range(phi.shape[0]):
            if i == j:
                ret[i,j] = phi[i]*(1-phi[i])
            else:
                ret[i,j] = - phi[i]*phi[j]
    return ret





def contrastive_learning(e, e_prime, temperature=0.1):
    """
    Args:
        e (torch.Tensor): A tensor of shape (B, d) representing the original embeddings.
        e_prime (torch.Tensor): A tensor of shape (B, d) representing the transformed embeddings.
        temperature (float): A temperature hyperparameter to scale the logits.
        
    Returns:
        loss (torch.Tensor): The contrastive loss value.
    """
    # Normalize the embeddings and transformations to have unit length
    e = e.float()
    e_prime = e_prime.float()
    e.requires_grad_(True)

    e = F.normalize(e, dim=-1)
    e_prime = F.normalize(e_prime, dim=-1)
    
    # Compute dot product similarity between e and e' (positive pairs)
    positive_similarity = torch.sum(e * e_prime, dim=-1)
    
    # Compute dot product similarity between e and all other e' (negative pairs)
    negative_similarity = torch.mm(e, e_prime.t())
    
    # Remove the similarity of positive pairs from negative_similarity
    diagonal_indices = torch.arange(e.shape[0])
    negative_similarity[diagonal_indices, diagonal_indices] = float('-inf')
    
    # Compute the logits for positive and negative pairs
    logits = torch.cat([positive_similarity.unsqueeze(1), negative_similarity], dim=1)
    
    # Apply temperature scaling to the logits
    logits /= temperature
    
    # Create labels for the positive pairs (the first column in logits)
    labels = torch.zeros(e.shape[0], dtype=torch.long, device=e.device)

    # gradient of e, not use autograd
    grad_e = torch.zeros_like(e)
    grad_e_prime = torch.zeros_like(e_prime)
    # Compute cross entropy loss for each sample
    loss = F.cross_entropy(logits, labels)
    # Compute gradients for e and e_prime
    grad_e = torch.autograd.grad(loss, e, retain_graph=True)[0]
    
    return grad_e, loss

def image_augment(image):
    b, c, h, w = image.shape
    contrast_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomResizedCrop(size=w),
                                          transforms.RandomApply([
                                              transforms.ColorJitter(brightness=0.5,
                                                                     contrast=0.5,
                                                                     saturation=0.5,
                                                                     hue=0.1)
                                          ], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          transforms.GaussianBlur(kernel_size=9),
                                          #transforms.ToTensor()
                                          transforms.Normalize((0.5,), (0.5,))
                                         ])
    image_trans = contrast_transforms(image).cuda()
    return image_trans

def contrative_transform(image, model, topic_model, temperature=0.1):
    # image: input image batch (B, C, H, W)
    # model: ViT model
    contrast_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomResizedCrop(size=96),
                                          transforms.RandomApply([
                                              transforms.ColorJitter(brightness=0.5,
                                                                     contrast=0.5,
                                                                     saturation=0.5,
                                                                     hue=0.1)
                                          ], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          transforms.GaussianBlur(kernel_size=9),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))
                                         ])
    image_trans = contrast_transforms(image)
    embed = model(image)
    embed_trans = model(image_trans)

    return embed, embed_trans
 