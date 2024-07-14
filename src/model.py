from transformers import pipeline
from transformers import ViTConfig, ViTForImageClassification
from transformers import ViTForImageClassification, ViTConfig
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pickle
import os
from math import pi
import numpy as np
from sklearn.mixture import GaussianMixture
import pickle
from numpy import random
import scipy.sparse as sp 
from scipy.special import gammaln
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn import manifold
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys, re, time, string
from scipy.special import gammaln, psi
from numpy.linalg import *
import math
import pandas as pd
from config import parser
#from config_parse_args import parser_args
from utils import accuracy_score, dirichlet_expectation, read_tsv_file, compute_metrics, Adam, posterior_mu_sigma
import time
from torch.nn import functional as F
from scipy.special import softmax
from augment import contrastive_learning, contrative_transform
from utils import Adam

args = parser.parse_args() 
args.save_path = os.path.join(args.save_path, args.name)



class ViTClassify(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim, layer):
        super(ViTClassify, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.layer = layer

        
        ViT_config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k", output_hidden_states=True, output_attentions=True, num_labels=out_dim)
        self.ViT = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', config=ViT_config)
           
        if not args.require_grad: # vit be none trainable, train PACE instead  
            for param in self.ViT.parameters():
                param.requires_grad = False        
        self.linear = nn.Linear(in_dim, hid_dim)
        self.embedding = None

    def forward(self, encodings, labels=None):
        ViT_output = self.ViT(encodings)
        logits = ViT_output['logits']
        all_states = ViT_output['hidden_states']
        attention = ViT_output['attentions']
        states = all_states[self.layer]  
        hidden = self.linear(states)
        self.embedding = all_states[args.layer]

        return logits, hidden, attention
class PACE:

    def __init__(self, d, K, D, N, alpha, C):
        '''
        Arguments:
        K: number of topics
        d: dimension of embedding space
        D: number of images
        alpha: prior on theta
        eta: prior on mu and Sigma
        '''
        self._d = d
        self._K = K
        self._D = D
        self._N = N
        self._C = C
        self._alpha = alpha
        self._updatect = 0
        self._gamma = None
        self._phi = None
        self._mu0 = np.random.randn(K,d) * 10  
        self._mus = np.random.randn(K,d) * 10
        self._sigmas = np.array([np.eye(d) for _ in range(K)])
        self._eta = np.random.randn(C,K) # (num_class, num_topic)
        self._updatect = 0
        self._eps = 1e-50
        self._converge = False
        self._m_mu = 0
        self._m_sigma = 0
        self._cnt = 0
        self._snum = 0
        self._sigma0 = np.array([np.eye(self._d) for _ in range(self._K)])
        self._lr = 1e-2
        self._embeds = None
        self._delta = None
        self.faitful_loss = 0
        self.stability_loss = 0
        self._adam_delta = Adam(1e-4,0.9,0.9)
        self._reg = 1e-3
        self._w = None
        self.linear = nn.Linear(args.b_dim, args.c_dim).cuda()
        self._lrm = torch.empty(args.b_dim, args.c_dim).cuda()
        torch.nn.init.normal_(self._lrm)
        self._phi_trans = None
        self.optimizer = None
        
    def log_p_y(self, logits):

        logits = logits.cpu().detach().numpy()
        log_p = 0
        B, _ = logits.shape
        for b in range(B):
            phi_mean = self._phi[b].mean(0)
            pred_y = softmax(logits[b])
            for i in range(self._C):
                log_p += pred_y[i] * np.log(np.exp(self._eta[i].dot(phi_mean))/np.sum(np.exp(self._eta.dot(phi_mean))))
        
        return log_p


    def update_eta(self, logits):
        logits = logits.cpu().detach().numpy()
        pred_y = softmax(logits, axis=1)
        phi_mean = self._phi.mean((0,1)) 
        for i in range(self._C): 
            self._eta[i] = self._eta[i] - self._lr * (pred_y[:,i].mean() - np.exp(self._eta[i].dot(phi_mean))/np.sum(np.exp(self._eta.dot(phi_mean)))) * phi_mean
    
    
    
    def do_e_step(self, embeds, frac=None):
        batchD = len(embeds) 
        
        
        if args.frac == 'equal': 
            self._w = torch.Tensor(1).cuda() 
        elif args.frac=='fix': # fixed length attention
            self._w = frac.mean(1)[:,0,:]

        phi = random.gamma(self._K, 1./self._K, (batchD, embeds.size()[1], self._K))
        gamma = phi.sum(1)

       
        it = 0
        meanchange = 0
        sigma_invs = []
        sigma_dets = []
        for i in range(self._K):
            sigma_inv = inv(self._sigmas[i] + self._eps * np.eye(self._d)) 
            sigma_invs.append(sigma_inv)
            sigma_det = det(self._sigmas[i])
            sigma_dets.append(sigma_det)

        sigma_invs = torch.Tensor(np.array(sigma_invs))
        sigma_invs = Variable(sigma_invs, requires_grad=True).cuda()
        sigma_dets = torch.Tensor(np.array(sigma_dets))
        sigma_dets = Variable(sigma_dets, requires_grad=True).cuda()
        self._delta = 0
        self._delta = Variable(torch.Tensor(self._delta), requires_grad=True)
        
        tensor_mus = Variable(torch.Tensor(np.array(self._mus)),requires_grad=False).cuda()
        tensor_sigmas = Variable(torch.Tensor(np.array(self._sigmas)),requires_grad=False).cuda()
        it = 0
        meanchange = 0
        sigma_invs = []
        sigma_dets = []
        for i in range(self._K):
            sigma_inv = inv(self._sigmas[i] + self._eps * np.eye(self._d)) 
            sigma_invs.append(sigma_inv)
            sigma_det = det(self._sigmas[i])
            sigma_dets.append(sigma_det)
        sigma_invs = torch.Tensor(np.array(sigma_invs))
        sigma_invs = Variable(sigma_invs, requires_grad=True).cuda()
        sigma_dets = torch.Tensor(np.array(sigma_dets))
        sigma_dets = Variable(sigma_dets, requires_grad=True).cuda()
        self._delta = 0
        self._delta = Variable(torch.Tensor(self._delta), requires_grad=True)
        
        # Iterate between gamma and phi until convergence
        tensor_mus = Variable(torch.Tensor(np.array(self._mus)),requires_grad=False).cuda()
        tensor_sigmas = Variable(torch.Tensor(np.array(self._sigmas)),requires_grad=False).cuda()
        for it in range(0,10): # train to converge. 
            
            mus = tensor_mus.unsqueeze(-1) 
            embeds = embeds.view(-1,self._d,1) 


            dir_exp = dirichlet_expectation(gamma) 
            dir_exp = Variable(torch.Tensor(dir_exp),requires_grad=False).view(-1,self._K).cuda()
            spd_tensor = None
            phi_tensor = Variable(torch.Tensor(phi),requires_grad=False).cuda()
            for i in range(self._K): 
                mul = torch.matmul((embeds-mus[i,:]).transpose(1,2), sigma_invs[i,:,:])
                spd = 0.5 * torch.matmul(mul,embeds-mus[i,:]).view(-1,self._N) 

                if spd_tensor is None:
                    spd_tensor = spd.unsqueeze(-1)
                else:
                    spd_tensor = torch.cat([spd_tensor,spd.unsqueeze(-1)],dim=2)

                phi_tmp = torch.exp(dir_exp[:,i].view(-1,1)-spd/100 )/torch.sqrt(abs(sigma_dets[i])+self._eps)   

                phi_tmp =phi_tmp * self._w       
                phi[:,:,i] = phi_tmp.cpu().detach().numpy()

            phi = (phi+self._eps)/(phi.sum(-1, keepdims=True) + self._eps * self._K)  # along K axis

            self._delta = phi_tensor * spd_tensor 
            
            self._delta = self._delta.mean() * batchD
            gamma = self._alpha + phi.sum(1)  # along n axis
            del dir_exp, phi_tensor, spd_tensor
        del sigma_invs, sigma_dets, tensor_mus, tensor_sigmas 
        self._phi = phi
        self._gamma = gamma

        return (gamma, phi)
    
    def do_cl_e_step(self, embeds, frac=None, y = None):
        '''
        frac: attention of last hidden layer, size [B,num_heads,N,N]
        '''
        batchD = len(embeds) 
        f_optimizer = Adam(1e-4,0.9,0.9)
        s_optimizer = Adam(1e-4,0.9,0.9)
        if args.frac == 'equal': 
            self._w = torch.Tensor(1).cuda() 
        elif args.frac=='fix': 
            self._w = frac.mean(1)[:,0,:]


        phi = random.gamma(self._K, 1./self._K, (batchD, embeds.size()[1], self._K))
        gamma = phi.sum(1)


        it = 0
        meanchange = 0
        sigma_invs = []
        sigma_dets = []

        for i in range(self._K):
            sigma_inv = inv(self._sigmas[i] + self._eps * np.eye(self._d)) 
            sigma_invs.append(sigma_inv)
            sigma_det = det(self._sigmas[i])
            sigma_dets.append(sigma_det)
        sigma_invs = torch.Tensor(np.array(sigma_invs))
        sigma_invs = Variable(sigma_invs, requires_grad=True).cuda()
        sigma_dets = torch.Tensor(np.array(sigma_dets))
        sigma_dets = Variable(sigma_dets, requires_grad=True).cuda()
        self._delta = 0
        self._delta = Variable(torch.Tensor(self._delta), requires_grad=True)
        
        tensor_mus = Variable(torch.Tensor(np.array(self._mus)),requires_grad=False).cuda()
        tensor_sigmas = Variable(torch.Tensor(np.array(self._sigmas)),requires_grad=False).cuda()
        it = 0
        meanchange = 0
        sigma_invs = []
        sigma_dets = []
        for i in range(self._K):
            sigma_inv = inv(self._sigmas[i] + self._eps * np.eye(self._d)) 
            sigma_invs.append(sigma_inv)
            sigma_det = det(self._sigmas[i])
            sigma_dets.append(sigma_det)

        sigma_invs = torch.Tensor(np.array(sigma_invs))
        sigma_invs = Variable(sigma_invs, requires_grad=True).cuda()
        sigma_dets = torch.Tensor(np.array(sigma_dets))
        sigma_dets = Variable(sigma_dets, requires_grad=True).cuda()
        self._delta = 0
        self._delta = Variable(torch.Tensor(self._delta), requires_grad=True)
        
        # Iterate between gamma and phi until convergence
        tensor_mus = Variable(torch.Tensor(np.array(self._mus)),requires_grad=False).cuda()
        tensor_sigmas = Variable(torch.Tensor(np.array(self._sigmas)),requires_grad=False).cuda()

        for it in range(0,10): # train to converge. 
            mus = tensor_mus.unsqueeze(-1) 
            embeds = embeds.view(-1,self._d,1) 
            dir_exp = dirichlet_expectation(gamma)  
            dir_exp = Variable(torch.Tensor(dir_exp),requires_grad=False).view(-1,self._K).cuda()
            spd_tensor = None
            phi_tensor = Variable(torch.Tensor(phi),requires_grad=False).cuda()
            phi_mean = torch.mean(phi_tensor, dim=1).detach().cpu().numpy() 
            phi_fair_delta = 1/self._N * (np.einsum('bi,ij->bj',y, self._eta)- np.einsum('bi,ij->bj',np.exp(np.einsum('ij,bj->bi',self._eta, phi_mean)), self._eta)/np.sum(np.exp(np.einsum('ij,bj->bi',self._eta, phi_mean)), axis=1, keepdims=True)  )
            phi_fair_delta = torch.from_numpy(phi_fair_delta).cuda()
            phi_trans_mean = self._phi_trans.mean(1)
            phi_stable_delta, _ =  contrastive_learning(torch.from_numpy(phi_mean).cuda(), torch.from_numpy(phi_trans_mean).cuda())
            phi_stable_delta *=  -1/self._N 
            eps = 1e-10
            for i in range(self._K):
                mul = torch.matmul((embeds-mus[i,:]).transpose(1,2), sigma_invs[i,:,:])
                spd = 0.5 * torch.matmul(mul,embeds-mus[i,:]).view(-1,self._N) 
                
                if spd_tensor is None:
                    spd_tensor = spd.unsqueeze(-1)
                else:
                    spd_tensor = torch.cat([spd_tensor,spd.unsqueeze(-1)],dim=2)
                phi_tmp = torch.exp(dir_exp[:,i].view(-1,1)-spd+self._eps)   
                phi_tmp =phi_tmp * self._w       
                phi[:,:,i] = phi_tmp.cpu().detach().numpy()
            phi = (phi+self._eps)/(phi.sum(-1, keepdims=True) + self._eps * self._K)  # along K axis
        
            self._delta = phi_tensor * spd_tensor 
            
            self._delta = self._delta.mean() * batchD
            relevance_scale = 1 
            stable_scale = 1
            phi = np.clip(phi, 0, 1e10)
            phi = (phi+self._eps)/(phi.sum(-1).reshape(batchD,self._N,1) + self._eps * self._K)
            # additional grad with second-order optimization
            f_grad = f_optimizer.update(phi_fair_delta.unsqueeze(1).detach().cpu().numpy())
            s_grad = s_optimizer.update(phi_stable_delta.unsqueeze(1).detach().cpu().numpy())
            phi += relevance_scale * f_grad + stable_scale * s_grad
            phi = (phi+self._eps)/(phi.sum(-1, keepdims=True) + self._eps * self._K)

            gamma = self._alpha + phi.sum(1)  # along n axis
            del dir_exp, phi_tensor, spd_tensor
        del sigma_invs, sigma_dets, tensor_mus, tensor_sigmas 
        self._phi = phi
        self._gamma = gamma

        return (gamma, phi)
    
    
    
    def update_lambda(self, embeds):
        '''
        update variational parameter lambda for beta
        '''
    def do_em_step(self, embeds,frac=None, cl=False, y=None):
        '''
        first E step,
        then M step
        
        gamma (B,K)
        phi (B,N,K)
        embeds (B,N,d)
        cl: using contrastive learning
        '''

        if cl and y is not None:
            gamma, phi = self.do_cl_e_step(embeds=embeds, frac=frac, y=y)
        else:
            gamma, phi = self.do_e_step(embeds,frac)
        embeds = embeds.cpu().detach().numpy()
        batchD = len(embeds)
        last_mus = self._mus.copy()
        last_sigmas = self._sigmas.copy()
        
        mu_init = 0
        sigma_init = 0
        norm = 0
        gamma = gamma.reshape(-1, self._K)
        phi = phi.reshape(-1,self._K)  
        embeds = embeds.reshape(-1,self._d)  
        mu_init = phi.reshape(-1,self._K,1) * embeds.reshape(-1,1,self._d) 
        mu_init = mu_init * self._w.cpu().detach().numpy().reshape(-1,1,1)
        mu_init = mu_init.sum(0) 
        delta = embeds.reshape(-1,1,self._d)-self._mus.reshape(1,-1,self._d) 
        square = delta.reshape(-1,self._K,1,self._d) * delta.reshape(-1,self._K,self._d, 1) # (B*N,K,d,d)
        sigma_init = phi.reshape(-1,self._K,1,1) * square 
        sigma_init = sigma_init * self._w.cpu().detach().numpy().reshape(-1,1,1,1)
        sigma_init = sigma_init.sum(0)
        norm = (phi*self._w.cpu().detach().numpy().reshape(-1,1)).sum(0) 
            
        rhot = 0.75 
        
        self._m_mu = self._mus * rhot * self._cnt + mu_init/(norm.reshape(-1,1)+self._eps) * (1-rhot) * batchD      
        self._m_sigma = self._sigmas * rhot * self._cnt + sigma_init/(norm.reshape(-1,1,1)+self._eps) * (1-rhot) * batchD
        self._cnt = self._cnt  + batchD * (1-rhot)
        self._mus = self._m_mu / self._cnt
        self._sigmas = self._m_sigma / self._cnt
       
        self._snum += 1
        for i in range(self._K):
           self._mus[i], self._sigmas[i] = posterior_mu_sigma(self._mus[i], self._sigmas[i], self._snum, self._mu0[i])           

        if abs(last_mus - self._mus).max() < 1e-5:
            self._converge = True

        phi = phi.reshape(-1,self._N,self._K)
        
        return gamma, phi

