from cProfile import label
from transformers import pipeline
from transformers import BertTokenizer, BertModel
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import EarlyStoppingCallback, TrainerCallback
from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datasets import load_metric,load_dataset
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
import numpy as np
from config import parser
import torchvision.transforms as transforms
import torchvision
from utils import accuracy_score, dirichlet_expectation, read_tsv_file, compute_metrics, Adam, posterior_mu, posterior_mu_sigma, vis, kmeans_init
from utils import  run_kmeans, plot_topics
from model import PACE, ViTClassify
from torchviz import make_dot
from utils import load_train_data, load_val_data, softmax, dirichlet_expectation
from torchvision.transforms.functional import InterpolationMode
#from captum.attr import Lime, LimeBase
from augment import contrastive_learning, contrative_transform, image_augment
import wandb
from transformers import TrainingArguments, Trainer, AutoFeatureExtractor
from utils import MyImageDataset
from datasets import load_dataset
from evaluate import stability, faithfulness, sparsity, parsimony, faithfulness_linear
#import shap, lime
from utils import attention_norm, topic_vis, load_dataset_by_task
from PIL import Image
import pdb
from evaluate import to_numpy

args = parser.parse_args() 
args.save_path = os.path.join(args.save_path, args.name)
sample_path = os.path.join('../sample', args.name)

np.random.seed(args.seed)   
torch.manual_seed(args.seed)
random.seed(args.seed)

# Dataset loading using load_dataset_by_task function
train_dataset, test_dataset, args.out_dim = load_dataset_by_task(args.task, args.data_path)
val_dataset = test_dataset

model = ViTClassify(in_dim = args.b_dim, out_dim=args.out_dim,hid_dim=args.c_dim, layer=args.layer)
#print(model)

model = model.cuda()

if 'PACE' in args.name:
    PACE = PACE(d=args.c_dim,K=args.K,D=args.D,N=args.N,alpha=args.alpha,C = args.out_dim)
else:
    PACE = None

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=args.num_epochs,      # total number of training epochs
    per_device_train_batch_size=args.train_batch_size,  # batch size per device during training
    per_device_eval_batch_size=args.eval_batch_size,   # batch size for evaluation
    warmup_steps=0,                # number of warmup steps for learning rate scheduler     change steps from 100 to 0
    weight_decay=args.weight_decay,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    seed = args.seed,
    load_best_model_at_end=True,
    metric_for_best_model=args.metric, # 'eval_matthews_correlation' for cola, etc.
    eval_strategy='epoch',
    save_strategy='epoch',
    learning_rate = args.lr,
    report_to="wandb",
    #resume_from_checkpoint=True,
   # eval_steps=100,
)



test_set = DataLoader(val_dataset,batch_size=args.eval_batch_size,shuffle=True)
train_set = DataLoader(train_dataset,batch_size=args.train_batch_size,shuffle=True)

print('train size', len(train_dataset))
print('eval size', len(val_dataset))

#X0 = np.load(os.path.join(args.save_path, 'X-L-2.npy'))
#PACE._mus = PACE._mu0 = run_kmeans(X0, args.K)

print('evaluating')

model.load_state_dict(torch.load(args.save_path +'/' + args.task + '_epoch'+str(args.num_epochs)+'.pt'))

#args.version = 'kmeans'
if PACE is not None:
    PACE._mus = np.load(args.save_path+'/' + args.task + '_mus-epoch'+str(args.num_epochs)+'.npy')
    PACE._sigmas = np.load(args.save_path+'/' + args.task + '_sigmas-epoch'+str(args.num_epochs)+'.npy')
    PACE._eta = np.load(args.save_path+'/' + args.task + '_eta-epoch'+str(args.num_epochs)+'.npy')




# temperarilly CPU bounded, instead of GPU-bounded, needs multi-thread if multiple run at the same time
# numpy matrix manipulation test

x = None
pos = []
topic = []
name = []
patch_img = []
full_img = []
word_embed = {}
word_cnt = {}
top_words = [{} for _ in range(args.K)] # maintain a priority queue of prob for tokens in each topic
pred_label = []
tok = []
font = []
topic_cnt = {}


# for idx in range(args.K):  # args.K
#         if PACE is None:
#             continue
#         name.append(0)
#         topic.append('T_'+str(idx))
#         #font.append(1) # np.exp(det(PACE._sigmas[idx]))
#         if x is None:
#             x = PACE._mus[idx].reshape(-1,args.c_dim)
#         else:
#             x = np.concatenate([x,PACE._mus[idx].reshape(-1,args.c_dim)],axis=0)
#         patch_img.append(np.ones((224//16,224//16,3))) # patch_img[-1].shape
#         full_img.append(np.ones((224,224,3)))  # full_img[-1].shape
#         pos.append((-1,-1))

# x = torch.Tensor(x).cuda()

topic_cnt = dict(sorted(topic_cnt.items(), key=lambda item: item[1],reverse=True))
print(topic_cnt)
concepts = [[] for _ in range(args.K)]
#top_topics = list(topic_cnt)[1:6]
top_topics = {}
tt_cp = [x for x in top_topics]
tw = {}


#top_topics = list(topic_cnt)[5:10]


topic_se = 24
class_1 = 10
class_2 = 20

sample_num = 50000#5000
avg_corr = 0
batch_cnt = 0

# test metrics for LIME model
# ref https://captum.ai/api/lime.html


# interprete classifier from embedding inputs

concept_all = []
label_all = []

model.eval()
concept_test = []
concept_aug_test = []
concept_train = []
concept_aug_train = []
prob_train = []
prob_test = []

pred_train = []
pred_test = []
embeds = []
corpus = []
patches = []
attentions = []
concept_images = []
masked_concept_images = []
concept_labels = []
left_up_att = []
right_up_att = []
left_down_att = []
right_down_att = []

model_map = {'Sedan', 'SUV', 'Convertible', 'Minivan', 'Coupe', 'Wagon', 'Hatchback', 'Van', 'Truck', 'Pickup'}
make_map = {'Audi', 'BMW', 'Chevrolet', 'Dodge', 'Ford', 'Honda', 'Hyundai', 'Jeep', 'Lexus', 'Mercedes-Benz', 'Nissan', 'Porsche', 'Subaru', 'Colorota', 'Volkswagen'}

dataset_model_cnt = {}
dataset_make_cnt = {}

# only perform inference if not loading concepts from saved path
if not args.load_concepts:
    print("Performing inference on train and test datasets...")
    
    #train datset
    with torch.no_grad():
        cnt = 0
        for id, inputs in enumerate(train_set):
            print('train batch', id)
            train_encodings = inputs['encodings'].cuda()
            train_labels = inputs['labels'].cuda()
            #test_path = inputs['path']
            #test_mask = inputs['attention_mask'].cuda()
            #print(test_encodings.size())
            logits, states, att = model(train_encodings)

            # get augmented outputs
            image_trans = image_augment(inputs['encodings'])
            logits_trans, states_trans, att_trans = model(image_trans) 

            preds = logits.argmax(-1)
            #print('preds', preds)
            #logits = logits.detach().cpu().numpy()
            
            for pp in preds:
                pred_train.append(pp)
            for i in range(len(logits)):
                prob_train.append((torch.softmax(logits[i], dim=0)).detach().cpu().numpy())
                #print('prob', prob_train[-1])
            if PACE is None:
                continue

            A_o = att[args.layer]
            #A_e = model.effective_attention(att[args.layer+1])
            #A_e = attention_norm(A_e)
            A_o = attention_norm(A_o)
            gamma, phi = PACE.do_e_step(states,  A_o) # inference w/o learning, so e step instead of em step.
            #gamma_trans, phi_trans = PACE.do_e_step(states_trans, att_trans[args.layer + 1])
            E_log_theta = dirichlet_expectation(gamma)
            concept_train.append(np.exp(E_log_theta))
            #concept_train.append(phi.mean(1))
            #print(phi.mean(1)[0])
            # A_o_trans = att_trans[args.layer]
            # A_o_trans = attention_norm(A_o_trans)
            # gamma_trans, phi_trans = PACE.do_e_step(states_trans,  A_o)
            # concept_aug_train.append(phi_trans.mean(1))

    # test dataset
    with torch.no_grad():
        cnt = 0
        for id, inputs in enumerate(test_set):
            print('test batch', id)
            test_encodings = inputs['encodings'].cuda()
            test_labels = inputs['labels'].cuda()
            #test_path = inputs['path']
            #test_mask = inputs['attention_mask'].cuda()
            #print(test_encodings.size())
            logits, states, att = model(test_encodings)

            # get augmented outputs
            image_trans = image_augment(inputs['encodings'])
            logits_trans, states_trans, att_trans = model(image_trans) 

            preds = logits.argmax(-1)
            #logits = logits.detach().cpu().numpy()
            
            for pp in preds:
                pred_test.append(pp)
            for i in range(len(logits)):
                prob_test.append((torch.softmax(logits[i], dim=0)).detach().cpu().numpy())
                #print('prob', prob_test[-1])
            if PACE is None:
                continue

            A_o = att[args.layer]
            #A_e = model.effective_attention(att[args.layer+1])
            #A_e = attention_norm(A_e)
            A_o = attention_norm(A_o)
            gamma, phi = PACE.do_e_step(states,  A_o) # inference w/o learning, so e step instead of em step.
            #gamma_trans, phi_trans = PACE.do_e_step(states_trans, att_trans[args.layer + 1])
            E_log_theta = dirichlet_expectation(gamma)
            concept_test.append(np.exp(E_log_theta))
            #concept_test.append(phi.mean(1))
            #print(phi.mean(1)[0])
            A_o_trans = att_trans[args.layer]
            A_o_trans = attention_norm(A_o_trans)
            gamma_trans, phi_trans = PACE.do_e_step(states_trans,  A_o)
            E_log_theta = dirichlet_expectation(gamma_trans)
            concept_aug_test.append(np.exp(E_log_theta))
            #concept_aug_test.append(phi_trans.mean(1))

    # process inferred data
    concept_train = np.concatenate(concept_train, axis=0)
    concept_test = np.concatenate(concept_test, axis=0)
    concept_aug_test = np.concatenate(concept_aug_test, axis=0)
    # concept_aug_train = np.concatenate(concept_aug_train, axis=0)
    pred_train = to_numpy(torch.stack(pred_train))
    pred_test = to_numpy(torch.stack(pred_test))
    prob_train = np.array(prob_train)
    prob_test = np.array(prob_test)
    print('prob_test', prob_test.shape)
else:
    print("Skipping inference - will load concepts and probabilities from saved path...")

# conditionally load concepts and probabilities from saved path or use inferred ones
if args.load_concepts:
    print("Loading concepts and probabilities from saved path...")
    concept_train = np.load(os.path.join(args.save_path, str(args.task)  +'_epoch'+str(args.num_epochs) + '-concept_train.npy'))
    concept_test = np.load(os.path.join(args.save_path, str(args.task)  +'_epoch'+str(args.num_epochs) + '-concept_test.npy'))
    # concept_aug_train = np.load(os.path.join(args.save_path, str(args.task)  +'_epoch'+str(args.num_epochs) + '-concept_aug_train.npy'))
    concept_aug_test = np.load(os.path.join(args.save_path, str(args.task)  +'_epoch'+str(args.num_epochs) + '-concept_aug_test.npy'))
    pred_train = np.load(os.path.join(args.save_path, str(args.task)  +'_epoch'+str(args.num_epochs) + '-pred_train.npy'))
    pred_test = np.load(os.path.join(args.save_path, str(args.task)  +'_epoch'+str(args.num_epochs) + '-pred_test.npy'))
    prob_train = np.load(os.path.join(args.save_path, str(args.task)  +'_epoch'+str(args.num_epochs) + '-prob_train.npy'))
    prob_test = np.load(os.path.join(args.save_path, str(args.task)  +'_epoch'+str(args.num_epochs) + '-prob_test.npy'))
else:
    print("Using inferred concepts and probabilities from model...")


# normalize concept_train and concept_test
concept_train = concept_train / concept_train.sum(axis=1)[:,None]
concept_test = concept_test / concept_test.sum(axis=1)[:,None]
# concept_aug_train = concept_aug_train / concept_aug_train.sum(axis=1)[:,None]
concept_aug_test = concept_aug_test / concept_aug_test.sum(axis=1)[:,None]


# only save concepts and probabilities if we inferred them (not loaded from saved path)
if not args.load_concepts:
    print("Saving inferred concepts and probabilities...")
    np.save(os.path.join(args.save_path, str(args.task)  +'_epoch'+str(args.num_epochs) + '-concept_train.npy'),concept_train)
    np.save(os.path.join(args.save_path, str(args.task)  +'_epoch'+str(args.num_epochs) + '-concept_test.npy'),concept_test)
    np.save(os.path.join(args.save_path, str(args.task)  +'_epoch'+str(args.num_epochs) + '-concept_aug_train.npy'),concept_aug_train)
    np.save(os.path.join(args.save_path, str(args.task)  +'_epoch'+str(args.num_epochs) + '-concept_aug_test.npy'),concept_aug_test)
    np.save(os.path.join(args.save_path, str(args.task)  +'_epoch'+str(args.num_epochs) + '-pred_train.npy'),pred_train)
    np.save(os.path.join(args.save_path, str(args.task)  +'_epoch'+str(args.num_epochs) + '-pred_test.npy'),pred_test)
    np.save(os.path.join(args.save_path, str(args.task)  +'_epoch'+str(args.num_epochs) + '-prob_train.npy'),prob_train)
    np.save(os.path.join(args.save_path, str(args.task)  +'_epoch'+str(args.num_epochs) + '-prob_test.npy'),prob_test)
else:
    print("Skipping save since concepts and probabilities were loaded from saved path...")

#pdb.set_trace()

stability_score = stability(concept_test, concept_aug_test)

print('stability', stability_score)

fhard, fsoft = faithfulness_linear(concept_train, pred_train, concept_test, pred_test, prob_test=prob_test)
#faithfulness = faithfulness(concept_test, pred_test, concept_test, pred_test)

print('faithfulness', fhard, fsoft)          
          
sparsity = sparsity(concept_test)

print('sparsity', sparsity)

parsimony = parsimony(concept_test)

print('parsimony', parsimony)

# log txt file
with open(args.save_path + '/' + args.task  + '_epoch' + str(args.num_epochs) + '.txt', 'w') as f:
    f.write('stability: ' + str(stability_score) + '\n')
    f.write('faithfulness: ' + str(fhard) + ' ' + str(fsoft) + '\n')
    f.write('sparsity: ' + str(sparsity) + '\n')
    f.write('parsimony: ' + str(parsimony) + '\n')


