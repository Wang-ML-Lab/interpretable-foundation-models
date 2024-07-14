from cProfile import label
from transformers import Trainer, TrainingArguments
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
from config import parser
import torchvision.transforms as transforms
import torchvision
from utils import accuracy_score, dirichlet_expectation, read_tsv_file, compute_metrics, Adam, posterior_mu_sigma
from model import PACE, ViTClassify
from torchviz import make_dot
from utils import dirichlet_expectation
from torchvision.transforms.functional import InterpolationMode
from augment import image_augment
from transformers import TrainingArguments, Trainer, AutoFeatureExtractor
from utils import MyImageDataset
from datasets import load_dataset
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import random_split

args = parser.parse_args() 


args.save_path = os.path.join(args.save_path, args.name)

np.random.seed(args.seed)   
torch.manual_seed(args.seed)
random.seed(args.seed)

class MyEarlyStoppingCallback(EarlyStoppingCallback):

    def __init__(self, early_stopping_patience=1, early_stopping_threshold=0.0, args=None):
        super(MyEarlyStoppingCallback,self).__init__(early_stopping_patience, early_stopping_threshold)
        self.epochs = 0
        self.ub = 0
        self.flag = True
        self.args = args
    
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_to_check = args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)

        self.check_metric_value(args, state, control, metric_value)
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True
        self.epochs += 1
        self.flag = True

        if PACE is None:
            return

        # save model
        args = self.args
        torch.save(model.state_dict(), args.save_path +'/' + args.task + '_' +'epoch'+str(self.epochs)+'.pt')
        np.save(args.save_path+'/' + args.task + '_'+'mus-epoch'+str(self.epochs) +'.npy',PACE._mus)
        np.save(args.save_path+'/' + args.task + '_' +'sigmas-epoch'+str(self.epochs)+'.npy',PACE._sigmas)
        np.save(args.save_path+'/' + args.task + '_'+'eta-epoch'+str(self.epochs)+'.npy',PACE._eta)


class PACETrainer(Trainer):

    def compute_loss(self,model,inputs,return_outputs=False): # **args...
        #output = model(inputs['encodings'])  # get predict outputs and last word embeddings
        logits, states, att = model(inputs['encodings']) 
        image_trans = image_augment(inputs['encodings'])
        logits_trans, states_trans, att_trans = model(image_trans) 
        
        loss = torch.nn.CrossEntropyLoss()
        pred_prob = np.exp(logits.detach().cpu().numpy())/np.exp(logits.detach().cpu().numpy()).sum(axis=1)[:,None]
        pred_ids = torch.argmax(logits,-1)
        ViT_loss = loss(logits, inputs['labels']).sum() 
        if  mycallback.flag is True:
            mycallback.flag = False
        
        if PACE is not None and return_outputs is False: 
            
            gamma_trans, phi_trans = PACE.do_e_step(states_trans, att_trans[args.layer + 1])
            PACE._phi_trans = PACE._phi
            gamma, phi = PACE.do_em_step(states, att[args.layer + 1], cl=True,y=pred_prob)
            PACE.update_eta(logits)
            pred_prob = torch.argmax(logits, dim=-1)
            PACE_loss = PACE._delta  
            custum_loss = PACE_loss
        else:
            custum_loss = ViT_loss
           
        
        pred = {'label_ids':inputs['labels'], 'predictions':pred_ids}
        return (custum_loss, pred) if return_outputs else custum_loss









img_size = 224
randaug_magnitude = 0



img_size = (224,224)
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
        #transforms.RandAugment(num_ops=2,magnitude=randaug_magnitude),
        transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
    ])







# Define a transformation to convert the images to PyTorch tensors
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[:3, ...]),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to range [-1,1]
])

# Load the images and labels
dataset = []
labels = []
for class_dir in ['../dataset/Color/class0', '../dataset/Color/class1']:
    for image_name in os.listdir(class_dir):
        # Read image
        image = Image.open(os.path.join(class_dir, image_name))
        
        # Add to the lists
        dataset.append(image)
        labels.append(int(class_dir[-1]))  # class ID from the directory name

# Convert lists to tensors
labels = torch.tensor(labels)

# Split into train and test sets
# Pair up the data and labels
paired_data = list(zip(dataset, labels))

# Perform the split on the paired data
train_size = int(0.8 * len(paired_data))  # 80% for training
test_size = len(paired_data) - train_size
train_data, test_data = random_split(paired_data, [train_size, test_size])


train_images, train_labels = zip(*train_data)
test_images, test_labels = zip(*test_data)

# Convert the zipped data back to lists or tensors as needed
train_images = list(train_images)
train_labels = list(train_labels)
test_images = list(test_images)
test_labels = list(test_labels)

# Create MyImageDataset instances
train_dataset = MyImageDataset(train_images, train_labels, transform=transform)
test_dataset = MyImageDataset(test_images, test_labels, transform=transform)
val_dataset = test_dataset


# Create data loaders for easier batch processing
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
args.out_dim = 2

model = ViTClassify(in_dim = args.b_dim, out_dim=args.out_dim,hid_dim=args.c_dim, layer=args.layer)
model = model.cuda()

if 'PACE' in args.name:
    PACE = PACE(d=args.c_dim,K=args.K,D=args.D,N=args.N,alpha=args.alpha,C = args.out_dim)
else:
    PACE = None

training_args = TrainingArguments(
    output_dir='../results',          # output directory
    num_train_epochs=args.num_epochs,      # total number of training epochs
    per_device_train_batch_size=args.train_batch_size,  # batch size per device during training
    per_device_eval_batch_size=args.eval_batch_size,   # batch size for evaluation
    warmup_steps=0,                # number of warmup steps for learning rate scheduler    
    weight_decay=args.weight_decay,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    seed = args.seed,
    load_best_model_at_end=True,
    metric_for_best_model=args.metric, # 'eval_matthews_correlation' for cola, etc.
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate = args.lr,
)

mycallback = MyEarlyStoppingCallback(early_stopping_patience=10, args=args)

trainer = PACETrainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,            # evaluation dataset
    compute_metrics=compute_metrics,
    callbacks=[mycallback],
    
)

test_set = DataLoader(val_dataset,batch_size=args.eval_batch_size,shuffle=False)

print('train size', len(train_dataset))
print('eval size', len(val_dataset))

if args.train:
    print('training...')
    if not args.require_grad: # Train PACE, otherwise train ViT      
        model.load_state_dict(torch.load('../ckpt/ViT-base' +'/' + args.task + '_' +'epoch5'+'.pt'))
    trainer.train()
    torch.save(model.state_dict(), args.save_path +'/' + args.task + '_' +'epoch'+str(args.num_epochs)+'.pt')
else:
    print('evaluating...')
    model.load_state_dict(torch.load(args.save_path +'/' + args.task + '_' +'epoch'+str(args.num_epochs)+'.pt'))
 
if PACE is not None:
    if args.train:
        np.save(args.save_path+'/' + args.task + '_'+'mus-epoch'+str(args.num_epochs)+'.npy',PACE._mus)
        np.save(args.save_path+'/' + args.task + '_' +'sigmas-epoch'+str(args.num_epochs)+'.npy',PACE._sigmas)
        np.save(args.save_path+'/' + args.task + '_'+'eta-epoch'+str(args.num_epochs)+'.npy',PACE._eta)
    else:
        PACE._mus = np.load(args.save_path+'/' + args.task + '_'+'mus-epoch'+str(args.num_epochs)+'.npy')
        PACE._sigmas = np.load(args.save_path+'/' + args.task + '_' +'sigmas-epoch'+str(args.num_epochs)+'.npy')
        PACE._eta = np.load(args.save_path+'/' + args.task + '_'+'eta-epoch'+str(args.num_epochs)+'.npy')

        # explain ViT
        print('PACE is explaining ViT...')
        for i, inputs in enumerate(test_set):

            test_encodings = inputs['encodings'].cuda()
            test_labels = inputs['labels'].cuda()
            logits, states, att = model(test_encodings)

            # infer phi and gamma
            gamma, phi = PACE.do_e_step(states, att[args.layer + 1])

            # infer E[log(theta)], which is the expectation of log(theta)
            E_log_theta = dirichlet_expectation(gamma)

