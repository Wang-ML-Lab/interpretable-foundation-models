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
from utils import  run_kmeans, ImageNetDataset, Cub2011, plot_topics
from model import CLDA, ViTClassify
from torchviz import make_dot
from utils import load_train_data, load_val_data, softmax, dirichlet_expectation
from torchvision.transforms.functional import InterpolationMode
#from captum.attr import Lime, LimeBase
from augment import contrastive_learning, contrative_transform, image_augment
import wandb
from transformers import TrainingArguments, Trainer, AutoFeatureExtractor
from utils import MyImageDataset
from datasets import load_dataset
from evaluate import stability, faithfulness, get_topics, sparsity, parsimony, coherence, diversity
#import shap, lime
from utils import StanfordCars, MyImageDatasetFromStanfordCars, build_transform, attention_norm, topic_vis
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import random_split
import pdb
from evaluate import to_numpy

args = parser.parse_args() 
args.save_path = os.path.join(args.save_path, args.name)
sample_path = os.path.join('../sample', args.name)

np.random.seed(args.seed)   
torch.manual_seed(args.seed)
random.seed(args.seed)

if args.task == 'flower102':
    dataset_name = "nelorth/oxford-flowers"
    dataset = load_dataset(dataset_name)

    extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    train_inputs = extractor(dataset['train']['image'], return_tensors="pt")
    test_inputs = extractor(dataset['test']['image'], return_tensors="pt")
    train_dataset = MyImageDataset(train_inputs['pixel_values'], dataset['train']['label'])
    test_dataset = MyImageDataset(test_inputs['pixel_values'], dataset['test']['label'])
    val_dataset = test_dataset
    args.out_dim = 102
elif args.task == 'cub2011':
    transform = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    train_dataset = Cub2011(args.data_path, train=True, transform=transform, download=False)
    val_dataset = Cub2011(args.data_path, train=False, transform=transform, download=False)
    test_dataset = val_dataset
    args.out_dim = 200
elif args.task == 'cars':
    from utils import build_transform
    train_transform = build_transform(output_size=(224,224), is_train=True)
    test_transform = build_transform(output_size=(224,224), is_train=False)
    train_dataset = StanfordCars(root="../dataset/stanford_cars/", split='train', transform=train_transform)
    test_dataset = StanfordCars(root="../dataset/stanford_cars/", split='test', transform=test_transform)
    train_dataset = MyImageDatasetFromStanfordCars(train_dataset)
    test_dataset = MyImageDatasetFromStanfordCars(test_dataset)
    #dataset = StanfordCars(root="../dataset/", download=True)
    #print(train_dataset[0])
    #print(test_dataset[0])
    #train_dataset = dataset['train']
    #test_dataset = dataset['test']
    val_dataset = test_dataset
    args.out_dim = 196 

elif args.task == 'toy':
    # Define a transformation to convert the images to PyTorch tensors
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[:3, ...]),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to range [-1,1]
    ])

    # Load the images and labels
    dataset = []
    labels = []
    for class_dir in ['../dataset/toy/class0', '../dataset/toy/class1']:
        for image_name in os.listdir(class_dir):
            # Read image
            image = Image.open(os.path.join(class_dir, image_name))
        
            # Add to the lists
            dataset.append(image)
            labels.append(int(class_dir[-1]))  # class ID from the directory name

    # Convert lists to tensors
    labels = torch.tensor(labels)

    #Split into train and test sets
    # Pair up the data and labels
    paired_data = list(zip(dataset, labels))

    # Perform the split on the paired data
    train_size = int(0.8 * len(paired_data))  # 80% for training
    test_size = len(paired_data) - train_size
    train_data, test_data = random_split(paired_data, [train_size, test_size])

    # Now, you can access the images and labels in each set like this:
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

    #print('dataset', train_dataset[0])

    # Create data loaders for easier batch processing
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    args.out_dim = 2
    # out dim is number of unique labels in train set

model = ViTClassify(in_dim = args.b_dim, out_dim=args.out_dim,hid_dim=args.c_dim, layer=args.layer)
#print(model)
#model.linear.load_state_dict(torch.load(load_path+'linear.pt'))
#model.classify.load_state_dict(torch.load(load_path+'classify.pt'))
model = model.cuda()
#model = nn.DataParallel(model, device_ids=[0,1,2,3])
#x = train_dataset['encodings'][0]
#x = torch.zeros((args.train_batch_size, 3, 224,224)).cuda()
#y = model(x)
#make_dot(y, params=dict(list(model.named_parameters()))).render("vit_torchviz", format="png")


if 'clda' in args.name:
    clda = CLDA(d=args.c_dim,K=args.K,D=args.D,N=args.N,alpha=args.alpha,C = args.out_dim)
else:
    clda = None

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=args.num_epoches,      # total number of training epochs
    per_device_train_batch_size=args.train_batch_size,  # batch size per device during training
    per_device_eval_batch_size=args.eval_batch_size,   # batch size for evaluation
    warmup_steps=0,                # number of warmup steps for learning rate scheduler     change steps from 100 to 0
    weight_decay=args.weight_decay,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    seed = args.seed,
    load_best_model_at_end=True,
    metric_for_best_model=args.metric, # 'eval_matthews_correlation' for cola, etc.
    evaluation_strategy='epoch',
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
#clda._mus = clda._mu0 = run_kmeans(X0, args.K)

print('evaluating')
#model.load_state_dict(torch.load('../ckpt/bert-base' +'/' + args.task + '_' +'epoch10'+'_L-2-MLP.pt'))
model.load_state_dict(torch.load(args.save_path +'/' + args.task + '_' +'epoch'+str(args.num_epoches)+ '_L'+ str(args.layer)+'-MLP-'  + str(args.version)+'.pt'))
#model.load_state_dict(torch.load(args.save_path +'/' + args.task + '_' +'epoch'+str(args.num_epoches)+ '_L'+ str(args.layer)+'-MLP.pt'))
    #score = trainer.evaluate()
    #print('score', score)
#torch.save(model.linear.state_dict(), args.save_path +'/' + args.task + '_' +'linear-epoch'+str(args.num_epoches)+'.pt')
#torch.save(model.classify.state_dict(), args.save_path+'/'+ args.task + '_' +'classify-epoch'+str(args.num_epoches)+'.pt')

#args.version = 'kmeans'
if clda is not None:
    clda._mus = np.load(args.save_path+'/' + args.task + '_'+'mus-epoch'+str(args.num_epoches)+ '_L'+ str(args.layer)+'-MLP-'  + str(args.version)+'.npy')
    clda._sigmas = np.load(args.save_path+'/' + args.task + '_' +'sigmas-epoch'+str(args.num_epoches)+ '_L'+ str(args.layer)+'-MLP-'  + str(args.version)+'.npy')
    clda._eta = np.load(args.save_path+'/' + args.task + '_'+'eta-epoch'+str(args.num_epoches)+ '_L'+ str(args.layer)+'-MLP-'  + str(args.version)+'.npy')





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
#         if clda is None:
#             continue
#         name.append(0)
#         topic.append('T_'+str(idx))
#         #font.append(1) # np.exp(det(clda._sigmas[idx]))
#         if x is None:
#             x = clda._mus[idx].reshape(-1,args.c_dim)
#         else:
#             x = np.concatenate([x,clda._mus[idx].reshape(-1,args.c_dim)],axis=0)
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
make_map = {'Audi', 'BMW', 'Chevrolet', 'Dodge', 'Ford', 'Honda', 'Hyundai', 'Jeep', 'Lexus', 'Mercedes-Benz', 'Nissan', 'Porsche', 'Subaru', 'Toyota', 'Volkswagen'}

dataset_model_cnt = {}
dataset_make_cnt = {}

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
        if clda is None:
            continue

        A_o = att[args.layer]
        #A_e = model.effective_attention(att[args.layer+1])
        #A_e = attention_norm(A_e)
        A_o = attention_norm(A_o)
        gamma, phi = clda.do_e_step(states,  A_o) # inference w/o learning, so e step instead of em step.
        #gamma_trans, phi_trans = clda.do_e_step(states_trans, att_trans[args.layer + 1])
        E_log_theta = dirichlet_expectation(gamma)
        concept_train.append(np.exp(E_log_theta))
        #concept_train.append(phi.mean(1))
        #print(phi.mean(1)[0])
        # A_o_trans = att_trans[args.layer]
        # A_o_trans = attention_norm(A_o_trans)
        # gamma_trans, phi_trans = clda.do_e_step(states_trans,  A_o)
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
        if clda is None:
            continue

        A_o = att[args.layer]
        #A_e = model.effective_attention(att[args.layer+1])
        #A_e = attention_norm(A_e)
        A_o = attention_norm(A_o)
        gamma, phi = clda.do_e_step(states,  A_o) # inference w/o learning, so e step instead of em step.
        #gamma_trans, phi_trans = clda.do_e_step(states_trans, att_trans[args.layer + 1])
        E_log_theta = dirichlet_expectation(gamma)
        concept_test.append(np.exp(E_log_theta))
        #concept_test.append(phi.mean(1))
        #print(phi.mean(1)[0])
        A_o_trans = att_trans[args.layer]
        A_o_trans = attention_norm(A_o_trans)
        gamma_trans, phi_trans = clda.do_e_step(states_trans,  A_o)
        E_log_theta = dirichlet_expectation(gamma_trans)
        concept_aug_test.append(np.exp(E_log_theta))
        #concept_aug_test.append(phi_trans.mean(1))



concept_test = np.concatenate(concept_test, axis=0)
concept_aug_test = np.concatenate(concept_aug_test, axis=0)
pred_train = to_numpy(torch.stack(pred_train))
pred_test = to_numpy(torch.stack(pred_test))

concept_train = np.concatenate(concept_train, axis=0)
#concept_aug_train = np.concatenate(concept_aug_train, axis=0)
prob_train = np.array(prob_train)
prob_test = np.array(prob_test)
print('prob_test', prob_test.shape)

# concept_train = np.load(os.path.join(args.save_path, str(args.task)+str(args.num_epoches)+str(args.version) + '-concept_train.npy'),concept_train)
# concept_test = np.load(os.path.join(args.save_path, str(args.task)+str(args.num_epoches)+str(args.version) + '-concept_test.npy'),concept_test)
# concept_aug_train = np.load(os.path.join(args.save_path, str(args.task)+str(args.num_epoches)+str(args.version) + '-concept_aug_train.npy'),concept_aug_train)
# concept_aug_test = np.load(os.path.join(args.save_path, str(args.task)+str(args.num_epoches)+str(args.version) + '-concept_aug_test.npy'),concept_aug_test)
# pred_train = np.load(os.path.join(args.save_path, str(args.task)+str(args.num_epoches)+str(args.version) + '-pred_train.npy'),pred_train)
# pred_test = np.load(os.path.join(args.save_path, str(args.task)+str(args.num_epoches)+str(args.version) + '-pred_test.npy'),pred_test)
# prob_train = np.load(os.path.join(args.save_path, str(args.task)+str(args.num_epoches)+str(args.version) + '-prob_train.npy'),prob_train)
# prob_test = np.load(os.path.join(args.save_path, str(args.task)+str(args.num_epoches)+str(args.version) + '-prob_test.npy'),prob_test)


np.save(os.path.join(args.save_path, str(args.task)+str(args.num_epoches)+str(args.version) + '-concept_train.npy'),concept_train)
np.save(os.path.join(args.save_path, str(args.task)+str(args.num_epoches)+str(args.version) + '-concept_test.npy'),concept_test)
np.save(os.path.join(args.save_path, str(args.task)+str(args.num_epoches)+str(args.version) + '-concept_aug_train.npy'),concept_aug_train)
np.save(os.path.join(args.save_path, str(args.task)+str(args.num_epoches)+str(args.version) + '-concept_aug_test.npy'),concept_aug_test)
np.save(os.path.join(args.save_path, str(args.task)+str(args.num_epoches)+str(args.version) + '-pred_train.npy'),pred_train)
np.save(os.path.join(args.save_path, str(args.task)+str(args.num_epoches)+str(args.version) + '-pred_test.npy'),pred_test)
np.save(os.path.join(args.save_path, str(args.task)+str(args.num_epoches)+str(args.version) + '-prob_train.npy'),prob_train)
np.save(os.path.join(args.save_path, str(args.task)+str(args.num_epoches)+str(args.version) + '-prob_test.npy'),prob_test)

#pdb.set_trace()

stability_score = stability(concept_test, concept_aug_test)

print('stability', stability_score)

fhard, fsoft = faithfulness(concept_train, pred_train, concept_test, pred_test, prob_test=prob_test)
#faithfulness = faithfulness(concept_test, pred_test, concept_test, pred_test)

print('faithfulness', fhard, fsoft)          
          
sparsity = sparsity(concept_test)

print('sparsity', sparsity)

parsimony = parsimony(concept_test)

print('parsimony', parsimony)


