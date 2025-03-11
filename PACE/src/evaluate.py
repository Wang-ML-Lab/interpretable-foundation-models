# import metric packages
import numpy as np
import pandas as pd
from numpy.linalg import *
from scipy.linalg import sqrtm
from scipy.special import gammaln, psi
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from datasets import load_metric
from sklearn.linear_model import LogisticRegression, LinearRegression
from utils import *
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from scipy.stats import entropy
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return x


def faithfulness(concept_train, pred_train, concept_test, pred_test, hard=False, prob_test=None):
    # faithfulness with MLP classifier
    concept_train = to_numpy(concept_train)
    pred_train = to_numpy(pred_train) 
    concept_test = to_numpy(concept_test)
    pred_test = to_numpy(pred_test)
    concept_train = concept_test
    pred_train = pred_test
    pipe = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    )

    clf = pipe.fit(concept_train, pred_train)
    score = clf.score(concept_test, pred_test)

    return score, -1


def faithfulness_linear(concept_train, pred_train, concept_test, pred_test, hard=False, prob_test=None):
    # faithfulness with linear classifier
    concept_train = to_numpy(concept_train)
    pred_train = to_numpy(pred_train)
    concept_test = to_numpy(concept_test)
    pred_test = to_numpy(pred_test)
    ct = concept_train
    total = np.prod(ct.shape)
    mx = np.max(ct)
    mn = np.min(ct)
    span = mx - mn
    pipe = make_pipeline(preprocessing.StandardScaler(), LogisticRegression(random_state=0, max_iter=2500))
    clf = pipe.fit(concept_train, pred_train)
    hard_score = clf.score(concept_test, pred_test)
    prob = clf.predict_proba(concept_test)
    soft_score = np.mean([entropy(prob[i], prob_test[i]) for i in range(len(prob))])
    
    return hard_score, soft_score 


def stability(concept_orig, concept_aug, compute=True):
    assert len(concept_orig.shape) == 2
    assert concept_orig.shape == concept_aug.shape
    delta = np.linalg.norm(concept_orig - concept_aug, axis=1) / np.linalg.norm(concept_orig, axis=1)

    return np.mean(delta)

def sparsity(concept):
    assert len(concept.shape) == 2
    eps = 0.1 / concept.shape[1] 
    return np.mean(concept < eps)

def parsimony(concept):
    assert len(concept.shape) == 2
    return concept.shape[1]
