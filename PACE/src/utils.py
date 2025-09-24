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


def attention_norm(attention, a=1):
    attention = attention.mean(1)
    # max(attention, 0)
    #attention = torch.max(attention, torch.zeros_like(attention))
    attention = attention - attention.min()
    
    #attention = attention ** a
    attention = attention / attention.sum(-1, keepdims=True)

    return attention


class StanfordCars_simplified(torch.utils.data.Dataset):
    def __init__(self, root, transform = None):
        self.images = [os.path.join(root, file) for file in os.listdir(root)]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_file = self.images[index]
        image = Image.open(image_file).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image[None]
  

class StanfordCars(VisionDataset):
    """`Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset

    The Cars dataset contains 16,185 images of 196 classes of cars. The data is
    split into 8,144 training images and 8,041 testing images, where each class
    has been split roughly in a 50-50 split

    .. note::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Args:
        root (string): Root directory of dataset
        split (string, optional): The dataset split, supports ``"train"`` (default) or ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again."""

    def __init__(
        self,
        root: str,
        split = "train",
        transform = None,
        target_transform = None,
        download: bool = False,
    ):

        try:
            import scipy.io as sio
        except ImportError:
            raise RuntimeError("Scipy is not found. This dataset needs to have scipy installed: pip install scipy")

        super().__init__(root, transform=transform, target_transform=target_transform)

        #self._split = verify_str_arg(split, "split", ("train", "test"))
        #self._base_folder = pathlib.Path(root) / "stanford_cars"
        self._split = split
        self._base_folder = root
        devkit = os.path.join(self._base_folder, "devkit")
        

        if self._split == "train":
            self._annotations_mat_path = os.path.join(devkit , "cars_train_annos.mat")
            self._images_base_path = os.path.join(self._base_folder , "cars_train")
        else:
            self._annotations_mat_path = os.path.join(self._base_folder , "cars_test_annos_withlabels.mat")
            self._images_base_path = os.path.join(self._base_folder , "cars_test")

        if download:
            self.download()

        self._samples = [
            (
                str(os.path.join(self._images_base_path , annotation["fname"])),
                annotation["class"] - 1,  # Original target mapping  starts from 1, hence -1
            )
            for annotation in sio.loadmat(self._annotations_mat_path, squeeze_me=True)["annotations"]
        ]

        self.classes = sio.loadmat(str(os.path.join(devkit , "cars_meta.mat")), squeeze_me=True)["class_names"].tolist()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx: int):
        """Returns pil_image and class_id for given index"""
        image_path, target = self._samples[idx]
        pil_image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            pil_image = self.transform(pil_image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return pil_image, target

    def download(self):
        if self._check_exists():
            return
    def _check_exists(self):
        if not (self._base_folder / "devkit").is_dir():
            return False

        return self._annotations_mat_path.exists() and self._images_base_path.is_dir()




def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['img']]
    return examples

def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]
    return examples



args = parser.parse_args() 

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


def build_transform_prev(crop_size, output_size, is_train=True):
    """
    Get the appropriate image transformation based on the training/testing phase.
    
    Parameters:
    - crop_size (int or tuple): Size for cropping. If int, a square crop is made.
    - output_size (int or tuple): Size for resizing the cropped image.
    - train (bool): If True, random crop and resize are performed. Otherwise, center crop and resize.
    
    Returns:
    - torchvision.transforms.Compose: A composition of transformations.
    """
    if is_train:
        return transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.Resize(output_size),
            transforms.ToTensor()
        ])
    else:
        return transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Resize(output_size),
            transforms.ToTensor()
        ])


class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        #print('sample', sample)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)
        #print(img)
        #print(target)
        if self.transform is not None:
            img = self.transform(img)['pixel_values'][0]
        return {'encodings':img, 'labels':target, 'path': sample.filepath}

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


class ImageNetDataset(Dataset):
    """Dataset class for ImageNet"""
    def __init__(self, dataset, labels, transform=None, normalize=None):
        super(ImageNetDataset, self).__init__()
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
        data = img_to_tensor(data)

        data = div(data, 255)
        # why would there by (1,224,224) samples?
        if data.size()[0] == 1:
            data = data.repeat(3,1,1)
        if self.normalize:
            data = self.normalize(data)
        
        return {'encodings':data, 'labels':self.labels[idx]}

class MyImageDatasetFromStanfordCars(Dataset):
    def __init__(self, stanford_cars_dataset, transform=None, normalize=None):
        super(MyImageDatasetFromStanfordCars, self).__init__()
        self.stanford_cars_dataset = stanford_cars_dataset
        self.transform = transform
        self.normalize = normalize

    def __len__(self):
        return len(self.stanford_cars_dataset)

    def __getitem__(self, idx):
        data, label = self.stanford_cars_dataset[idx]
        
        if self.transform:
            data = self.transform(data)
        img_to_tensor = transforms.ToTensor()
        # if data is not tensor
        if not isinstance(data, torch.Tensor):
            data = img_to_tensor(data)
        if self.normalize:
            data = self.normalize(data)
        
        return {'encodings': data, 'labels': label}


def load_train_data(data_dir, dataset,  img_size, magnitude, batch_size):
    with open(data_dir, 'rb') as f:
        ds = pickle.load(f)
        train_data = ds['image']
        train_labels = ds['label']
    transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
        transforms.RandAugment(num_ops=2,magnitude=magnitude),
    ])
    train_dataset = dataset(train_data, train_labels, transform,
        normalize=transforms.Compose([
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ]),
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    f.close()
    return train_dataset, train_loader

def load_val_data(data_dir, dataset, img_size, batch_size):
    with open(data_dir, 'rb') as f:
        ds = pickle.load(f)
        val_data = ds['image']
        val_labels = ds['label']
    transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
    ])
    val_dataset = dataset(val_data, val_labels, transform,
        normalize=transforms.Compose([
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ])
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    f.close()
    return val_dataset, val_loader

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

def posterior_mu(mu, mu0, n, n0=1):
    res = (n0*mu0+n*mu)/(n+n0)
    return res


# initializing should adapt to the data (mean, variance, norm...)
def posterior_mu_sigma(mu, sigma, n, mu0, n0=500, lamda0=None):
    #lamda = inv(np.matmul(sigma, sigma.T))
    #print('size', sigma.size)
    # fix mu, update sigma
    if lamda0 is None:
        lamda0 = np.eye(sigma.shape[-1])
        
    post_mu = (n0*mu0+n*mu)/(n+n0)
    #post_sigma = (n0* lamda0 + n* sigma) / (n0+n)
    #print('prod', np.matmul((mu-mu0).T,mu-mu0))
    lamda = (n0* lamda0  +  n* inv(sigma)) / (n+n0) #+ n*n0/(n0+n) * np.matmul((mu-mu0).T,mu-mu0)
    post_sigma = (n0*lamda0 + n*inv(lamda) )/ (n+n0)
    #print('post_sigma', det(post_sigma))
    return post_mu, post_sigma


def get_tf_idf_mask(corpus_words):
    #corpus_words = [[w for w in seq if w is not 0] for seq in corpus_words]
    corpus_bow = [list(Counter(seq).items()) for seq in corpus_words]

    tfidf = TfidfModel(corpus_bow, normalize=False)
    #filter low value words
    low_value = 4
    filtered_ids = []
    tfidf_mask = []
    for i in range(0, len(corpus_bow)):
        bow = corpus_bow[i]
        low_value_words = [] #reinitialize to be safe. You can skip this.
        low_value_words = [id for id, value in tfidf[bow] if value < low_value]
        new_mask = []
        for idx, w in enumerate(corpus_words[i]):
            if w in low_value_words or w==0 or idx==0: # consider padding, 'CLS' token
                new_mask.append(0)
            else:
                new_mask.append(1)
        #reassign 
        tfidf_mask.append(new_mask)
        #print(new_mask)
    return tfidf_mask


def topic_vis(mus, coherence, topic):
   pca = PCA(n_components=2)
   pc = pca.fit_transform(mus)
   ax = plt.subplot()
   for i in topic:
       coh = coherence[i]
       coh = round(coh, 1)
       ax.annotate( str(i), (pc[i,0], pc[i,1]), size=10) # + '-' + str(coh)
    
       ax.scatter(pc[i,0], pc[i, 1])
   plt.savefig('topic_scatter_vis.jpg')
   return ax

def vis(x,name,topic,patch=None, pos=None, full_img=None, sigmas = None, div= None, coh=None, top_topics = None, indiv_coh=None, attentions=None):
    '''
    x: features
    K: topic numbers
    topic: concept of patch/word
    patch: image of patch, numpy array (optional)
    '''
    #if pos is not None:
    #    print(pos)
    #print('topic', topic)
    # Get the Matplotlib logger
    mpl_logger = logging.getLogger('matplotlib')

    # Set the level to WARNING or higher
    mpl_logger.setLevel(logging.WARNING)

    mean = x.mean(axis=0)
    var = x.var(axis=0)
    #print(x.shape)
    #print(mean.shape)
    #print(var.shape)
    '''
    outer_indices = [i for i in range(x.shape[0]) if (abs(x[i]-mean) >  3*np.sqrt(var)).any()]
    print('outer', outer_indices)
    x = np.delete(x, outer_indices, axis=0)
    name = np.delete(name, outer_indices, axis=0)
    topic = np.delete(topic, outer_indices, axis=0)
    if patch is not None:
        patch = np.delete(patch, outer_indices, axis=0)
    if pos is not None:
        pos = np.delete(pos, outer_indices, axis=0)
    if full_img is not None:
        full_img = np.delete(full_img, outer_indices, axis=0)
    #if sigmas is not None:
    #    sigmas = np.delete(sigmas, outer_indices, axis=0)
    '''
    pca = PCA(n_components=2)
    pc = pca.fit_transform(x)
    colors = [ 'blue', 'red','purple', 'green','lime', 'cyan', 'orange', 'pink','black', 'grey'] * 20
#    for i, txt in enumerate(name):
#        ax.annotate(txt, (pc[i,0], pc[i,1]),
#                color=colors[topic[i]], size=10)
#    ax.scatter(pc[:,0], pc[:, 1]) 
    '''
    fig = plt.figure(figsize=(10,10))
    axs = fig.add_subplot(1,1,1)
    axs.set_xlabel('pc 1', fontsize=15)
    axs.set_ylabel('pc 2', fontsize=15)
    axs.set_title('2D PCA', fontsize=20)
    print(topic)
    for i, txt in enumerate(topic):
        axs.annotate(txt, (pc[i,0], pc[i,1]),
                size=10,color=colors[name[i]])
    axs.scatter(pc[:,0], pc[:, 1])   
    plt.savefig('topic_vis.jpg')
    plt.savefig('topic_vis.pdf') 

    scale = np.linalg.norm(pc) / np.linalg.norm(x)
    '''
    #for i in range(args.K):
    #    print('concept ', i, sigmas[i] * scale)

# f, axarr = plt.subplots(2,2)
# axarr[0,0].imshow(image_datas[0])
    # Title for the whole figure


    grid_size = 20
    scale = np.sqrt(var).mean()
    # Define center point and range
    #x0, y0 = 5, 5  # replace with your desired center point
    pc = pc * 20
    tt = 24
    x0, y0 = int(pc[tt,0]), int(pc[tt,1])
    #x0/=100
    #y0/=100
    x0, y0 = 0,0
    print('center', x0, y0)
    print('pc x', pc[100:150,0])
    print('pc y', pc[100:150,1])
    # norm pc[0,:], pc[1,:] to be in range [-10,10]
    # pc[0,:] = pc[0,:] - pc[0,:].mean()
    # pc[1,:] = pc[1,:] - pc[1,:].mean()
    # pc[0,:] = pc[0,:] / pc[0,:].max() * 10
    # pc[1,:] = pc[1,:] / pc[1,:].max() * 10

    r = 10  # replace with your desired range

    gflag = np.zeros((grid_size,grid_size))
    max_att = np.zeros((grid_size,grid_size))
    f, arr = plt.subplots(grid_size,grid_size,figsize=(grid_size,grid_size))
    

    f.suptitle(f'div {div}, coh {coh}', fontsize=20)  # Add your title here
    #print(pc)
    for i in range(grid_size):
        for j in range(grid_size):
            arr[i,j].axis('off')
    #print('patch', patch)
    #print nearest distance from i (i<100) to j (j>=100)
    
    
    for i in range(100):
        min_dist = 100000
        min_coord = (0,0)
        print('orig', pc[i,0], pc[i,1])
        for j in range(100,pc.shape[0]):
            dist = np.linalg.norm(pc[i]-pc[j])
            if dist < min_dist:
                min_dist = dist
                min_coord = (pc[j,0],pc[j,1])
        print('i=',i,', min_dist', min_dist, min_coord)

    patch = np.array(patch)
    mean_coord = patch.mean(axis=0)
    for i, _ in enumerate(patch):
        #if topic[i] not in top_topics:
        #    continue
        ax = int(pc[i,0]) - int(x0) + grid_size//2
        ay = int(pc[i,1]) - int(y0) + grid_size//2
        #print(ax, ay, grid_size)
        #if topic[i] == tt:
        #    print('topic ',tt, ax, ay)
        if abs(ax+1/2-grid_size/2)>=grid_size/2 or abs(ay+1/2-grid_size/2)>=grid_size/2:
            continue
        if int(gflag[ax,ay]):
            '''
            if attentions is None:
                continue
            elif attentions[i] < max_att[ax,ay]:
                continue
            else:
                max_att[ax,ay] = attentions[i]
                '''
            if np.linalg.norm(patch[i]-mean_coord) < max_att[ax,ay]:
                continue
        max_att[ax,ay] = np.linalg.norm(patch[i]-mean_coord)
        gflag[ax,ay] = 1

            
        img = arr[ax,ay].imshow(patch[i], interpolation='nearest')
        img.set_cmap('hot')
        plt.axis('off')
        

        # Removes axis numbers
        
        #arr[ax,ay].text(-0.1, 1.1, pos[i], transform=arr[ax,ay].transAxes, 
        #    size=10, weight='bold',color=colors[name[i]])
        arr[ax,ay].text(-0.1, 1.1, topic[i], transform=arr[ax,ay].transAxes, 
            size=20, weight='bold',color=colors[name[i]])
        
    # we want to center at (5, 5). So, we set the limits to go from 0 to 10.
    #arr.set_xlim(0, 10)
    #arr.set_ylim(0, 10)

    # Setting x-ticks and y-ticks to clearly see the center of image at (5,5)
    #arr.set_xticks(np.arange(0,11,1))
    #arr.set_yticks(np.arange(0,11,1))
    plt.savefig('dataset_vis.jpg')
    plt.savefig('dataset_vis.pdf')
    for ii in top_topics:
        print(indiv_coh[ii])

    
    grid_size = 20
    gflag = np.zeros((grid_size,grid_size))
    max_att = np.zeros((grid_size,grid_size))
    f, arr = plt.subplots(grid_size,grid_size,figsize=(20,20))
    #print(pc)
    for i in range(grid_size):
        for j in range(grid_size):
            arr[i,j].axis('off')
    for i, _ in enumerate(full_img):
        #if topic[i] not in top_topics:
        #    continue
        ax = int(pc[i,0]) - int(x0) + grid_size//2
        ay = int(pc[i,1]) - int(y0) + grid_size//2
        #arr[ax,ay].axis('off')
        
        if abs(ax+1/2-grid_size/2)>=grid_size/2 or abs(ay+1/2-grid_size/2)>=grid_size/2:
            continue
        if int(gflag[ax,ay]):
            if np.linalg.norm(patch[i]-mean_coord) < max_att[ax,ay]:
                continue
        max_att[ax,ay] = np.linalg.norm(patch[i]-mean_coord)
        gflag[ax,ay] = 1
        img = arr[ax,ay].imshow(full_img[i], interpolation='nearest')
        img.set_cmap('hot')
        plt.axis('off')
        # Removes axis numbers
        
        arr[ax,ay].text(-0.1, 1.1, pos[i], transform=arr[ax,ay].transAxes, 
            size=10, weight='bold',color=colors[name[i]])
        #arr[ax,ay].text(-0.1, 1.1, topic[i], transform=arr[ax,ay].transAxes, 
        #    size=20, weight='bold',color=colors[name[i]])
    plt.savefig('dataset_image_vis.jpg')
    plt.savefig('dataset_image_vis.pdf')


    grid_size = 20
    gflag = np.zeros((grid_size,grid_size))
    max_att = np.zeros((grid_size,grid_size))
    f, arr = plt.subplots(grid_size,grid_size,figsize=(20,20))
    #print(pc)
    for i in range(grid_size):
        for j in range(grid_size):
            arr[i,j].axis('off')
    for i, _ in enumerate(full_img):
        #if topic[i] not in top_topics:
        #    continue
        if pos[i] == (-1,-1):
            continue 
        ax = int(pc[i,0]) - int(x0) + grid_size//2
        ay = int(pc[i,1]) - int(y0) + grid_size//2
        #arr[ax,ay].axis('off')
        # set the pos in image i to be stripes of black and white
        for ii in range(pos[i][0]*16, pos[i][0]*16+16):
            for jj in range(pos[i][1]*16, pos[i][1]*16+16):
                if (ii+jj)%2 == 0:
                    full_img[i][ii,jj] = 0
                else:
                    full_img[i][ii,jj] = 1

        if abs(ax+1/2-grid_size/2)>=grid_size/2 or abs(ay+1/2-grid_size/2)>=grid_size/2:
            continue
        if int(gflag[ax,ay]):
            if np.linalg.norm(patch[i]-mean_coord) < max_att[ax,ay]:
                continue
        max_att[ax,ay] = np.linalg.norm(patch[i]-mean_coord)
        gflag[ax,ay] = 1
        img = arr[ax,ay].imshow(full_img[i], interpolation='nearest')
        img.set_cmap('hot')
        plt.axis('off')
        # Removes axis numbers
        
        arr[ax,ay].text(-0.1, 1.1, pos[i], transform=arr[ax,ay].transAxes, 
            size=10, weight='bold',color=colors[name[i]])
        #arr[ax,ay].text(-0.1, 1.1, topic[i], transform=arr[ax,ay].transAxes, 
        #    size=20, weight='bold',color=colors[name[i]])
    plt.savefig('dataset_image_masked_vis.jpg')
    plt.savefig('dataset_image_masked_vis.pdf')

def plot_topics(topic_patches, topic_attention, topic_image, topic_masked_image, topic_labels):
    #num_topic, num_patch = len(topic_patches), len(topic_patches[0])
    #matched_class = [17, 0, -1, 10,18, 4,8, 12,6, 7]
    order = [0,6,2,8,1,5,9,3,4,7]
    oo = [0,1,2,3,4,5,6,7,8,9]
    mp = dict(zip(order,oo))
    matched_class = [0,0,0,1,0,1,0,0,0,1]
    num_topic, num_patch = 10, 40
    width_per_subplot = 0.5
    height_per_subplot = 0.5
    custom_figsize=(num_patch * width_per_subplot, num_topic * height_per_subplot)
    f, arr = plt.subplots(num_topic, num_patch, figsize=custom_figsize)
    
    for i in range(num_topic):
        if matched_class[i] == -1:
            continue
        selected_idx = np.where(topic_labels[i] == matched_class[i])[0]
        while len(selected_idx) < num_patch:
            ll = selected_idx[-1].item()
            if ll == 199:
                ll -=50
            selected_idx = np.concatenate((selected_idx, np.array([ll+1])))
        #import pdb; pdb.set_trace()
        topic_patches[i][:len(selected_idx)] = topic_patches[i][selected_idx]
        #topic_attention[i] = topic_attention[i][selected_idx]
        topic_image[i] = [topic_image[i][j] for j in selected_idx]
        topic_masked_image[i] = [topic_masked_image[i][j] for j in selected_idx]
        topic_labels[i][:len(selected_idx)] = topic_labels[i][selected_idx] 


    #f, arr = plt.subplots(num_topic,num_patch,figsize=(num_topic,num_patch))
    for i in range(num_topic):
        for j in range(num_patch):
            arr[i,j].axis('off')
            if j >= num_patch:
                continue
            #import pdb
            #pdb.set_trace()
            topic_attention[i][j] = int(1000*topic_attention[i][j])/1000
            #arr[i,j].text(-0.1, 1.1, topic_attention[i][j], transform=arr[i,j].transAxes, 
            #size=10, weight='bold')
            img = arr[mp[i],j].imshow(topic_patches[i][j], interpolation='nearest')
            img.set_cmap('hot')
    #f.subplots_adjust(wspace=0.1, hspace=0) # Adjust the spacing between subplots here
    plt.savefig('topic_patch_vis.jpg')
    plt.savefig('topic_patch_vis.pdf')

    f, arr = plt.subplots(num_topic,num_patch,figsize=custom_figsize) #(num_topic,num_patch)
    for i in range(num_topic):
        for j in range(num_patch):
            arr[i,j].axis('off')
            if j >= num_patch:
                continue
            
            topic_attention[i][j] = int(1000*topic_attention[i][j])/1000
            #arr[i,j].text(-0.1, 1.1, topic_attention[i][j], transform=arr[i,j].transAxes, 
            #size=10, weight='bold')
            img = arr[mp[i],j].imshow(topic_image[i][j], interpolation='nearest')
            img.set_cmap('hot')

    plt.savefig('topic_image_vis.jpg')
    plt.savefig('topic_image_vis.pdf')


    f, arr = plt.subplots(num_topic,num_patch,figsize=custom_figsize)
    for i in range(num_topic):
        for j in range(num_patch):
            arr[i,j].axis('off')
            if j >= num_patch:
                continue
            
            topic_attention[i][j] = int(1000*topic_attention[i][j])/1000
            arr[i,j].text(-0.1, 1.1, topic_labels[i][j], transform=arr[i,j].transAxes, 
            size=5, weight='bold')
            img = arr[mp[i],j].imshow(topic_masked_image[i][j], interpolation='nearest')
            img.set_cmap('hot')

    plt.savefig('topic_masked_image_vis.jpg')
    plt.savefig('topic_masked_image_vis.pdf')
    return f, arr


def run_kmeans(X,K):
    # initialize centers, return mean and variance
    #centers = kmeans_init(X,K)
    # run kmeans algorithm for 1/2 iterations and return the mean and variance
    kmeans = KMeans(n_clusters=args.K,random_state=0, n_init=args.K, max_iter=10)
    ret = kmeans.fit(X)
    mean = ret.cluster_centers_
    variance  = 0
    print(mean.mean(0))
    print(X.mean(axis=0))
    return mean#, variance



def kmeans_init_prev(X,K, init_idx=None):
    centers = []
    index = []
    avg = 0
    if init_idx is None:
        init_idx = np.random.randint(0,len(X))
        avg = X[init_idx]
        centers.append(avg)
    
    else:
        for idx, j in enumerate(init_idx):
            avg = (avg*idx + X[j])/(idx+1)
            centers.append(X[j])
            index.append(j)
    #init_len = len(centers)
    for idx in range(K):
        if len(centers) == K:
            break
        dist = 0
        new_center = None
        new_index = None
        for j in range(len(X)):
            if np.linalg.norm(X[j]-avg) > dist:
                dist = np.linalg.norm(X[j]-avg)
                new_center = X[j]
                new_index = j
                print(idx, dist)
        avg = (avg*len(centers) + new_center)/(len(centers)+1)
        centers.append(new_center)
        index.append(new_index)



    return centers, index

def kmeans_init(X, K, init_idx=None):
    centers = []
    index = []

    # If initial indices are not provided, select one data point randomly
    if init_idx is None:
        init_idx = [np.random.randint(0, len(X))]

    # Initialize the centers and indices using provided or random indices
    for idx in init_idx:
        centers.append(X[idx])
        index.append(idx)

    # Fill in the remaining centers
    while len(centers) < K:
        avg = np.mean(centers, axis=0)  # Calculate the average of current centers
        dist = 0
        new_center = None
        new_index = None
        
        for j, x in enumerate(X):
            # Check if point is not already a center and is farther from the average
            if j not in index and np.linalg.norm(x - avg) > dist:
                dist = np.linalg.norm(x - avg)
                new_center = x
                new_index = j
        
        if new_center is not None:
            centers.append(new_center)
            index.append(new_index)
        print(len(centers), dist)
    return centers, index

def softmax(x): # 2D 
    """Compute softmax values for each sets of scores in x."""
    # x (B,d)
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)
    

'''
def row_norms(X, squared=False):
    """Row-wise (squared) Euclidean norm of X.
    Equivalent to np.sqrt((X * X).sum(axis=1)), but also supports sparse
    matrices and does not create an X.shape-sized temporary.
    Performs no input validation.
    Parameters
    ----------
    X : array-like
        The input array.
    squared : bool, default=False
        If True, return squared norms.
    Returns
    -------
    array-like
        The row-wise (squared) Euclidean norm of X.
    """
    if sparse.issparse(X):
        if not isinstance(X, sparse.csr_matrix):
            X = sparse.csr_matrix(X)
        norms = csr_row_norms(X)
    else:
        norms = np.einsum("ij,ij->i", X, X)

    if not squared:
        np.sqrt(norms, norms)
    return norms

def _euclidean_distances(X, Y, X_norm_squared=None, Y_norm_squared=None, squared=False):
    """Computational part of euclidean_distances
    Assumes inputs are already checked.
    If norms are passed as float32, they are unused. If arrays are passed as
    float32, norms needs to be recomputed on upcast chunks.
    TODO: use a float64 accumulator in row_norms to avoid the latter.
    """
    if X_norm_squared is not None:
        if X_norm_squared.dtype == np.float32:
            XX = None
        else:
            XX = X_norm_squared.reshape(-1, 1)
    elif X.dtype == np.float32:
        XX = None
    else:
        XX = row_norms(X, squared=True)[:, np.newaxis]

    if Y is X:
        YY = None if XX is None else XX.T
    else:
        if Y_norm_squared is not None:
            if Y_norm_squared.dtype == np.float32:
                YY = None
            else:
                YY = Y_norm_squared.reshape(1, -1)
        elif Y.dtype == np.float32:
            YY = None
        else:
            YY = row_norms(Y, squared=True)[np.newaxis, :]

    if X.dtype == np.float32:
        # To minimize precision issues with float32, we compute the distance
        # matrix on chunks of X and Y upcast to float64
        distances = _euclidean_distances_upcast(X, XX, Y, YY)
    else:
        # if dtype is already float64, no need to chunk and upcast
        distances = -2 * safe_sparse_dot(X, Y.T, dense_output=True)
        distances += XX
        distances += YY
    np.maximum(distances, 0, out=distances)

    # Ensure that distances between vectors and themselves are set to 0.0.
    # This may not be the case due to floating point rounding errors.
    if X is Y:
        np.fill_diagonal(distances, 0)

    return distances if squared else np.sqrt(distances, out=distances)


# ref: https://github.com/scikit-learn/scikit-learn/blob/baf0ea25d/sklearn/cluster/_kmeans.py#L154

def _kmeans_plusplus(X, n_clusters, x_squared_norms, random_state, n_local_trials=None):
    """Computational component for initialization of n_clusters by
    k-means++. Prior validation of data is assumed.
    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The data to pick seeds for.
    n_clusters : int
        The number of seeds to choose.
    x_squared_norms : ndarray of shape (n_samples,)
        Squared Euclidean norm of each data point.
    random_state : RandomState instance
        The generator used to initialize the centers.
        See :term:`Glossary <random_state>`.
    n_local_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.
    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        The initial centers for k-means.
    indices : ndarray of shape (n_clusters,)
        The index location of the chosen centers in the data array X. For a
        given index and center, X[index] = center.
    """
    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly and track index of point
    center_id = random_state.randint(n_samples)
    indices = np.full(n_clusters, -1, dtype=int)
    if sp.issparse(X):
        centers[0] = X[center_id].toarray()
    else:
        centers[0] = X[center_id]
    indices[0] = center_id

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = _euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms, squared=True
    )
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.uniform(size=n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq), rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = _euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True
        )

        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]
        indices[c] = best_candidate

    return centers, indices

'''