import os
import albumentations
import  matplotlib.pyplot as plt
import pandas as pd

import tez

from tez.datasets import ImageDataset
from tez.callbacks import EarlyStopping

import torch
import torch.nn as nn

import torchvision

from sklearn import metrics, model_selection
from efficientnet_pytorch import EfficientNet
from pathlib import Path
import argparse
import os

import albumentations
import pandas as pd
import tez
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from sklearn import metrics, model_selection, preprocessing
from tez.callbacks import EarlyStopping, Callback
from tez.datasets import ImageDataset
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

import numpy as np

import cv2

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image


import os
import torch

import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm

import subprocess
import torch
import numpy as np

import os
import torch

import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm

from PIL import Image



def get_features(model, dataloader, device='cuda'):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader)):
            features = model.encode_image(data['image'].to(device))
            print(features.shape)
            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()



class FashionImageDataset(Dataset):

    def __init__(self, IMAGE_PATH, DF_PATH, aug=None):
        """
        Args:
            IMAGE_PATH (string): Directory with all the images or vectors
            DF_PATH (string): Path to csv file
            aug: augumentation
            
        """
        self.image_dir = IMAGE_PATH
        self.df = pd.read_csv(DF_PATH)
    

        self.images = [str(i) + '.jpg' for i in self.df.id.tolist()]
        self.aug = aug

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        filename =self.images[idx]

        # img = cv2.imread(os.path.join(self.image_dir, filename))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img,(256,256))
        # img = torch.from_numpy(self.aug(image=img)['image']).type(torch.FloatTensor)
        img = Image.open(os.path.join(self.image_dir, filename))
        img = self.aug(img).type(torch.FloatTensor)
        # img = img.permute(2, 0, 1)
        # print(img.shape)

        return {'image': img, 'filename': filename.split('.')[0]}



if __name__ == "__main__":


    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default='./')
    parser.add_argument("--image_path", type=str, default='images')
    parser.add_argument("--big_model_vector_path", type=str, default='big_model1_vectors')
    parser.add_argument("--model_name", type=str, default='model.pt')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()


    BASE_DIR              = Path(args.base_path)
    IMAGE_PATH            = Path(args.image_path)
    BIG_MODEL_VECTOR_PATH =  Path(args.big_model_vector_path)
    
    os.makedirs(BIG_MODEL_VECTOR_PATH, exist_ok=True)

    dfx = pd.read_csv(BASE_DIR/'df_final.csv')


    model = torch.jit.load(args.model_name).cuda().eval()

    input_resolution = model.input_resolution.item()
    context_length   = model.context_length.item()
    vocab_size       = model.vocab_size.item()

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)


    preprocess = Compose([
        Resize(input_resolution, interpolation=Image.BICUBIC),
        CenterCrop(input_resolution),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


    fashion_dataset    = FashionImageDataset(IMAGE_PATH=IMAGE_PATH, DF_PATH=BASE_DIR/'df_final.csv', aug = preprocess)
    fashion_dataloader = DataLoader(fashion_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)


    for i, data in tqdm(enumerate(fashion_dataloader), total=len(fashion_dataloader)):
            image_vector = model.encode_image(data['image'].to(args.device)).cpu().detach().numpy()
            for i in range(len(data['filename'])):
                f_name = data['filename'][i]
                np.save(os.path.join(BIG_MODEL_VECTOR_PATH, f_name), image_vector[i])





   
