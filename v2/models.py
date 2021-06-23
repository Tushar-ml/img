import argparse
import os
import warnings
from pathlib import Path

import albumentations
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tez
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from efficientnet_pytorch import EfficientNet, EfficientNet
from sklearn import metrics, metrics, model_selection, model_selection, preprocessing
from sklearn.model_selection import train_test_split
from tez.callbacks import Callback, EarlyStopping, EarlyStopping
from tez.datasets import ImageDataset, ImageDataset
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from tqdm import tqdm

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import pickle

import json
import sys
from PIL import Image

from nfnets import pretrained_nfnet, SGD_AGC

import deepspeed

from torch.nn import functional as F



from datasets import FashionImageDataset


#######################################################################################################
#                                       NFNets Model                                                  #
#######################################################################################################
class FashionModel(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        self.config = config
        
        self.in_features = config['in_features']

        self.intermediate_features = config['intermediate_features']

        self.num_classes = num_classes

        self.dropout_prob = config['dropout']

        self.model_name = config['model_name']



        if self.model_name == 'efficientnet':
            self.model = EfficientNet.from_pretrained('efficientnet-b3')
            # self.out_feature_size = self.effnet._conv_head.out_channels
        elif self.model_name == 'nfnets':
            self.model = pretrained_nfnet(config['model_path'])
            self.model = torch.nn.Sequential(*(list(self.model.children())[:-1] + [nn.AdaptiveMaxPool2d(1)]))
        
        self.dropout = nn.Dropout(self.dropout_prob)
        self.relu = nn.ReLU()

        # Layer 1
        self.linear1        = nn.Linear(in_features=self.in_features, out_features=256, bias=False)
        
        # Layer 2
        self.linear2        = nn.Linear(in_features=256, out_features=self.intermediate_features, bias=False)

        self.gender         = nn.Linear(self.intermediate_features, self.num_classes['gender'])
        self.masterCategory = nn.Linear(self.intermediate_features, self.num_classes['masterCategory'])
        self.subCategory    = nn.Linear(self.intermediate_features, self.num_classes['subCategory'])
        self.articleType    = nn.Linear(self.intermediate_features, self.num_classes['articleType'])
        self.baseColour     = nn.Linear(self.intermediate_features, self.num_classes['baseColour'])
        self.season         = nn.Linear(self.intermediate_features, self.num_classes['season'])
        self.usage          = nn.Linear(self.intermediate_features, self.num_classes['usage'])
       
        self.step_scheduler_after = "epoch"

    def monitor_metrics(self, outputs, targets):
        if targets is None:
            return {}
        accuracy = []
        for k,v in outputs.items():
            out  = outputs[k]
            targ = targets[k]
            # print(out)
            out  = torch.argmax(out, dim=1).cpu().detach().numpy()
            targ = targ.cpu().detach().numpy()
            accuracy.append(metrics.accuracy_score(targ, out))
        return {'accuracy': sum(accuracy)/len(accuracy)}

    def forward(self, image, genderLabel=None, masterCategoryLabel=None, subCategoryLabel=None, 
                articleTypeLabel=None, baseColourLabel=None, seasonLabel=None, usageLabel=None):
        batch_size = image.shape[0]

        if self.model_name == 'nfnets':
            x = self.model(image).view(batch_size, -1)
        elif self.model_name == 'efficientnet':
            x = self.model.extract_features(image)
            x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
            
        else:
            x = image

        x = self.relu(self.linear1(self.dropout(x)))
        x = self.relu(self.linear2(self.dropout(x)))

        targets = {}
        if genderLabel is None:
            targets = None
        else:
            targets['gender']         = genderLabel
            targets['masterCategory'] = masterCategoryLabel
            targets['subCategory']    = subCategoryLabel
            targets['articleType']    = articleTypeLabel
            targets['baseColour']     = baseColourLabel
            targets['season']         = seasonLabel
            targets['usage']          = usageLabel
        outputs                   = {}
        outputs["gender"]         = self.gender(x)
        outputs["masterCategory"] = self.masterCategory(x)
        outputs["subCategory"]    = self.subCategory(x)
        outputs["articleType"]    = self.articleType(x)
        outputs["baseColour"]     = self.baseColour(x)
        outputs["season"]         = self.season(x)
        outputs["usage"]          = self.usage(x)

        if targets is not None:
            loss = []
            for k,v in outputs.items():
                loss.append(nn.CrossEntropyLoss()(outputs[k], targets[k]))
            loss = sum(loss)
            metrics = self.monitor_metrics(outputs, targets)
            return outputs, loss, metrics
        return outputs, None, None
    
    def extract_features(self, image):
        batch_size = image.shape[0]

        if self.model_name == 'nfnets':
            x = self.model(image).view(batch_size, -1)
        elif self.model_name == 'efficientnet':
            x = self.model.extract_features(image)
            x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
            
        else:
            x = image

        x = self.relu(self.linear1(self.dropout(x)))
        x = self.relu(self.linear2(self.dropout(x)))

        return x



if __name__ == "__main__":
   import fire
   fire.Fire()