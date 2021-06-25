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



# =====================================================================
# Dataset                                                        =
# =====================================================================
from datasets import FashionImageDataset

from models import FashionModel


#######################################################################################################
#                                       NFNets Model                                                  #
#######################################################################################################

"""
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

    def forward(self, image, genderLabel=None, masterCategoryLabel=None, subCategoryLabel=None, articleTypeLabel=None, baseColourLabel=None, seasonLabel=None, usageLabel=None):
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
"""
    
def add_argument():

    parser = argparse.ArgumentParser(description='FASHION')

    #data
    # cuda
    parser.add_argument('--with_cuda',
                        default=False,
                        action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema',
                        default=False,
                        action='store_true',
                        help='whether use exponential moving average')

    # train
    parser.add_argument('-b',
                        '--batch_size',
                        default=32,
                        type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e',
                        '--epochs',
                        default=30,
                        type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')

    parser.add_argument('--model',
                        type=str,
                        default="nfnets",
                        help='cfg file')

    parser.add_argument('--cfg',
                        type=str,
                        default="./ds_config.json",
                        help='cfg file')
    try:
        parser_d = deepspeed.add_config_arguments(parser)
        args = parser_d.parse_args()
    except:
        args = parser.parse_args()

    return args



def train_normal():
    pass


def train_deepspeed():
    pass



def run():
    """
       pip install fire

    """
    args = add_argument()

    #import utilmy
    #args = utilmy.to_namespace(ddict)

    if args.config != "":
        with open(args.config) as f:
            config = json.load(f)["custom_params"]
    else:
        with open(args.deepspeed_config) as f:
            config = json.load(f)["custom_params"]

    # m_config = cfg['model_params'][cfg["model_name"]]
    model_config = config['model_params'][args.model]


    CSV_PATH = os.path.join(config['base_dir'], config['input_dir'], config['dataset_name'], config['csv_dir'])

    MODEL_PATH = os.path.join(config['base_dir'], config['models_dir'], config['dataset_name'], model_config['model_name'], str(model_config['version']))
    os.makedirs(MODEL_PATH, exist_ok=True)
    # print(cfg)
    dfx = pd.read_csv(os.path.join(CSV_PATH, config['root_df']))
    class_dict = dict([[i, dfx[i].nunique()] for i in dfx.columns.tolist()[1:]])

    if model_config['model_name'] in ['efficientnet', 'nfnets']:
        IMAGE_PATH = os.path.join(config['base_dir'], config['input_dir'], config['dataset_name'], config['train_dir'])
        train_dataset = FashionImageDataset(IMAGE_PATH=IMAGE_PATH, DF_PATH= os.path.join(CSV_PATH, config['train_df']), config=config, use_features=False)
        val_dataset = FashionImageDataset(IMAGE_PATH=IMAGE_PATH, DF_PATH=os.path.join(CSV_PATH, config['val_df']), config=config, use_features=False)
    else:
        IMAGE_PATH = os.path.join(config['base_dir'], config['vector_dir'], config['dataset_name'], model_config['input_data_dir'])
        train_dataset = FashionImageDataset(IMAGE_PATH=IMAGE_PATH, DF_PATH= os.path.join(CSV_PATH, config['train_df']), config=config)
        val_dataset   = FashionImageDataset(IMAGE_PATH=IMAGE_PATH, DF_PATH= os.path.join(CSV_PATH, config['val_df']), config=config)


    train_dl = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8)
    val_dl = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=8)


    model = FashionModel(config=model_config, num_classes=class_dict)
    try:
        config["use_deepspeed"] = True if args.deepspeed_config else False
    except:
        config["use_deepspeed"] = False

    print(config)
    if not config['use_deepspeed']:
        ##################################################################################################################
        #                                          Train with PyTorch                                                    #
        ##################################################################################################################
        print("============================================")
        print("Training model with pytorch")
        print("============================================")
        model.to(config['device']);

        if model_config['model_name'] == 'nfnets':
            optimizer = SGD_AGC(
                    named_params=model.named_parameters(), # Pass named parameters
                    lr=config['lr_rate'],
                    momentum=0.9,
                    clipping=0.1, # New clipping parameter
                    weight_decay=config['weight_decay'],
                    nesterov=True)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr_rate'])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1)

        best_loss = float('inf')
        for epoch in range(config['epochs']):  # loop over the dataset multiple times
            tk0 = tqdm(train_dl, total=len(train_dl))
            losses = []
            monitor = []

            model.train()

            for i, data in enumerate(tk0):
                for key, value in data.items():
                    data[key] = value.to(config['device'])

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs, loss, metric = model(**data)

                losses.append(loss.item())
                monitor.append(metric['accuracy'])
                loss.backward()
                optimizer.step()
                tk0.set_postfix(loss=round(sum(losses)/len(losses), 2), stage="train", accuracy=round(sum(monitor)/len(monitor), 3))
            tk0.close()
            scheduler.step()
            tk0 = tqdm(val_dl, total=len(val_dl))
            val_losses = []
            val_monitor = []
            model.eval()
            for i, data in enumerate(tk0):
                for key, value in data.items():
                    data[key] = value.to(config['device'])
                optimizer.zero_grad()
                with torch.no_grad():
                    outputs, loss, metric = model(**data)
                val_losses.append(loss.item())
                val_monitor.append(metric['accuracy'])
                tk0.set_postfix(loss=round(sum(val_losses)/len(val_losses), 2), stage="valid", accuracy=round(sum(val_monitor)/len(val_monitor), 3))
            tk0.close()
            curr_loss = sum(val_losses)/len(val_losses)
            if curr_loss < best_loss:
                model_dict = {}
                model_dict["state_dict"] = model.state_dict()
                torch.save(model_dict, os.path.join(MODEL_PATH, f"checkpoint_best.pt"))
                print(f"Model Saved | Loss impoved from {best_loss} -----> {curr_loss}")
                best_loss = curr_loss
                counter = 0
            else:
                counter += 1
            # print((epoch+1) % cfg['save_interval'])
            if (epoch+1) % config['save_interval'] ==  0:
                model_dict = {}
                model_dict["state_dict"] = model.state_dict()
                torch.save(model_dict, os.path.join(MODEL_PATH, f"checkpoint_epoch_{epoch+1}.pt"))
                print(f"Model Saved")
            if counter == config['patience']:
                print(f"Model not improved from {config['patience']} epochs...............")
                print("Training Finished..................")
                break
        torch.save(model_dict, os.path.join(MODEL_PATH, f"checkpoint_last.pt"))
        print('Finished Training')


    else:
        ##################################################################################################################
        #                                          Train with Deepspeed                                                  #
        ##################################################################################################################
        print("============================================")
        print("Training model with deepspeed")
        print("============================================")
        config["model_name"] = config["model_name"] + "_deepspeed"



        parameters = filter(lambda p: p.requires_grad, model.parameters())


        model_engine, optimizer, trainloader, __ = deepspeed.initialize(args=args,
                    model=model, model_parameters=parameters, training_data=train_dataset)
        best_loss = float('inf')
        for epoch in range(config['epochs']):  # loop over the dataset multiple times
            tk0 = tqdm(train_dl, total=len(train_dl))
            losses = []
            monitor = []

            model.train()

            for i, data in enumerate(tk0):
                # get the inputs; data is a list of [inputs, labels]
                for key, value in data.items():
                    data[key] = value.to(model_engine.local_rank)

                outputs, loss, metric = model_engine(**data)

                losses.append(loss.item())
                monitor.append(metric['accuracy'])
                model_engine.backward(loss)
                model_engine.step()

                tk0.set_postfix(loss=round(sum(losses)/len(losses), 2), stage="train", accuracy=round(sum(monitor)/len(monitor), 3))
            tk0.close()
            tk0 = tqdm(val_dl, total=len(val_dl))
            val_losses = []
            val_monitor = []
            model.eval()
            for i, data in enumerate(tk0):
                for key, value in data.items():
                    data[key] = value.to(model_engine.local_rank)
                optimizer.zero_grad()
                with torch.no_grad():
                    outputs, loss, metric = model(**data)
                val_losses.append(loss.item())
                val_monitor.append(metric['accuracy'])
                tk0.set_postfix(loss=round(sum(val_losses)/len(val_losses), 2), stage="valid", accuracy=round(sum(val_monitor)/len(val_monitor), 3))
            tk0.close()
            curr_loss = sum(val_losses)/len(val_losses)
            if curr_loss < best_loss:
                model_dict = {}
                model_dict["state_dict"] = model.state_dict()
                # torch.save(model_dict, os.path.join(cfg['base_dir'], 'models', f"checkpoint_best.pt"))
                torch.save(model_dict, os.path.join(MODEL_PATH, f"checkpoint_best.pt"))
                print(f"Model Saved | Loss impoved from {best_loss} -----> {curr_loss}")
                best_loss = curr_loss
                counter = 0
            else:
                counter += 1
            # print((epoch+1) % cfg['save_interval'])
            if (epoch+1) % config['save_interval'] ==  0:
                model_dict = {}
                model_dict["state_dict"] = model.state_dict()
                torch.save(model_dict, os.path.join(MODEL_PATH, f"checkpoint_epoch_{epoch+1}.pt"))
                print(f"Model Saved")
            if counter == config['patience']:
                print(f"Model not improved from {config['patience']} epochs...............")
                print("Training Finished..................")
                break
        # torch.save(model_dict, os.path.join(cfg['base_dir'], 'models', f"{cfg['model_name']}_last.pt"))
        torch.save(model_dict, os.path.join(MODEL_PATH, f"checkpoint_last.pt"))
        print('Finished Training')




if __name__ == "__main__":
   import fire
   fire.Fire()