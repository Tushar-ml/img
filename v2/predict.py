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


from pathlib import Path
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

from trainer import FashionModel


def log(*s):
    print(*s, flush=True)


def log2(*s):
    print(*s, flush=True)

def logw(*s):
    print(*s, flush=True)


def loge(*s):
    print(*s, flush=True)





from datasets import FashionImageDataset

from models import FashionModel



def encode_label(df, dfx, column):
    if column not in df.columns:
        log(f"column: {column} nor present in dataframe")
    le = preprocessing.LabelEncoder()
    le.fit(df[column].tolist())
    # log("Number of classes: ", len(le.classes_))
    return le.transform(dfx[column].tolist())


def decode_label(df, dfx, column):
    if column not in df.columns:
        log(f"column: {column} nor present in dataframe")
    le = preprocessing.LabelEncoder()
    le.fit(df[column].tolist())
    # log("Number of classes: ", len(le.classes_))
    return list(le.inverse_transform(dfx[column].tolist()))



def config_get_paths(cfg, m_config=None, name='preprocess/train/predict'):

    if 'preprocess' in name :
        config_preprocess = cfg['preprocess']
        IMAGE_PATH = os.path.join(cfg['base_dir'], cfg['input_dir'], cfg['dataset_name'], cfg['train_dir'])
        CSV_PATH = os.path.join(cfg['base_dir'], cfg['input_dir'], cfg['dataset_name'], cfg['csv_dir'])

        return IMAGE_PATH, CSV_PATH


    if 'train' in name or 'predict' in name:
        CSV_PATH = os.path.join(cfg['base_dir'], cfg['input_dir'], cfg['dataset_name'], cfg['csv_dir'])
        MODEL_PATH = os.path.join(cfg['base_dir'], cfg['models_dir'], cfg['dataset_name'], \
                                  m_config['model_name'], str(m_config['version']))

        if cfg['feature_extractor']['input_data_type'] == 'train':
            if m_config['model_name'] == 'smallmodel':
                INPUT_PATH = os.path.join(cfg['base_dir'], cfg['vector_dir'], cfg['dataset_name'], m_config['input_data_dir'])
            else:
                INPUT_PATH = os.path.join(cfg['base_dir'], cfg['input_dir'], cfg['dataset_name'], cfg['train_dir'])
            OUTPUT_PATH = os.path.join(cfg['base_dir'], cfg['vector_dir'], cfg['dataset_name'], m_config['model_name'])

        else:
            if m_config['model_name'] == 'smallmodel':
                INPUT_PATH = os.path.join(cfg['base_dir'], cfg['vector_dir'], cfg['dataset_name'], "test_" + m_config['input_data_dir'])
            else:
                INPUT_PATH = os.path.join(cfg['base_dir'], cfg['input_dir'], cfg['dataset_name'], cfg['test_dir'])
            OUTPUT_PATH = os.path.join(cfg['base_dir'], cfg['vector_dir'], cfg['dataset_name'], "test_" + m_config['model_name'])


        log(INPUT_PATH)
        os.makedirs(MODEL_PATH, exist_ok=True)
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        return   CSV_PATH, MODEL_PATH, INPUT_PATH, OUTPUT_PATH



def generate_dataset(config):
    
    cfg2 = config['preprocess']

    IMAGE_PATH = os.path.join(config['base_dir'], config['input_dir'], config['dataset_name'], config['train_dir'])

    CSV_PATH = os.path.join(config['base_dir'], config['input_dir'], config['dataset_name'], config['csv_dir'])

    os.makedirs(CSV_PATH, exist_ok=True)
    
    dfx = pd.read_csv(os.path.join(CSV_PATH, cfg2['base_df']), error_bad_lines=False)
    log("Dataset Size: ", dfx.shape)
    dfx = dfx.dropna()
    dfx = dfx[['id', 'gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage']]

    file_not_found = []
    for i in dfx.id.tolist():
        flag = False
        for ext in cfg2['image_exts']:
            if os.path.isfile(os.path.join(IMAGE_PATH, str(i) + f".{ext}")):
                flag = True
                break
        if not flag:
            file_not_found.append(i)
    log(f"Total {len(file_not_found)} images didn't found")

    
    dfx = dfx[~dfx['id'].isin(file_not_found)]
    dfx.to_csv(os.path.join(CSV_PATH, cfg2['base_df'].split('.')[0] + '_df.csv'), index=False)
    articleTypes = []
    for k, v in dict(dfx.articleType.value_counts()).items():
        if v <= 10:
            articleTypes.append(k)
    dfx = dfx[~dfx['articleType'].isin(articleTypes)]

    # dfx.to_csv(config_preprocess['preprocessed_df_with_classes'], index=False)
    dfx.to_csv(os.path.join(CSV_PATH, cfg2['preprocessed_df_with_classes']), index=False)

    for col in dfx.columns.tolist()[1:]:
        dfx[col] = encode_label(dfx, dfx, col)
    
    log("Final Dataset Size: ", dfx.shape)
            
    # dfx.to_csv(config_preprocess['preprocessed_df'], index=False)
    dfx.to_csv(os.path.join(CSV_PATH, cfg2['preprocessed_df']), index=False)
    train_dfx, val_dfx = train_test_split(dfx, test_size=cfg2["val_set_size"], random_state=42)

    # train_dfx.to_csv(config_preprocess['train_df'], index=False)
    # val_dfx.to_csv(config_preprocess['val_df'], index=False)
    train_dfx.to_csv(os.path.join(CSV_PATH, cfg2['train_df']), index=False)
    val_dfx.to_csv(os.path.join(CSV_PATH, cfg2['val_df']), index=False)
    log("Stage: df_preprocess Finished ------------------------------------------------")





######################################################################
if __name__ == "__main__":
    import fire
    fire.Fire(generate_dataset)



