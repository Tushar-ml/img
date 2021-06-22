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


from Path import  pathlib
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




def config_load(config_path: str = None):
    """Load Config file into a dict
    1) load config_path
    2) If not, load in HOME USER
    3) If not, create default one
    Args:
        config_path: path of config or 'default' tag value
    Returns: dict config
    """
    path_default = pathlib.Path.home() / ".mygenerator"
    config_path_default = path_default / "config.yaml"

    if config_path is None or config_path == "default":
        logw(f"Using config: {config_path_default}")
        config_path = config_path_default

    try:
        log2("loading config", config_path)
        return yaml.load(config_path.read_text(), Loader=yaml.Loader)

    except Exception as e:
        logw(f"Cannot read yaml file {config_path}", e)

    logw("#### Using default configuration")
    config_default = {
        "current_dataset": "mnist",
        "datasets": {
            "mnist": {
                "url": "https://github.com/arita37/mnist_png/raw/master/mnist_png.tar.gz",
                "path": str(path_default / "mnist_png" / "training"),
            }
        },
    }
    log2(config_default)

    log(f"Creating config file in {config_path_default}")
    os.makedirs(path_default, exist_ok=True)
    with open(config_path_default, mode="w") as fp:
        json.dump(config_default, fp)
    return config_default






def dataset_donwload(url, path_target):
    """Donwload on disk the tar.gz file
    Args:
        url:
        path_target:
    Returns:
    """
    log(f"Donwloading mnist dataset in {path_target}")
    os.makedirs(path_target, exist_ok=True)
    wget.download(url, path_target)
    tar_name = url.split("/")[-1]
    os_extract_archive(path_target + "/" + tar_name, path_target)
    log2(path_target)
    return path_target + tar_name





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
    fire.Fire()



