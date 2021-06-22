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

from trainer import FashionModel


def log(*s):
    print(*s, flush=True)


# =====================================================================
# Dataset for Big Model1                                                =
# =====================================================================

class FashionImageDataset(Dataset):
    def __init__(self, IMAGE_PATH, config, m_config, aug=None):
        """
        Args:
            IMAGE_PATH (string): Directory with all the images or vectors
            DF_PATH (string): Path to csv file
            aug: augumentation

        """
        self.image_dir = IMAGE_PATH
        self.config = config
        self.m_config = m_config

        if self.config['sample']:
            if self.m_config['model_name'] == "smallmodel":
                self.images = list(Path(IMAGE_PATH).glob("*.npy"))[:200]
            else:
                self.images = list(Path(IMAGE_PATH).glob("*.jpg"))[:200]
        else:
            if self.m_config['model_name'] == "smallmodel":
                self.images = list(Path(IMAGE_PATH).glob("*.npy"))
            else:
                self.images = list(Path(IMAGE_PATH).glob("*.jpg"))

        if aug is None:
              self.aug = Compose([
                    Resize((224, 224), interpolation=Image.BICUBIC),
                    # CenterCrop((224, 224)),
                    ToTensor(),
                    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])
        else:
            self.aug = aug


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        filename =self.images[idx]
        if self.m_config['model_name'] == "smallmodel":
            img = np.load(filename).astype(np.float32)
            img = torch.from_numpy(img)
        else:
            img = Image.open(filename).convert('RGB')
            img = self.aug(img).type(torch.FloatTensor)

        return {'image': img, 'filename': str(filename.stem)}


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




def feature_extractor(cfg, m_config):

    CSV_PATH = os.path.join(cfg['base_dir'], cfg['input_dir'], cfg['dataset_name'], cfg['csv_dir'])
    MODEL_PATH = os.path.join(cfg['base_dir'], cfg['models_dir'], cfg['dataset_name'], \
                              m_config['model_name'], str(m_config['version']))

    if cfg['feature_extractor']['input_data_type'] == 'train':
        if m_config['model_name'] == 'smallmodel':
            INPUT_PATH = os.path.join(cfg['base_dir'], cfg['vector_dir'], cfg['dataset_name'], m_config['input_data_dir'])

        else:
            INPUT_PATH = os.path.join(cfg['base_dir'], cfg['input_dir'], cfg['dataset_name'], cfg['train_dir'])

        MODEL_PATH = os.path.join(MODEL_PATH, 'checkpoint_best.pt')
        OUTPUT_PATH = os.path.join(cfg['base_dir'], cfg['vector_dir'], cfg['dataset_name'], m_config['model_name'])

    else:
        if m_config['model_name'] == 'smallmodel':
            INPUT_PATH = os.path.join(cfg['base_dir'], cfg['vector_dir'], cfg['dataset_name'], "test_" + m_config['input_data_dir'])
        else:
            INPUT_PATH = os.path.join(cfg['base_dir'], cfg['input_dir'], cfg['dataset_name'], cfg['test_dir'])

        MODEL_PATH = os.path.join(MODEL_PATH, 'checkpoint_best.pt')
        OUTPUT_PATH = os.path.join(cfg['base_dir'], cfg['vector_dir'], cfg['dataset_name'], "test_" + m_config['model_name'])

    log(INPUT_PATH)
    os.makedirs(OUTPUT_PATH, exist_ok=True)


    log("########### Feature Extraction Started ############################################")
    if m_config['model_name'] == 'bigmodel':

        MODEL_PATH = m_config["model_path"]

        def get_features(model, dataloader, device='cuda'):
            all_features = []
            all_labels = []
            with torch.no_grad():
                for i, data in tqdm(enumerate(dataloader)):
                    features = model.encode_image(data['image'].to(device))
                    log(features.shape)
                    all_features.append(features)
                    all_labels.append(labels)
        
            return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

        dfx = pd.read_csv(os.path.join(CSV_PATH, cfg['root_df']))
        
        model = torch.jit.load(MODEL_PATH).to(cfg["device"]).eval()
        
        input_resolution = model.input_resolution.item()
        context_length   = model.context_length.item()
        vocab_size       = model.vocab_size.item()
        
        log("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
        log("Input resolution:", input_resolution)
        log("Context length:", context_length)
        log("Vocab size:", vocab_size)
        
        
        preprocess = Compose([
            Resize(input_resolution, interpolation=Image.BICUBIC),
            CenterCrop(input_resolution),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        
        
        fashion_dataset    = FashionImageDataset(IMAGE_PATH=INPUT_PATH, config=cfg, m_config=m_config, aug = preprocess)
        fashion_dataloader = DataLoader(fashion_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=8)
        
        
        for i, data in tqdm(enumerate(fashion_dataloader), total=len(fashion_dataloader)):
                image_vector = model.encode_image(data['image'].to(cfg["device"])).cpu().detach().numpy()
                for i in range(len(data['filename'])):
                    f_name = data['filename'][i]
                    np.save(os.path.join(OUTPUT_PATH, f_name), image_vector[i])
    else:

        MODEL_PATH = os.path.join(MODEL_PATH, 'checkpoint_best.pt')

        dfx = pd.read_csv(os.path.join(CSV_PATH, cfg['root_df']))
        class_dict = dict([[i, dfx[i].nunique()] for i in dfx.columns.tolist()[1:]])

        if m_config['model_name'] in ['efficientnet', 'nfnets']:
            fashion_dataset = FashionImageDataset(IMAGE_PATH=INPUT_PATH, config=cfg, m_config=m_config)
        else:
            fashion_dataset = FashionImageDataset(IMAGE_PATH=INPUT_PATH, config=cfg, m_config=m_config)

        fashion_dataloader = DataLoader(fashion_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=8)

        model = FashionModel(config=m_config, num_classes=class_dict)
        model.load_state_dict(torch.load(MODEL_PATH)['state_dict'])
        model.to(cfg["device"])
        model.eval()

        for i, data in tqdm(enumerate(fashion_dataloader), total=len(fashion_dataloader)):
            image_vector = model.extract_features(data['image'].to(cfg["device"])).cpu().detach().numpy()
            for i in range(len(data['filename'])):
                f_name = data['filename'][i]
                np.save(os.path.join(OUTPUT_PATH, f_name), image_vector[i])
    log("########### Feature Extraction Finished ###########")



def inference(cfg, m_config):

    log("########### Prediction Started ###########")

    CSV_PATH = os.path.join(cfg['base_dir'], cfg['input_dir'], cfg['dataset_name'], cfg['csv_dir'])
    MODEL_PATH = os.path.join(cfg['base_dir'], cfg['models_dir'], cfg['dataset_name'], \
                              m_config['model_name'], str(m_config['version']))
    if cfg['feature_extractor']['input_data_type'] == 'train':
        if m_config['model_name'] == 'smallmodel':
            INPUT_PATH = os.path.join(cfg['base_dir'], cfg['vector_dir'], cfg['dataset_name'], m_config['input_data_dir'])
        else:
            INPUT_PATH = os.path.join(cfg['base_dir'], cfg['input_dir'], cfg['dataset_name'], cfg['train_dir'])
        OUTPUT_PATH = os.path.join(cfg['base_dir'], cfg['output_dir'], cfg['dataset_name'], m_config['model_name'])
    else:
        if m_config['model_name'] == 'smallmodel':
            INPUT_PATH = os.path.join(cfg['base_dir'], cfg['vector_dir'], cfg['dataset_name'], "test_" + m_config['input_data_dir'])
        else:
            INPUT_PATH = os.path.join(cfg['base_dir'], cfg['input_dir'], cfg['dataset_name'], cfg['test_dir'])
        OUTPUT_PATH = os.path.join(cfg['base_dir'], cfg['output_dir'], cfg['dataset_name'], "test_" + m_config['model_name'])

    log(INPUT_PATH)


    os.makedirs(OUTPUT_PATH, exist_ok=True)

    MODEL_PATH = os.path.join(MODEL_PATH, 'checkpoint_best.pt')

    dfx = pd.read_csv(os.path.join(CSV_PATH, cfg["preprocess"]["preprocessed_df_with_classes"]))
    class_dict = dict([[i, dfx[i].nunique()] for i in dfx.columns.tolist()[1:]])

    if m_config['model_name'] in ['efficientnet', 'nfnets']:
        fashion_dataset = FashionImageDataset(IMAGE_PATH=INPUT_PATH, config=cfg, m_config=m_config)
    else:
        fashion_dataset = FashionImageDataset(IMAGE_PATH=INPUT_PATH, config=cfg, m_config=m_config)

    fashion_dataloader = DataLoader(fashion_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=8)

    model = FashionModel(config=m_config, num_classes=class_dict)
    model.load_state_dict(torch.load(MODEL_PATH)['state_dict'])
    model.to(cfg["device"])
    model.eval()


    with torch.no_grad():
        pred_dict = dict([[key, []] for key in dict([[i, dfx[i].nunique()] for i in dfx.columns.tolist()]).keys()])
        softmax = nn.Softmax(dim=1)
        with torch.no_grad():
            for i, data in tqdm(enumerate(fashion_dataloader), total=len(fashion_dataloader)):
                image_vec = data['image'].to(cfg["device"])
                filenames = data['filename']
                preds, _, _ = model(image_vec)
                pred_dict['id'].extend(data['filename'])
                for k,v in preds.items():
                    pred_dict[k].extend(torch.argmax(softmax(v), dim=1).cpu().detach().numpy())
        pred_df = pd.DataFrame.from_dict(pred_dict)

    log(pred_df.shape)

    for col in dfx.columns.tolist()[1:]:
        pred_df[col] = decode_label(dfx, pred_df, col)

    out = os.path.join(OUTPUT_PATH, m_config["model_name"] + "_" + str(m_config["version"]) + "_predicted.csv")
    pred_df.to_csv(out, index=False)

    log("Prediction file saved to ", out)

    log("########### Prediction Finished ###########")








def logistic_regression(cfg, m_config):

    """
           Load Final vectors, Load Labels (only one class), and train a logistic regression
           and check accuracy.
        
        
    """

    log("########### Prediction Started ###########")

    CSV_PATH = os.path.join(cfg['base_dir'], cfg['input_dir'], cfg['dataset_name'], cfg['csv_dir'])

    MODEL_PATH = os.path.join(cfg['base_dir'], cfg['models_dir'], cfg['dataset_name'], 'logistic')

    INPUT_PATH = os.path.join(cfg['base_dir'], cfg['vector_dir'], cfg['dataset_name'], m_config['model_name'])

    TARGET_COL = cfg["logistic_regression"]["target_col"]


    os.makedirs(MODEL_PATH, exist_ok=True)

    dfx = pd.read_csv(os.path.join(CSV_PATH, cfg["root_df"]))
    class_dict = dict([[i, dfx[i].nunique()] for i in dfx.columns.tolist()[1:]])
    
    dfx_train = pd.read_csv(os.path.join(CSV_PATH, cfg["train_df"]))
    dfx_val = pd.read_csv(os.path.join(CSV_PATH, cfg["val_df"]))

    def create_logistic_regression_dataset(df, path=INPUT_PATH, target_column=None, targets=None):
        X = []
        for i in df.id.tolist():
            X.append(np.load(os.path.join(path, str(i) + '.npy')).astype(np.float32))
        if targets is not None and target_column is not None:
            if target_column in df.columns.tolist():
                y = np.array(df[target_column].tolist()).reshape(-1, 1)
            X = np.vstack(X)
        return X, y

    X_train, y_train = create_logistic_regression_dataset(dfx_train, INPUT_PATH, target_column=TARGET_COL, targets=True)
    X_val, y_val = create_logistic_regression_dataset(dfx_train, INPUT_PATH, target_column=TARGET_COL, targets=True)
    
    
    log(f"Traing Logistic Regression for {TARGET_COL}------------------")

    # # Perform logistic regression
    classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1, n_jobs=-1)
    classifier.fit(X_train, y_train)

    pickle.dump(classifier, open(os.path.join(MODEL_PATH, f'{m_config["model_name"]}_{m_config["version"]}_{TARGET_COL}.pkl'), 'wb'))
    
    # Evaluate using the logistic regression classifier
    predictions = classifier.predict(X_val)
    # log(predictions.shape)
    accuracy = np.mean((y_val.reshape(-1) == predictions.reshape(-1)).astype(np.float)) * 100.
    log(f"Accuracy = {accuracy:.3f}")
    # accuracy = accuracy_score(y_val.reshape(-1), predictions.reshape(-1))
    # log(f"Accuracy = {accuracy:.3f}")

    log("Logistic Regression Finished ------------------------------------------------")




def add_argument():

    parser = argparse.ArgumentParser(description='FASHION')

    parser.add_argument('--cfg',
                        type=str,
                        default="./cfg.json",
                        help='cconfig file')

    parser.add_argument('--stage',
                        type=str,
                        required=True,
                        help='stage')
    parser.add_argument('--model',
                        type=str,
                        default="bigmodel",
                        help='stage')
    args = parser.parse_args()
    return args



def run(cfg='./confing.json', stage='train', model='bigmodel'):
    """
       python  utils.py  run  --stage preprocess  --model bigmodel


    """
    #args = add_argument()

    with open(cfg) as f:
            config = json.load(f)

    if stage == "preprocess":
        generate_dataset(config)

    elif stage == "feature_extract":
        m_config = config["model_params"][model]
        log(m_config)
        feature_extractor(config, m_config)

    elif stage == "inference":
        m_config = config["model_params"][model]
        log(m_config)
        inference(config, m_config)

    elif stage == "logistic":
        m_config = config["model_params"][model]
        log(m_config)
        logistic_regression(config, m_config)



######################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()



