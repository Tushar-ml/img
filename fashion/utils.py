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

from sklearn.model_selection import train_test_split

import warnings
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
        print(f"column: {column} nor present in dataframe")
    le = preprocessing.LabelEncoder()
    le.fit(df[column].tolist())
    # print("Number of classes: ", len(le.classes_))
    return le.transform(dfx[column].tolist())

def decode_label(df, dfx, column):
    if column not in df.columns:
        print(f"column: {column} nor present in dataframe")
    le = preprocessing.LabelEncoder()
    le.fit(df[column].tolist())
    # print("Number of classes: ", len(le.classes_))
    return list(le.inverse_transform(dfx[column].tolist()))


def generate_dataset(config):
    
    config_preprocess = config['preprocess']

    IMAGE_PATH = os.path.join(config['base_dir'], config['input_dir'], config['dataset_name'], config['train_dir'])

    CSV_PATH = os.path.join(config['base_dir'], config['input_dir'], config['dataset_name'], config['csv_dir'])

    os.makedirs(CSV_PATH, exist_ok=True)
    
    dfx = pd.read_csv(os.path.join(CSV_PATH, config_preprocess['base_df']), error_bad_lines=False)
    print("Dataset Size: ", dfx.shape)
    dfx = dfx.dropna()
    dfx = dfx[['id', 'gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage']]

    file_not_found = []
    for i in dfx.id.tolist():
        flag = False
        for ext in config_preprocess['image_exts']:
            if os.path.isfile(os.path.join(IMAGE_PATH, str(i) + f".{ext}")):
                flag = True
                break
        if not flag:
            file_not_found.append(i)
    print(f"Total {len(file_not_found)} images didn't found")

    
    dfx = dfx[~dfx['id'].isin(file_not_found)]
    dfx.to_csv(os.path.join(CSV_PATH, config_preprocess['base_df'].split('.')[0] + '_df.csv'), index=False)
    articleTypes = []
    for k, v in dict(dfx.articleType.value_counts()).items():
        if v <= 10:
            articleTypes.append(k)
    dfx = dfx[~dfx['articleType'].isin(articleTypes)]

    # dfx.to_csv(config_preprocess['preprocessed_df_with_classes'], index=False)
    dfx.to_csv(os.path.join(CSV_PATH, config_preprocess['preprocessed_df_with_classes']), index=False)

    for col in dfx.columns.tolist()[1:]:
        dfx[col] = encode_label(dfx, dfx, col)
    
    print("Final Dataset Size: ", dfx.shape)
            
    # dfx.to_csv(config_preprocess['preprocessed_df'], index=False)
    dfx.to_csv(os.path.join(CSV_PATH, config_preprocess['preprocessed_df']), index=False)
    train_dfx, val_dfx = train_test_split(dfx, test_size=config_preprocess["val_set_size"], random_state=42)

    # train_dfx.to_csv(config_preprocess['train_df'], index=False)
    # val_dfx.to_csv(config_preprocess['val_df'], index=False)
    train_dfx.to_csv(os.path.join(CSV_PATH, config_preprocess['train_df']), index=False)
    val_dfx.to_csv(os.path.join(CSV_PATH, config_preprocess['val_df']), index=False)
    print("Stage: df_preprocess Finished ------------------------------------------------")


def feature_extractor(config, model_config):

    CSV_PATH = os.path.join(config['base_dir'], config['input_dir'], config['dataset_name'], config['csv_dir'])
    MODEL_PATH = os.path.join(config['base_dir'], config['models_dir'], config['dataset_name'], \
        model_config['model_name'], str(model_config['version']))
    if config['feature_extractor']['input_data_type'] == 'train':
        if model_config['model_name'] == 'smallmodel':
            INPUT_PATH = os.path.join(config['base_dir'], config['vector_dir'], config['dataset_name'], model_config['input_data_dir'])
        else:
            INPUT_PATH = os.path.join(config['base_dir'], config['input_dir'], config['dataset_name'], config['train_dir'])
        OUTPUT_PATH = os.path.join(config['base_dir'], config['vector_dir'], config['dataset_name'], model_config['model_name'])
    else:
        if model_config['model_name'] == 'smallmodel':
            INPUT_PATH = os.path.join(config['base_dir'], config['vector_dir'], config['dataset_name'], "test_" + model_config['input_data_dir'])
        else:
            INPUT_PATH = os.path.join(config['base_dir'], config['input_dir'], config['dataset_name'], config['test_dir'])
        OUTPUT_PATH = os.path.join(config['base_dir'], config['vector_dir'], config['dataset_name'], "test_" + model_config['model_name'])

    print(INPUT_PATH)
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    print("########### Feature Extraction Started ###########")

    if model_config['model_name'] == 'bigmodel':

        MODEL_PATH = model_config["model_path"]

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

        dfx = pd.read_csv(os.path.join(CSV_PATH, config['root_df']))
        
        model = torch.jit.load(MODEL_PATH).to(config["device"]).eval()
        
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
        
        
        fashion_dataset    = FashionImageDataset(IMAGE_PATH=INPUT_PATH, config=config, m_config=m_config, aug = preprocess)
        fashion_dataloader = DataLoader(fashion_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=8)
        
        
        for i, data in tqdm(enumerate(fashion_dataloader), total=len(fashion_dataloader)):
                image_vector = model.encode_image(data['image'].to(config["device"])).cpu().detach().numpy()
                for i in range(len(data['filename'])):
                    f_name = data['filename'][i]
                    np.save(os.path.join(OUTPUT_PATH, f_name), image_vector[i])
    else:

        MODEL_PATH = os.path.join(MODEL_PATH, 'checkpoint_best.pt')

        dfx = pd.read_csv(os.path.join(CSV_PATH, config['root_df']))
        class_dict = dict([[i, dfx[i].nunique()] for i in dfx.columns.tolist()[1:]])

        if m_config['model_name'] in ['efficientnet', 'nfnets']:
            fashion_dataset = FashionImageDataset(IMAGE_PATH=INPUT_PATH, config=config, m_config=m_config)
        else:
            fashion_dataset = FashionImageDataset(IMAGE_PATH=INPUT_PATH, config=config, m_config=m_config)

        fashion_dataloader = DataLoader(fashion_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=8)

        model = FashionModel(config=m_config, num_classes=class_dict)
        model.load_state_dict(torch.load(MODEL_PATH)['state_dict'])
        model.to(config["device"])
        model.eval()

        for i, data in tqdm(enumerate(fashion_dataloader), total=len(fashion_dataloader)):
            image_vector = model.extract_features(data['image'].to(config["device"])).cpu().detach().numpy()
            for i in range(len(data['filename'])):
                f_name = data['filename'][i]
                np.save(os.path.join(OUTPUT_PATH, f_name), image_vector[i])
    print("########### Feature Extraction Finished ###########")



def inference(config, model_config):

    print("########### Prediction Started ###########")

    CSV_PATH = os.path.join(config['base_dir'], config['input_dir'], config['dataset_name'], config['csv_dir'])
    MODEL_PATH = os.path.join(config['base_dir'], config['models_dir'], config['dataset_name'], \
        model_config['model_name'], str(model_config['version']))
    if config['feature_extractor']['input_data_type'] == 'train':
        if model_config['model_name'] == 'smallmodel':
            INPUT_PATH = os.path.join(config['base_dir'], config['vector_dir'], config['dataset_name'], model_config['input_data_dir'])
        else:
            INPUT_PATH = os.path.join(config['base_dir'], config['input_dir'], config['dataset_name'], config['train_dir'])
        OUTPUT_PATH = os.path.join(config['base_dir'], config['output_dir'], config['dataset_name'], model_config['model_name'])
    else:
        if model_config['model_name'] == 'smallmodel':
            INPUT_PATH = os.path.join(config['base_dir'], config['vector_dir'], config['dataset_name'], "test_" + model_config['input_data_dir'])
        else:
            INPUT_PATH = os.path.join(config['base_dir'], config['input_dir'], config['dataset_name'], config['test_dir'])
        OUTPUT_PATH = os.path.join(config['base_dir'], config['output_dir'], config['dataset_name'], "test_" + model_config['model_name'])

    print(INPUT_PATH)


    os.makedirs(OUTPUT_PATH, exist_ok=True)

    MODEL_PATH = os.path.join(MODEL_PATH, 'checkpoint_best.pt')

    dfx = pd.read_csv(os.path.join(CSV_PATH, config["preprocess"]["preprocessed_df_with_classes"]))
    class_dict = dict([[i, dfx[i].nunique()] for i in dfx.columns.tolist()[1:]])

    if m_config['model_name'] in ['efficientnet', 'nfnets']:
        fashion_dataset = FashionImageDataset(IMAGE_PATH=INPUT_PATH, config=config, m_config=m_config)
    else:
        fashion_dataset = FashionImageDataset(IMAGE_PATH=INPUT_PATH, config=config, m_config=m_config)

    fashion_dataloader = DataLoader(fashion_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=8)

    model = FashionModel(config=m_config, num_classes=class_dict)
    model.load_state_dict(torch.load(MODEL_PATH)['state_dict'])
    model.to(config["device"])
    model.eval()


    with torch.no_grad():
        pred_dict = dict([[key, []] for key in dict([[i, dfx[i].nunique()] for i in dfx.columns.tolist()]).keys()])
        softmax = nn.Softmax(dim=1)
        with torch.no_grad():
            for i, data in tqdm(enumerate(fashion_dataloader), total=len(fashion_dataloader)):
                image_vec = data['image'].to(config["device"])
                filenames = data['filename']
                preds, _, _ = model(image_vec)
                pred_dict['id'].extend(data['filename'])
                for k,v in preds.items():
                    pred_dict[k].extend(torch.argmax(softmax(v), dim=1).cpu().detach().numpy())
        pred_df = pd.DataFrame.from_dict(pred_dict)

    print(pred_df.shape)

    for col in dfx.columns.tolist()[1:]:
        pred_df[col] = decode_label(dfx, pred_df, col)

    out = os.path.join(OUTPUT_PATH, model_config["model_name"] + "_" + str(model_config["version"]) + "_predicted.csv")
    pred_df.to_csv(out, index=False)

    print("Prediction file saved to ", out)

    print("########### Prediction Finished ###########")


def logistic_regression(config, model_config):

    """
           Load Final vectors, Load Labels (only one class), and train a logistic regression
           and check accuracy.
        
        
    """

    print("########### Prediction Started ###########")

    CSV_PATH = os.path.join(config['base_dir'], config['input_dir'], config['dataset_name'], config['csv_dir'])

    MODEL_PATH = os.path.join(config['base_dir'], config['models_dir'], config['dataset_name'], 'logistic')

    INPUT_PATH = os.path.join(config['base_dir'], config['vector_dir'], config['dataset_name'], model_config['model_name'])

    TARGET_COL = config["logistic_regression"]["target_col"]


    os.makedirs(MODEL_PATH, exist_ok=True)

    dfx = pd.read_csv(os.path.join(CSV_PATH, config["root_df"]))
    class_dict = dict([[i, dfx[i].nunique()] for i in dfx.columns.tolist()[1:]])
    
    dfx_train = pd.read_csv(os.path.join(CSV_PATH, config["train_df"]))
    dfx_val = pd.read_csv(os.path.join(CSV_PATH, config["val_df"]))

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
    
    
    print(f"Traing Logistic Regression for {TARGET_COL}------------------")

    # # Perform logistic regression
    classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1, n_jobs=-1)
    classifier.fit(X_train, y_train)

    pickle.dump(classifier, open(os.path.join(MODEL_PATH, f'{m_config["model_name"]}_{m_config["version"]}_{TARGET_COL}.pkl'), 'wb'))
    
    # Evaluate using the logistic regression classifier
    predictions = classifier.predict(X_val)
    # print(predictions.shape)
    accuracy = np.mean((y_val.reshape(-1) == predictions.reshape(-1)).astype(np.float)) * 100.
    print(f"Accuracy = {accuracy:.3f}")
    # accuracy = accuracy_score(y_val.reshape(-1), predictions.reshape(-1))
    # print(f"Accuracy = {accuracy:.3f}")

    print("Logistic Regression Finished ------------------------------------------------")




def add_argument():

    parser = argparse.ArgumentParser(description='FASHION')

    parser.add_argument('--config',
                        type=str,
                        default="./config.json",
                        help='config file')

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

if __name__ == "__main__":

    args = add_argument()

    with open(args.config) as f:
            config = json.load(f)

    if args.stage == "preprocessing":
        generate_dataset(config)

    elif args.stage == "feature_extractor":
        m_config = config["model_params"][args.model]
        print(m_config)
        feature_extractor(config, m_config)

    elif args.stage == "inference":
        m_config = config["model_params"][args.model]
        print(m_config)
        inference(config, m_config)

    elif args.stage == "logistic":
        m_config = config["model_params"][args.model]
        print(m_config)
        logistic_regression(config, m_config)


