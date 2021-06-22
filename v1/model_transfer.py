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



# =====================================================================
# Dataset                                                        =
# =====================================================================
class FashionImageDataset(Dataset):

    def __init__(self, IMAGE_PATH, DF_PATH, labels_required = True, use_features=True, aug=None, inference=False):
        """
        Args:
            IMAGE_PATH (string): Directory with all the images or vectors
            DF_PATH (string): Path to csv file
            labels_required (bool): target labels required or not
            use_features (bool): set this to false if want to use images as source instead of vectors or use_features
            aug: augumentation
            inference: set it to True when use the model for inference
        """
        self.image_dir       = IMAGE_PATH
        self.df              = pd.read_csv(DF_PATH)
       
        self.labels_required = labels_required
        self.use_features    = use_features
        self.inference       = inference

        if self.use_features:
             self.images = [str(i) + '.npy' for i in self.df.id.tolist()]
        else:
             self.images = [str(i) + '.jpg' for i in self.df.id.tolist()]

        if aug is None:
          self.aug = albumentations.Compose(
              [
                  albumentations.Normalize(
                      mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225],
                      max_pixel_value=255.0,
                      p=1.0,
                  ),
              ],
              p=1.0,
          )
        else:
          self.aug = aug

        if self.labels_required:
            self.genderLabels         = self.df.gender.tolist()
            self.masterCategoryLabels = self.df.masterCategory.tolist()
            self.subCategoryLabels    = self.df.subCategory.tolist()
            self.articleTypeLabels    = self.df.articleType.tolist()
            self.baseColourLabels     = self.df.baseColour.tolist()
            self.seasonLabels         = self.df.season.tolist()
            self.usageLabels          = self.df.usage.tolist()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        filename =self.images[idx]

        if self.use_features:
            img = np.load(os.path.join(self.image_dir, filename))
            img = torch.from_numpy(img)
        else:
            try:
                img = cv2.imread(os.path.join(self.image_dir, filename))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img,(256,256))
                img = torch.from_numpy(self.aug(image=img)['image']).type(torch.FloatTensor)
                img = img.permute(2, 0, 1)
            except:
                print(filename)
                img = torch.rand((3, 256, 256)).type(torch.FloatTensor)


        if self.labels_required:
            genderLabel         = torch.tensor(self.genderLabels[idx])
            masterCategoryLabel = torch.tensor(self.masterCategoryLabels[idx])
            subCategoryLabel    = torch.tensor(self.subCategoryLabels[idx])
            articleTypeLabel    = torch.tensor(self.articleTypeLabels[idx])
            baseColourLabel     = torch.tensor(self.baseColourLabels[idx])
            seasonLabel         = torch.tensor(self.seasonLabels[idx])
            usageLabel          = torch.tensor(self.usageLabels[idx])

            return {'image': img, 'genderLabel': genderLabel, 'masterCategoryLabel': masterCategoryLabel, 'subCategoryLabel': subCategoryLabel, 
                    'articleTypeLabel': articleTypeLabel, 'baseColourLabel': baseColourLabel, 'seasonLabel': seasonLabel, 'usageLabel': usageLabel
            }

        if self.inference:
            return {'image': img }

        return {'image': img, 'filename': filename.split('.')[0]}


# =====================================================================
# Feature Extractor Model                                                        =
# =====================================================================
class FeatureExtractorEfficientNet(nn.Module):
    def __init__(self, efficientnet_model_name='efficientnet-b3'):
        super().__init__()
        self.effnet = EfficientNet.from_pretrained(efficientnet_model_name)
        self.out_feature_size = self.effnet._conv_head.out_channels

    def forward(self, image):
        batch_size, _, _, _ = image.shape

        x = self.effnet.extract_features(image)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        return x


# =====================================================================
# Model    Small                                                    =
# =====================================================================
class FashionModel(tez.Model):
    def __init__(self, num_classes, in_features=1536, intermediate_features=512):
        super().__init__()

        # self.effnet = EfficientNet.from_pretrained("efficientnet-b3")
        # in_features = 1536
        # intermediate_features = 512
        # Layer 1
        self.bn1            = nn.BatchNorm1d(num_features=in_features)
        self.dropout1       = nn.Dropout(0.25)
        self.linear1        = nn.Linear(in_features=in_features, out_features=intermediate_features, bias=False)
        self.relu           = nn.ReLU()
        # Layer 2
        self.bn2            = nn.BatchNorm1d(num_features=intermediate_features)
        self.dropout2       = nn.Dropout(0.5)

        self.gender         = nn.Linear(intermediate_features, num_classes['gender'])
        self.masterCategory = nn.Linear(intermediate_features, num_classes['masterCategory'])
        self.subCategory    = nn.Linear(intermediate_features, num_classes['subCategory'])
        self.articleType    = nn.Linear(intermediate_features, num_classes['articleType'])
        self.baseColour     = nn.Linear(intermediate_features, num_classes['baseColour'])
        self.season         = nn.Linear(intermediate_features, num_classes['season'])
        self.usage          = nn.Linear(intermediate_features, num_classes['usage'])
       
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

    def fetch_optimizer(self):
        opt = torch.optim.Adam(self.parameters(), lr=3e-4)
        return opt

    def fetch_scheduler(self):
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
        )
        return sch

    def forward(self, image, genderLabel=None, masterCategoryLabel=None, subCategoryLabel=None, articleTypeLabel=None, baseColourLabel=None, seasonLabel=None, usageLabel=None):
        batch_size, _ = image.shape

        x = self.linear1(self.dropout1(self.bn1(image)))
        x = self.relu(x)
        x = self.dropout2(self.bn2(x))

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
        batch_size, _ = image.shape

        features = self.linear1(self.dropout1(self.bn1(image)))

        return features
    
    

class SaveEfterEpoch(Callback):
    def __init__(self, model_path, model_name='model', save_interval=5):
        self.save_interval = save_interval
        self.model_path = model_path
        self.model_name = model_name
    
    def on_epoch_end(self, model):
        if model.current_epoch % self.save_interval == 0:
            model.save(os.path.join(self.model_path, f"{self.model_name}_epoch_{model.current_epoch}.pth"))
            print(f"model saved at epoch: {model.current_epoch}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path",                type=str,  default='./')
    parser.add_argument("--image_path",               type=str,  default='./data/fashion-dataset/images')
    parser.add_argument("--intermediate_vector_path", type=str,  default='./data/fashion-dataset/efficienet_vectors')
    parser.add_argument("--final_vector_path",        type=str,  default='./data/fashion-dataset/final_vectors')
    parser.add_argument("--model_name",               type=str,  default='fashion')
    parser.add_argument("--batch_size",               type=int,  default=8)
    parser.add_argument("--epochs",                   type=int,  default=1)
    parser.add_argument("--save_interval",            type=int,  default=5)
    parser.add_argument("--device",                   type=str,  default='cuda')
    parser.add_argument("--fp16",                     type=bool, default=False)
    parser.add_argument("--patience",                 type=int,  default=5)
    parser.add_argument("--stage",                    type=str,  default='')
    args = parser.parse_args()


    # args.stage = list(args.stage)
    print(args.stage)

    BASE_DIR                 = Path(args.base_path)
    IMAGE_PATH               = Path(args.image_path)
    INTERMEDIATE_VECTOR_PATH = Path(args.intermediate_vector_path)
    FINAL_VECTOR_PATH        = Path(args.final_vector_path)

    MODEL_PATH               = BASE_DIR/'models'
    MODEL_NAME               = args.model_name
    TRAIN_BATCH_SIZE         = args.batch_size
    VALID_BATCH_SIZE         = args.batch_size
    EPOCHS                   = args.epochs

    
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(INTERMEDIATE_VECTOR_PATH, exist_ok=True)
    os.makedirs(FINAL_VECTOR_PATH, exist_ok=True)

    dfx = pd.read_csv(BASE_DIR/'df_final.csv')

    # print(os.path.isfile('images/1536.jpg'))

    # print(set([i.split('.')[-1] for i in os.listdir('images')]))

    # for i in dfx.id.tolist():
    #     if not os.path.isfile(IMAGE_PATH/(str(i) + '.jpg')):
    #         print(os.path.join(IMAGE_PATH, str(i) + '.jpg'))
    #         print(i)


    # print(IMAGE_PATH)
    
    ####### Extract from Efficient Net  ##########################################################################################
    if 'extract_efficienet' in args.stage:
        print("Starting Extract Vectors from Big model -------------------------")
        fashion_dataset    = FashionImageDataset(IMAGE_PATH=IMAGE_PATH, DF_PATH=BASE_DIR/'df_final.csv', use_features=False, labels_required=False)
        fashion_dataloader = DataLoader(fashion_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=8)

        model = FeatureExtractorEfficientNet()
        model.to(args.device)
        model.eval()

        for i, data in tqdm(enumerate(fashion_dataloader), total=len(fashion_dataloader)):
            image_vector = model(data['image'].to(args.device)).cpu().detach().numpy()
            for i in range(len(data['filename'])):
                f_name = data['filename'][i]
                np.save(os.path.join(INTERMEDIATE_VECTOR_PATH, f_name), image_vector[i])
                
                
        print("Stage 1 Finished ------------------------------------------------")
        




    ####### Extract from igmodel  ##########################################################################################
    if 'extract_bigmodel1' in args.stage:
            
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
        
        









    ##############################################################################################################    
    if 'train_small' in args.stage:
        print("Starting  Train Small model -------------------------")
        dfx = pd.read_csv(BASE_DIR/'df_final.csv')
        class_dict = dict([[i, dfx[i].nunique()] for i in dfx.columns.tolist()[1:]])
        
        
        path_data = "data/fashion-dataset/labels/"
        train_dataset = FashionImageDataset(IMAGE_PATH=INTERMEDIATE_VECTOR_PATH, DF_PATH= path_data + '/train.csv')
        val_dataset   = FashionImageDataset(IMAGE_PATH=INTERMEDIATE_VECTOR_PATH, DF_PATH= path_data + '/test.csv')


        ###### Model
        model = FashionModel(num_classes=class_dict)
        
        es = EarlyStopping(
            monitor    = "valid_loss",
            model_path = os.path.join(MODEL_PATH, MODEL_NAME + "_best.bin"),
            patience   = args.patience,
            mode       = "min",
        )

        sfe = SaveEfterEpoch(save_interval=args.save_interval, model_path=MODEL_PATH, model_name=args.model_name)

        model.fit(
            train_dataset,
            valid_dataset = val_dataset,
            train_bs      = TRAIN_BATCH_SIZE,
            valid_bs      = VALID_BATCH_SIZE,
            device        = args.device,
            epochs        = EPOCHS,
            callbacks     = [es, sfe],
            fp16          = args.fp16,
        )

        model.save(os.path.join(MODEL_PATH, MODEL_NAME + "_last.bin"))
        print("Stage 2 Finished -------------------------")


    ##################################################################################################################
    if 'extract_small' in args.stage:
        print("Starting Extract Vector from Small model -------------------------")
        fashion_dataset = FashionImageDataset(IMAGE_PATH=INTERMEDIATE_VECTOR_PATH, DF_PATH=BASE_DIR/'df_final.csv', use_features=True, labels_required=False)
        fashion_dataloader = DataLoader(fashion_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=8)

        model = FashionModel(num_classes=class_dict)
        model.load(os.path.join(MODEL_PATH, MODEL_NAME + "_best.bin"))
        model.to(args.device)
        model.eval()

        for i, data in tqdm(enumerate(fashion_dataloader), total=len(fashion_dataloader)):
            image_vector = model.extract_features(data['image'].to(args.device)).cpu().detach().numpy()
            for i in range(len(data['filename'])):
                f_name = data['filename'][i]
                np.save(os.path.join(FINAL_VECTOR_PATH, f_name), image_vector[i])
        print("Stage 3 Finished -------------------------")



    ##############################################################################################################    
    if 'check_vectors' in args.stage:
        """
           Load Final vectors, Load Labels (only one class), and train a logistic regression
           and check accuracy.
        
        
        """
        print("Starting   -------------------------", args.stage)
        dfx = pd.read_csv(BASE_DIR/'df_final.csv')

        train_dataset = FashionImageDataset(IMAGE_PATH=INTERMEDIATE_VECTOR_PATH, DF_PATH=BASE_DIR/'train.csv')
        val_dataset   = FashionImageDataset(IMAGE_PATH=INTERMEDIATE_VECTOR_PATH, DF_PATH=BASE_DIR/'test.csv')


        fashion_dataset    = FashionImageDataset(IMAGE_PATH=INTERMEDIATE_VECTOR_PATH, DF_PATH=BASE_DIR/'df_final.csv', use_features=True, labels_required=False)
        fashion_dataloader = DataLoader(fashion_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=8)
        
        ###### ONLY check one attribute
        class_dict = dict([[i, dfx[i].nunique()] for i in dfx.columns.tolist()[1:]])
        
        train_features, train_labels =  to_numpy( train_dataset  )
        test_features,  test_labels =  to_numpy(  val_dataset  )
        
        


        # Perform logistic regression
        classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
        classifier.fit(train_features, train_labels)
        
        # Evaluate using the logistic regression classifier
        predictions = classifier.predict(test_features)
        accuracy = np.mean((test_labels == predictions).astype(np.float)) * 100.
        print(f"Accuracy = {accuracy:.3f}")



 



