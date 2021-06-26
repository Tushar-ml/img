import os
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

warnings.filterwarnings("ignore")

from PIL import Image


# =====================================================================
# Dataset                                                        =
# =====================================================================
class FashionImageDataset(Dataset):

    def __init__(self, IMAGE_PATH, DF_PATH, config, labels_required = True, use_features=False, aug=None, inference=False):
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
        if config['sample']:
            self.df              = pd.read_csv(DF_PATH, nrows=config['sample'])
        else:
            self.df              = pd.read_csv(DF_PATH)
       
        self.labels_required = labels_required
        self.use_features    = use_features
        self.inference       = inference

        if self.use_features:
             self.images = [str(i) + '.npy' for i in self.df.id.tolist()]
        else:
             self.images = [str(i) + '.jpg' for i in self.df.id.tolist()]

        if aug is None:
         
          self.aug = Compose([
                Resize((384, 384), interpolation=Image.BICUBIC),
                CenterCrop((224, 224)),
                ToTensor(),
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
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
            img = np.load(os.path.join(self.image_dir, filename)).astype(np.float32)
            img = torch.from_numpy(img)
        else:
            img = Image.open(os.path.join(self.image_dir, filename)).convert('RGB')
            # 
            # print(image.shape)
            # img = np.array(Image.open(os.path.join(self.image_dir, filename)).convert('RGB'))
            img = self.aug(img).type(torch.FloatTensor)
            # img = cv2.resize(img,(256,256))
            # img = torch.from_numpy(self.aug(image=img)['image']).type(torch.FloatTensor)
            # img = img.permute(2, 0, 1)


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





if __name__ == "__main__":
   import fire
   image_path = 'data/fashion/train'
   csv_path = 'data/fashion/csv/train.csv'
   config = {
       'sample':7000
   }
   fire.Fire(FashionImageDataset(image_path,csv_path,config)[5244])