import warnings

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from sklearn import metrics

warnings.filterwarnings("ignore")

from nfnets import pretrained_nfnet

from torch.nn import functional as F


#######################################################################################################
#                                       NFNets Model                                                  #
#######################################################################################################
class FashionModel(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        self.config       = config
        self.in_features  = config['in_features']
        self.inter_feat   = config['inter_feat']
        self.num_classes  = num_classes
        self.dropout_prob = config['dropout']
        self.model_name   = config['model_name']



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
        self.linear2        = nn.Linear(in_features=256, out_features=self.inter_feat, bias=False)

        self.gender         = nn.Linear(self.inter_feat, self.num_classes['gender'])
        self.masterCategory = nn.Linear(self.inter_feat, self.num_classes['masterCategory'])
        self.subCategory    = nn.Linear(self.inter_feat, self.num_classes['subCategory'])
        self.articleType    = nn.Linear(self.inter_feat, self.num_classes['articleType'])
        self.baseColour     = nn.Linear(self.inter_feat, self.num_classes['baseColour'])
        self.season         = nn.Linear(self.inter_feat, self.num_classes['season'])
        self.usage          = nn.Linear(self.inter_feat, self.num_classes['usage'])
       
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
        out                   = {}
        out["gender"]         = self.gender(x)
        out["masterCategory"] = self.masterCategory(x)
        out["subCategory"]    = self.subCategory(x)
        out["articleType"]    = self.articleType(x)
        out["baseColour"]     = self.baseColour(x)
        out["season"]         = self.season(x)
        out["usage"]          = self.usage(x)

        if targets is not None:
            loss = []
            for k,v in out.items():
                loss.append(nn.CrossEntropyLoss()(out[k], targets[k]))
            loss = sum(loss)
            metrics = self.monitor_metrics(out, targets)
            return out, loss, metrics
        return out, None, None
    
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