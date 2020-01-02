# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F
import torch.nn as nn

from .unet_model import UNet
from .unet_parts import *
from .utils import init_weights


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

class GlobalAveragePooling(nn.Module):
    def forward(self, x):        
        return torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
  

class ViewNet(nn.Module):
    def __init__(self, pretrain_dict = None, freeze = False):
        super(ViewNet, self).__init__()
        pretrain_net = UNet(n_channels = 1, n_classes = 2)
        if pretrain_dict != None:
            print('loading weights...')
            pretrain_net.load_state_dict(pretrain_dict)
        
        modules = list(pretrain_net.children())[:5]
        self.encoder = nn.Sequential(*modules)
        if freeze:
            print('freezing the encoder')
            for param in self.encoder.parameters():
                param.requires_grad = False        
        
        self.classifier = nn.Sequential(                
            nn.Dropout(),
            nn.Conv2d(512,1000, kernel_size = (1,1), stride = (1,1)),                
            nn.ReLU(inplace=True),            
            nn.Dropout(),
            nn.Conv2d(1000,256, kernel_size = (1,1), stride = (1,1)),                
            nn.ReLU(inplace=True),            
            GlobalAveragePooling(),            
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),                        
            nn.Linear(64, 1),
            nn.Sigmoid()
            )
        
    def forward(self, x):
        x = self.encoder(x)
        #x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        x = self.classifier(x)        
        return x
        