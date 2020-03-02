import torch.nn.functional as F
from torch.nn.functional import interpolate
import torch.nn as nn
import torchvision.models as models
import torch

from .utils import init_weights


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, padding = 1, output_padding = 1):
        super().__init__()

        self.block = nn.Sequential(
            ConvBatchRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3, stride=2, padding= padding, output_padding= output_padding),            
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class ConvBatchRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = nn.Conv2d(in_, out, 3, padding=1)
        self.batchnorm = nn.BatchNorm2d(out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        return x



class VGGCEUNet(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=False, self_trained = '', freeze = False, activation = 'none', down_sample = 1):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with VGG16
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes
        self.down_sample = down_sample
        self.activation = activation
        self.encoder = models.vgg16_bn(pretrained=pretrained).features
                                      
        self.relu = nn.ReLU(inplace=True)                
        self.pool = nn.MaxPool2d(2, 2)
                
        self.conv1 = nn.Sequential(self.encoder[0],
                                   self.encoder[1],
                                   self.relu,
                                   self.encoder[3],
                                   self.encoder[4],
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[7],
                                   self.encoder[8],
                                   self.relu,
                                   self.encoder[10],
                                   self.encoder[11],
                                   self.relu)

        self.conv3 = nn.Sequential(self.encoder[14],
                                   self.encoder[15],
                                   self.relu,
                                   self.encoder[17],
                                   self.encoder[18],
                                   self.relu,
                                   self.encoder[20],
                                   self.encoder[21],
                                   self.relu)

        self.conv4 = nn.Sequential(self.encoder[24],
                                   self.encoder[25],
                                   self.relu,
                                   self.encoder[27],
                                   self.encoder[28],
                                   self.relu,
                                   self.encoder[30],
                                   self.encoder[31],
                                   self.relu)

        self.conv5 = nn.Sequential(self.encoder[34],
                                   self.encoder[35],
                                   self.relu,
                                   self.encoder[37],
                                   self.encoder[38],
                                   self.relu,
                                   self.encoder[40],
                                   self.encoder[41],
                                   self.relu)
    
        self.center = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8, padding = (1,0), output_padding = (1,0))

        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2 * 2, num_filters)
        self.dec1 = ConvBatchRelu(64 + num_filters, num_filters)
        
        if self_trained != '':            
            print(f"Load weights from {self_trained}")
            pretrained_dict = torch.load(self_trained)            
            model_dict = self.state_dict()            
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)             
            
            # 3. load the new state dict0            
            self.load_state_dict(model_dict)                    
        
        if freeze:
            for param in self.encoder.parameters():
                param.require_grad = False
        
        
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)
        
        if self.activation == 'sigmoid':
            self.final_act = nn.Sigmoid()
        elif self.activation == 'tanh':
            self.final_act = nn.Tanh()
        

    def forward(self, x):        
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
      
        x_out = self.final(dec1)
        
        if self.down_sample < 1:
            x_out = interpolate(x_out, scale_factor = self.down_sample)

        if self.activation != 'none':
            x_out = self.final_act(x_out)
            
        return x_out
