# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F
import torch.nn as nn

from .unet_parts import *
from .utils import init_weights

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, pretrain_dict = None, freeze = False, pretrain_model = 'localizer'):
        super(UNet, self).__init__()
        
        '''
        if pretrain_model == 'localizer':
            self.pretrain_model = UNetLocalizer(n_channels, 2)
        elif pretrain_model == 'autoencoder':
            self.pretrain_model = AutoEncoder(n_channels)
        
        if pretrain_dict:
            print("Loading pretrain model ...")
            self.pretrain_dict = pretrain_dict                
            self.pretrain_model.load_state_dict(pretrain_dict)
            if freeze:
                for param in self.pretrain_model.parameters():
                    param.require_grad = False
        else:
            # initialise weights
            print("Initializing with HE weights")
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    init_weights(m, init_type = 'kaiming')
                elif isinstance(m, nn.BatchNorm2d):
                    init_weights(m, init_type = 'kaiming')
        

        self.inc = self.pretrain_model.inc
        self.down1 = self.pretrain_model.down1
        self.down2 = self.pretrain_model.down2
        self.down3 = self.pretrain_model.down3
        self.down4 = self.pretrain_model.down4
        '''
        
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return F.sigmoid(x)
    
    
class UNetLocalizer(nn.Module):
    def __init__(self, n_channels, n_classes, gap = True):
        super(UNetLocalizer, self).__init__()
        self.gap = gap
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.conv = oneConv(512)
        self.out = outlinear(40 * 30, n_classes)
        
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type = 'kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type = 'kaiming')

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.gap:
            x = x5.mean(1)
        else:
            x = self.conv(x5)
        x = x5.mean(1)
        x = x.view(x.size()[0], -1)
        out = self.out(x)
        return out

class UNetLight(nn.Module):
    def __init__(self, n_channels, n_classes, pretrain_dict = None, freeze = False):
        super(UNetLight, self).__init__()
        self.localizer = UNetLocalizerLight(n_channels, 2)
        
        if pretrain_dict:
            print("Loading pretrain model ...")
            self.pretrain_dict = pretrain_dict                
            self.localizer.load_state_dict(pretrain_dict)
            if freeze:
                for param in self.localizer.parameters():
                    param.require_grad = False
        else:
            # initialise weights
            print("Initializing with HE weights")
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    init_weights(m, init_type = 'kaiming')
                elif isinstance(m, nn.BatchNorm2d):
                    init_weights(m, init_type = 'kaiming')
        

        self.inc = self.localizer.inc
        self.down1 = self.localizer.down1
        self.down2 = self.localizer.down2
        self.down3 = self.localizer.down3
        self.down4 = self.localizer.down4

        self.up1 = up(512, 128)
        self.up2 = up(256, 64)
        self.up3 = up(128, 32)
        self.up4 = up(64, 32)
        self.outc = outconv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return F.sigmoid(x)

class UNetLocalizerLight(nn.Module):
    def __init__(self, n_channels, n_classes, gap = True):
        super(UNetLocalizerLight, self).__init__()
        self.gap = gap
        self.inc = inconv(n_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 256)
        self.conv = oneConv(256)
        self.out = outlinear(40 * 30, n_classes)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type = 'kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type = 'kaiming')

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.gap:
            x = x5.mean(1)
        else:
            x = self.conv(x5)
        x = x.view(x.size()[0], -1)
        out = self.out(x)
        return out
        

class AutoEncoder(nn.Module):
    def __init__(self, n_channels):
        super(AutoEncoder, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = auto_up(512, 256)
        self.up2 = auto_up(256, 128)
        self.up3 = auto_up(128, 64)
        self.up4 = auto_up(64, 32)
        self.outc = outconv(32, n_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.outc(x)
        
        return F.tanh(x)
    
class VGG11UNet(nn.Module):
    def __init__(self, num_filters=32, pretrained=False, freeze = False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with VGG11
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = models.vgg11(pretrained=pretrained).features
        if freeze:
            for param in self.encoder.parameters():
                param.require_grad = False

        self.relu = self.encoder[1]
        self.conv1 = self.encoder[0]
        self.conv2 = self.encoder[3]
        self.conv3s = self.encoder[6]
        self.conv3 = self.encoder[8]
        self.conv4s = self.encoder[11]
        self.conv4 = self.encoder[13]
        self.conv5s = self.encoder[16]
        self.conv5 = self.encoder[18]

        self.center = DecoderBlock(num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8)
        self.dec5 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4)
        self.dec3 = DecoderBlock(num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(num_filters * (4 + 2), num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters)

        self.final = nn.Conv2d(num_filters, 1, kernel_size=1)

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(self.pool(conv1)))
        conv3s = self.relu(self.conv3s(self.pool(conv2)))
        conv3 = self.relu(self.conv3(conv3s))
        conv4s = self.relu(self.conv4s(self.pool(conv3)))
        conv4 = self.relu(self.conv4(conv4s))
        conv5s = self.relu(self.conv5s(self.pool(conv4)))
        conv5 = self.relu(self.conv5(conv5s))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return F.sigmoid(self.final(dec1))
    
    

class VGG16UNet(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=False, freeze = False, is_deconv=False):
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

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = models.vgg16(pretrained=pretrained).features
        if freeze:
            for param in self.encoder.parameters():
                param.require_grad = False

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder[0],
                                   self.relu,
                                   self.encoder[2],
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[5],
                                   self.relu,
                                   self.encoder[7],
                                   self.relu)

        self.conv3 = nn.Sequential(self.encoder[10],
                                   self.relu,
                                   self.encoder[12],
                                   self.relu,
                                   self.encoder[14],
                                   self.relu)

        self.conv4 = nn.Sequential(self.encoder[17],
                                   self.relu,
                                   self.encoder[19],
                                   self.relu,
                                   self.encoder[21],
                                   self.relu)

        self.conv5 = nn.Sequential(self.encoder[24],
                                   self.relu,
                                   self.encoder[26],
                                   self.relu,
                                   self.encoder[28],
                                   self.relu)

        self.center = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(128 + num_filters * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

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

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec1), dim=1)
        else:
            x_out = self.final(dec1)

        return F.sigmoid(x_out)
