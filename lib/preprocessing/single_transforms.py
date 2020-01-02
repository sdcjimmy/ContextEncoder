from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import functional as F
import numpy as np
import random

        
class RandomLightVar(object):
    def __call__(self, img):
        return (img+random.random()*64-32).astype('uint8')

class Equalization(object):
    def __call__(self, img):
        return ImageOps.equalize(img)
    
class GammaAdjustment(object):
    def __init__(self, gamma = 1.3):
        self.gamma = gamma
    def __call__(self, img):
        return transforms.functional.adjust_gamma(img = img, gamma = self.gamma)
        
class ContrastAdjustment(object):
    def __init__(self, contrast_factor = 2):
        self.contrast_factor = contrast_factor
    def __call__(self, img):
        return transforms.functional.adjust_contrast(img = img, contrast_factor=self.contrast_factor)
    
def get_transformer():
    transform = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),        
        transforms.Resize((256,400)),
        ContrastAdjustment(2),
        GammaAdjustment(),                
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),                
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),        
        transforms.Resize((256,400)),
        ContrastAdjustment(2),
        GammaAdjustment(),    
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),       
    ])
    }
    return transform
    
def get_transformer_norm():
    transform = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),        
        transforms.Resize((256,400)),
        ContrastAdjustment(2),
        GammaAdjustment(),                
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),                
        transforms.Normalize([0.170,0.170,0.170],[0.276,0.276,0.276])
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),        
        transforms.Resize((256,400)),
        ContrastAdjustment(2),
        GammaAdjustment(),    
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),       
        transforms.Normalize([0.170,0.170,0.170],[0.276,0.276,0.276])
    ])
    }
    return transform


def get_resnet_transformer():
    transform = {
    'train': transforms.Compose([
        #RandomLightVar(),
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),        
        transforms.Resize((256,400)),
        Equalization(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
        transforms.Normalize([0.371,0.371,0.371],[0.400,0.400, 0.400])
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),        
        transforms.Resize((256,400)),
        Equalization(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
        transforms.Normalize([0.371,0.371,0.371],[0.400,0.400, 0.400])
    ])
    }
    return transform


def get_nonorm_transformer():
    return transforms.Compose([transforms.ToPILImage(),
                               transforms.Resize((256,400)),
                            ContrastAdjustment(2),
                            GammaAdjustment(),                
                            transforms.Grayscale(num_output_channels=3),
                            transforms.ToTensor()])
