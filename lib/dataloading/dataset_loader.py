import numpy as np
import torch
import os
import pandas as pd
import pydicom
import random
import pickle

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage.color import rgb2gray, gray2rgb
from .mapping_dict import *
from sklearn.preprocessing import MultiLabelBinarizer

class SSIDataset(Dataset):    
    
    def __init__(self, img_file = '/home/jimmy/Data/SSI/ssi.csv', shuffle = True, list_id = None, transform = None, inpaint = True, rand = None):   
        
        self.df = pd.read_csv(img_file)        
        self.indices = np.arange(self.df.shape[0])
        self.transform = transform
        self.inpaint = inpaint
<<<<<<< HEAD
        self.probe = self.df.Probe.map(probe_dict)
        self.study = pd.get_dummies(self.df.Study.map(study_dict))
=======
        self.probe = self.df.Probe.map(probe_dict)        
        self.study_binarize = self._binarize_study(self.df.Study.map(study_dict))
>>>>>>> 73baf5d2b1245eb86fffc05e14675e3857981e39
        self.rand = rand

        if shuffle == True:            
            np.random.shuffle(self.indices)                        
        
        if list_id != None:
            self.indices = self.indices[list_id]                        
    
    def __len__(self):
        return len(self.indices)
    
    def _get_coord(self, ds):
        coord = ds.SequenceOfUltrasoundRegions[0]
        return coord.RegionLocationMaxX1, coord.RegionLocationMinX0, coord.RegionLocationMaxY1, coord.RegionLocationMinY0
    
    def _inpaint(self, img, is_tensor = True):
        if is_tensor:
            w,h = img.shape[1], img.shape[2]
            label = img[:, w//4:(w*3)//4, h//4:(h*3)//4].clone()
            img[0,w//4:(w*3)//4, h//4:(h*3)//4] = img[0,:, :].min()            
            img[1,w//4:(w*3)//4, h//4:(h*3)//4] = img[1,:, :].min()             
            img[2,w//4:(w*3)//4, h//4:(h*3)//4] = img[2,:, :].min()
            
            if self.rand == 'gaussian':
                img[0,w//4:(w*3)//4, h//4:(h*3)//4] += torch.randn_like(img[0,w//4:(w*3)//4, h//4:(h*3)//4])
                img[1,w//4:(w*3)//4, h//4:(h*3)//4] += torch.randn_like(img[1,w//4:(w*3)//4, h//4:(h*3)//4])
                img[2,w//4:(w*3)//4, h//4:(h*3)//4] += torch.randn_like(img[2,w//4:(w*3)//4, h//4:(h*3)//4])
            elif self.rand == 'uniform':
                img[0,w//4:(w*3)//4, h//4:(h*3)//4] += torch.rand_like(img[0,w//4:(w*3)//4, h//4:(h*3)//4]) *2 - 1
                img[1,w//4:(w*3)//4, h//4:(h*3)//4] += torch.rand_like(img[1,w//4:(w*3)//4, h//4:(h*3)//4]) *2 - 1
                img[2,w//4:(w*3)//4, h//4:(h*3)//4] += torch.rand_like(img[2,w//4:(w*3)//4, h//4:(h*3)//4]) *2 - 1
            return img, label
        else:
            w, h = img.shape[0], img.shape[1]
            img[w//4:(w*3)//4, h//4:(h*3)//4, 0] = img[:, :, 0].min()
            img[w//4:(w*3)//4, h//4:(h*3)//4, 1] = img[:, :, 1].min()
            img[w//4:(w*3)//4, h//4:(h*3)//4, 2] = img[:, :, 2].min()
        return img, label
    
    def _binarize_study(self, study):        
        mlb = MultiLabelBinarizer()
        return mlb.fit_transform(study)
    
    
    def _get_labels(self, idx):
        # probe
        probe = torch.tensor([self.probe[idx]], dtype= torch.uint8)
        study = torch.tensor(self.study_binarize[idx], dtype= torch.uint8)
        return torch.cat([probe, study])
    
    def __getitem__(self, idx):    
        ds = pydicom.dcmread(self.df.iloc[idx].Image_Path)
        try:
            img = ds.pixel_array
        except Exception as e:
            print(self.files[idx], 'do not have pixel array')
            return None
        
        assert len(img.shape) == 2 or len(img.shape) == 3, 'This is not an 2 or 3-dimension static image'
        if len(img.shape) == 2:            
            img = gray2rgb(img)
        
        try:        
            x1, x0, y1, y0 = self._get_coord(ds)
            img = img[y0:y1, x0:x1,:]
            
        except Exception as e:
            print(self.files[idx], 'do not have coordinate')
            print(e)
            
                  
        if self.transform is not None:
            img = self.transform(img)
        
        if self.inpaint:
            labels = self._get_labels(idx)
            crop_img, center = self._inpaint(img)            
            return crop_img, center, labels
        else:
            return img
