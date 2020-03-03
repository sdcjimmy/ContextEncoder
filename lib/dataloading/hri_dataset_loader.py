import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import glob
import pandas as pd
import pydicom
import imageio

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage.color import rgb2gray, gray2rgb
from skimage.filters import gaussian
from skimage.measure import label
from scipy.ndimage.morphology import binary_fill_holes
from PIL import Image, ImageFile
from torchvision.utils import save_image
ImageFile.LOAD_TRUNCATED_IMAGES = True

              
class LiverHRIDatasetMulti_simple(Dataset):
    
    def __init__(self, img_dir, kidney_mask_dir, liver_mask_dir, list_id = None, transform = None, gpu='0'):
    
        self.img_dir = img_dir
        self.kidney_mask_dir = kidney_mask_dir
        self.liver_mask_dir = liver_mask_dir
        self.list_id = list_id
        self.transform = transform        
        
        
        self.img_paths, self.kidney_mask_paths, self.liver_mask_paths, self.file_names = self._get_img_list()
        
    def _get_img_list(self):
        img_paths = glob.glob(os.path.join(self.img_dir, '*.dcm'))
        kidney_mask_paths = glob.glob(os.path.join(self.kidney_mask_dir, '*.png'))
        liver_mask_paths = glob.glob(os.path.join(self.liver_mask_dir, '*.png'))
        
        img_names = set([file.split('/')[-1][:-4] for file in img_paths])
        kidney_mask_names = set([file.split('/')[-1][:-4] for file in kidney_mask_paths])
        liver_mask_names = set([file.split('/')[-1][:-4] for file in liver_mask_paths])
  
        file_names = np.array(list(img_names.intersection(kidney_mask_names).intersection(liver_mask_names)))
        
        if self.list_id:
            file_names = file_names[self.list_id]
        
        
        img_paths = [os.path.join(self.img_dir, file+'.dcm') for file in file_names]
        kidney_mask_paths = [os.path.join(self.kidney_mask_dir, file+'.png') for file in file_names]
        liver_mask_paths = [os.path.join(self.liver_mask_dir, file+'.png') for file in file_names]
        
        return img_paths, kidney_mask_paths, liver_mask_paths, file_names
                              
                         
    def _get_coord(self, ds):
        coord = ds.SequenceOfUltrasoundRegions[0]
        return coord.RegionLocationMaxX1, coord.RegionLocationMinX0, coord.RegionLocationMaxY1, coord.RegionLocationMinY0
       
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        while True:
            try:
                # Get Image DICOM file
                img_ds = pydicom.dcmread(self.img_paths[idx])
                break
            except:
                print("Errors during reading image %s !" % self.img_paths[idx])
                
        # Get Trimmed Coordinate
        try:
            x1, x0, y1, y0 = self._get_coord(img_ds)            
        except:
            x1, x0, y1, y0 = 1200, 290, 850, 240
        
        img = img_ds.pixel_array

        if len(img.shape) == 2:
            img = gray2rgb(img)
            
        img = img[y0:y1, x0:x1, :]
                            
        
        while True:
            try:
                kidney_mask = imageio.imread(self.kidney_mask_paths[idx])
                liver_mask = imageio.imread(self.liver_mask_paths[idx])
                break
            except:
                print("Errors during reading mask %s !" % self.kidney_mask_paths[idx])        
        kidney_mask = kidney_mask[y0:y1, x0:x1, :]
        liver_mask = liver_mask[y0:y1, x0:x1, :]        
        
        if self.transform:
            # the transform should be joint transform from joint_transform.py
            img, liver_mask, kidney_mask = self.transform(img[:,:,0], liver_mask.astype('uint8'), (kidney_mask).astype('uint8'))
                             
            return img, np.stack((liver_mask[0,:,:], kidney_mask[0,:,:])) 
        
        else:
            return img, np.stack((liver_mask[:,:,0], kidney_mask[:,:,0])) 
