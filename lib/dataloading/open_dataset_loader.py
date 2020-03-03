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


def get_train_val_indices(img_dir, validation_split=0, validation_id=1, shuffle_dataset=False, random_seed=1, ext =None):
    if ext is not None:
        img_path_list = glob.glob(os.path.join(img_dir, ext))
    else:
        img_names = os.listdir(img_dir)
        img_path_list = [os.path.join(img_dir, file) for file in img_names]
    
    dataset_size = len(img_path_list)
    indices = list(range(dataset_size))
    split_s = int(np.floor(validation_split*(validation_id-1) * dataset_size))
    split_e = int(np.floor(validation_split*validation_id * dataset_size))
    
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    return indices[:split_s]+indices[split_e:], indices[split_s:split_e]

def get_dataset_dir(dataset_location):
    if dataset_location == 0: #server
        dir_path = '../Data/'
    elif dataset_location == 1:
        dir_path = '/media/curt/E634C46C34C440F3/Data/'
    elif dataset_location == 2: #Shuhang lcoation
        dir_path = '/media/sw/3A5C3A755C3A2C4F/Data/'
    elif dataset_location == 3: #Sijia Location
        dir_path = '/media/curt/D0289F0B289EEFA8/Data/'
    
    return dir_path

def get_train_val_loader(dataset, dataset_location, validation_split, validation_id, image_size,
                         random_seed, shuffle_dataset, batch_size, gpu):
    
    dir_path = get_dataset_dir(dataset_location)

    
    transform = get_transformer(image_size, image_size)
    
    if dataset == 'liver_kidney':
        img_dir = dir_path + 'liver_kidney/img/'
        liver_mask_dir = dir_path + 'liver_kidney/mask_liver/'
        kidney_mask_dir = dir_path + 'liver_kidney/mask_kidney/'
        
        # dataset split
        train_indices, val_indices = get_train_val_indices(img_dir=img_dir, validation_split=validation_split, 
                                                           validation_id=validation_id, shuffle_dataset=shuffle_dataset,
                                                           random_seed = random_seed)
        # creating dataloader:
        train_dataset = LiverHRIDatasetMulti_simple(img_dir = img_dir, kidney_mask_dir = kidney_mask_dir, 
                                                          liver_mask_dir = liver_mask_dir, list_id = train_indices, 
                                                          transform = transform['train'], gpu=gpu) 
        val_dataset = LiverHRIDatasetMulti_simple(img_dir = img_dir, kidney_mask_dir = kidney_mask_dir, 
                                                        liver_mask_dir = liver_mask_dir, list_id = val_indices, 
                                                        transform = transform['val'], gpu=gpu)
    elif dataset == 'nerve':
        img_dir = dir_path + 'ultrasound-nerve-segmentation/train_ori/'
        mask_dir = dir_path + 'ultrasound-nerve-segmentation/train_mask/'
        
        # dataset split
        train_indices, val_indices = get_train_val_indices(img_dir=img_dir, validation_split=validation_split, 
                                                           validation_id=validation_id, shuffle_dataset=shuffle_dataset,
                                                           random_seed = random_seed)
        # creating dataloader:
        train_dataset = SingleClassDataset_Gray(img_dir = img_dir, mask_dir = mask_dir, list_id = train_indices, 
                                           transform = transform['train'], gpu=gpu,
                                           ori_ext = '*.tif', mask_ext = '*.tif') 
        val_dataset = SingleClassDataset_Gray(img_dir = img_dir, mask_dir = mask_dir, list_id = val_indices,
                                         transform = transform['val'], gpu=gpu,
                                         ori_ext = '*.tif', mask_ext = '*.tif')
    elif dataset == 'skin':
        img_dir = dir_path + 'ISIC2018/ISIC2018_Task1-2_Training_Input/'
        mask_dir = dir_path + 'ISIC2018/ISIC2018_Task1_Training_GroundTruth/'
        
        # dataset split
        train_indices, val_indices = get_train_val_indices(img_dir=img_dir, validation_split=validation_split, 
                                                           validation_id=validation_id, shuffle_dataset=shuffle_dataset,
                                                           random_seed = random_seed, ext = '*.jpg')

        # creating dataloader:
        train_dataset = SingleClassDataset_RGB(img_dir = img_dir, mask_dir = mask_dir, list_id = train_indices, 
                                               transform = transform['train'],gpu=gpu, 
                                               ori_ext = '*.jpg', mask_ext = '*.png', mask_appendix = '_segmentation') 
        val_dataset = SingleClassDataset_RGB(img_dir = img_dir, mask_dir = mask_dir, list_id = val_indices, 
                                             transform = transform['val'], gpu=gpu,
                                             ori_ext = '*.jpg', mask_ext = '*.png', mask_appendix = '_segmentation')
    elif dataset == 'thyroid':
        img_dir = dir_path + 'thyroid-ultrasound/Image-slc-train/'
        mask_dir = dir_path + 'thyroid-ultrasound/Mask-slc-train/'
        
        # dataset split
        train_indices, val_indices = get_train_val_indices(img_dir=img_dir, validation_split=validation_split, 
                                                           validation_id=validation_id, shuffle_dataset=shuffle_dataset,
                                                           random_seed = random_seed, ext = '*.png')
        # creating dataloader:
        train_dataset = SingleClassDataset_Gray(img_dir = img_dir, mask_dir = mask_dir, list_id = train_indices, 
                                           transform = transform['train'], gpu=gpu,
                                           ori_ext = '*.png', mask_ext = '*.png') 
        val_dataset = SingleClassDataset_Gray(img_dir = img_dir, mask_dir = mask_dir, list_id = val_indices,
                                         transform = transform['val'], gpu=gpu,
                                         ori_ext = '*.png', mask_ext = '*.png')
    elif dataset == 'breast-curt':
        
        #img_dir = dir_path + 'breast-curt/original_preprocessed/'
        #mask_dir = dir_path + 'breast-curt/mask_preprocessed/'
        
        #img_dir = dir_path + 'breast-curt/2_original_preprocessed_XW_EC/'
        #mask_dir = dir_path + 'breast-curt/2_mask_preprocessed_XW_EC/'
        
        #img_dir = dir_path + 'breast-curt/3_original_preprocessed_XW_EC/'
        #mask_dir = dir_path + 'breast-curt/3_mask_preprocessed_XW_EC/'
        
        img_dir = dir_path + 'breast-curt/4_original_preprocessed_XW_EC/'
        mask_dir = dir_path + 'breast-curt/4_mask_preprocessed_XW_EC/'
        
        # dataset split
        train_indices, val_indices = get_train_val_indices(img_dir=img_dir, validation_split=validation_split, 
                                                           validation_id=validation_id, shuffle_dataset=shuffle_dataset,
                                                           random_seed = random_seed, ext = '*.png')
        # creating dataloader:
        train_dataset = SingleClassDataset_Gray(img_dir = img_dir, mask_dir = mask_dir, 
                                                          list_id = train_indices, 
                                                          transform = transform['train'], gpu=gpu,
                                                          ori_ext = '*.png', mask_ext = '*.png') 
        val_dataset = SingleClassDataset_Gray(img_dir = img_dir, mask_dir = mask_dir, 
                                                        list_id = val_indices, 
                                                        transform = transform['val'], gpu=gpu,
                                                        ori_ext = '*.png', mask_ext = '*.png')
        
        
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers = 4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers = 4)       

    return train_loader, val_loader


def get_test_loader(dataset, 
                    dataset_location, 
                    image_size,
                    batch_size,
                    gpu):
    
    dir_path = get_dataset_dir(dataset_location)
    
    
    transform = get_transformer(image_size, image_size)
    
    if dataset == 'liver_kidney':
        img_dir = dir_path + 'liver_kidney/img_test_Xiaohong/'
        liver_mask_dir = dir_path + 'liver_kidney/mask_liver_test_Xiaohong/'
        kidney_mask_dir = dir_path + 'liver_kidney/mask_kidney_test_Xiaohong/'
        
        test_indices, _ = get_train_val_indices(img_dir = img_dir, ext = '*.dcm')
        # creating dataloader:
        test_dataset = LiverHRIDatasetMulti_simple(img_dir = img_dir, kidney_mask_dir = kidney_mask_dir, 
                                                          liver_mask_dir = liver_mask_dir, list_id = test_indices, 
                                                          transform = transform['val'], gpu=gpu,test=True) 

    elif dataset == 'nerve':
        img_dir = dir_path + 'ultrasound-nerve-segmentation/train_ori/'
        mask_dir = dir_path + 'ultrasound-nerve-segmentation/train_mask/'
        
        test_indices, _ = get_train_val_indices(img_dir = img_dir)
        # creating dataloader:
        test_dataset = SingleClassDataset_Gray(img_dir = img_dir, mask_dir = mask_dir, list_id = test_indices, 
                                           transform = transform['val'], gpu=gpu,
                                           ori_ext = '*.tif', mask_ext = '*.tif',test=True) 

    elif dataset == 'skin':
        img_dir = dir_path + 'ISIC2018/ISIC2018_Task1-2_Training_Input/'
        mask_dir = dir_path + 'ISIC2018/ISIC2018_Task1_Training_GroundTruth/'
        
        test_indices, _ = get_train_val_indices(img_dir = img_dir, ext = '*.jpg')
        # creating dataloader:
        test_dataset = SingleClassDataset_RGB(img_dir = img_dir, mask_dir = mask_dir, list_id = test_indices, 
                                               transform = transform['val'],gpu=gpu, 
                                               ori_ext = '*.jpg', mask_ext = '*.png', mask_appendix = '_segmentation') 

    elif dataset == 'thyroid':        
        img_dir = dir_path + 'thyroid-ultrasound/Image-slc-train/'
        mask_dir = dir_path + 'thyroid-ultrasound/Mask-slc-train/'
        
        test_indices, _ = get_train_val_indices(img_dir = img_dir)
        # creating dataloader:
        test_dataset = SingleClassDataset_Gray(img_dir = img_dir, mask_dir = mask_dir, list_id = test_indices, 
                                           transform = transform['val'], gpu=gpu,
                                           ori_ext = '*.png', mask_ext = '*.png',test=True) 
    elif dataset == 'breast-curt':
        
        img_dir_val = dir_path + 'breast-curt/test_preprocessed_XW_EC/original_v1/'
        mask_dir_val = dir_path + 'breast-curt/test_preprocessed_XW_EC/mask_v1/'
        
        test_indices, _ = get_train_val_indices(img_dir = img_dir_val, ext = '*.png')
        
        # creating dataloader:
        test_dataset = SingleClassDataset_Gray(img_dir = img_dir_val, mask_dir = mask_dir_val, 
                                                        list_id = test_indices, 
                                                        transform = transform['val'], gpu=gpu,
                                                        ori_ext = '*.png', mask_ext = '*.png',test=True)
    
    
    elif dataset == 'breast-curt':
        
        img_dir = dir_path + 'breast-curt/4_original_preprocessed_XW_EC/'
        mask_dir = dir_path + 'breast-curt/4_mask_preprocessed_XW_EC/'
        
        train_indices, val_indices = get_train_val_indices(img_dir = img_dir, validation_split = 0.2, 
                                                           shuffle_dataset = True, random_seed = 123)
        
        # creating dataloader:
        train_dataset = SingleClassDataset_Gray(img_dir = img_dir, mask_dir = mask_dir, 
                                                          list_id = train_indices, 
                                                          transform = transform['train'], gpu=gpu,
                                                          ori_ext = '*.png', mask_ext = '*.png') 
        test_dataset = SingleClassDataset_Gray(img_dir = img_dir, mask_dir = mask_dir, 
                                                        list_id = val_indices, 
                                                        transform = transform['val'], gpu=gpu,
                                                        ori_ext = '*.png', mask_ext = '*.png', test=True)
    
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers = 1)
    return test_dataset, test_loader


       
class LiverHRIDatasetMulti_simple(Dataset):
    
    def __init__(self, img_dir, kidney_mask_dir, liver_mask_dir, list_id = None, transform = None, gpu='0', test=False):
    
        self.img_dir = img_dir
        self.kidney_mask_dir = kidney_mask_dir
        self.liver_mask_dir = liver_mask_dir
        self.list_id = list_id
        self.transform = transform
        self.img_paths, self.kidney_mask_paths, self.liver_mask_paths, self.file_names = self._get_img_list()
        self.test = test
        
    def _get_img_list(self):
        #import pdb; pdb.set_trace()
        img_names = os.listdir(self.img_dir)
        img_paths = [os.path.join(self.img_dir, file) for file in img_names]
        #img_ori_paths = glob.glob(os.path.join(self.img_dir, '*.dcm'))
        kidney_mask_paths = glob.glob(os.path.join(self.kidney_mask_dir, '*.png'))
        liver_mask_paths = glob.glob(os.path.join(self.liver_mask_dir, '*.png'))
        
        #linux
        #img_names = set([file.split('/')[-1][:-4] for file in img_ori_paths])
        kidney_mask_names = set([file.split('/')[-1][:-4] for file in kidney_mask_paths])
        liver_mask_names = set([file.split('/')[-1][:-4] for file in liver_mask_paths])
        '''
        #windows
        img_names = set([file.split('\\')[-1][:-4] for file in img_paths])
        kidney_mask_names = set([file.split('\\')[-1][:-4] for file in kidney_mask_paths])
        liver_mask_names = set([file.split('\\')[-1][:-4] for file in liver_mask_paths])
        '''
        
        file_names = np.array(list(kidney_mask_names.intersection(liver_mask_names)))
        file_names.sort()
        #file_names = np.array(list(kidney_mask_names.intersection(liver_mask_names).intersection(img_names)))
        
        if self.list_id:
            file_names = file_names[self.list_id]
        
        
        #img_paths = [os.path.join(self.img_dir, file+'.dcm') for file in file_names]
        img_paths = [os.path.join(self.img_dir, file) if file in img_names else os.path.join(self.img_dir, file + '.dcm') for file in file_names]
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
        '''
        try:
            x1, x0, y1, y0 = self._get_coord(img_ds)
        except Exception as e:
            print(self.img_paths[idx], 'do not have coordinate')
            print(e)
        '''
        
        try:
            # GE data
            x1, x0, y1, y0 = self._get_coord(img_ds)
        except:
            # SSI data
            x1, x0, y1, y0 = 1200, 290, 850, 240
        
        img = img_ds.pixel_array
        if len(img.shape) == 2:
            img = gray2rgb(img)
        img = img[y0:y1, x0:x1, 0]
        
        while True:
            try:
                kidney_mask = imageio.imread(self.kidney_mask_paths[idx])
                liver_mask = imageio.imread(self.liver_mask_paths[idx])
                break
            except:
                print("Errors during reading mask %s !" % self.kidney_mask_paths[idx])
        
        kidney_mask = kidney_mask[y0:y1, x0:x1, 0]
        liver_mask = liver_mask[y0:y1, x0:x1, 0]
                
        if self.transform:
            img, liver_mask, kidney_mask = self.transform(img, liver_mask.astype('uint8'), (kidney_mask).astype('uint8'))

        if self.test:
            return img, np.stack((liver_mask[0,:,:], kidney_mask[0,:,:])), self.img_paths[idx]
        else:
            return img, np.stack((liver_mask[0,:,:], kidney_mask[0,:,:]))

  
    
class SingleClassDataset_DCM(Dataset):
    def __init__(self, img_dir, mask_dir, list_id = None, transform = None, gpu='0', test=False):

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.list_id = list_id
        self.transform = transform
        self.img_paths, self.mask_paths, self.file_names = self._get_img_list()
        self.test = test

    def _get_img_list(self):

        img_names = os.listdir(self.img_dir)
        img_paths = [os.path.join(self.img_dir, file) for file in img_names]

        mask_paths = glob.glob(os.path.join(self.mask_dir, '*.png'))
        
        #linux
        #img_names = set([file.split('/')[-1][:-4] for file in img_ori_paths])
        mask_names = set([file.split('/')[-1][:-4] for file in mask_paths])
        
        file_names = np.array(list(mask_names))
        file_names.sort()
      
        if self.list_id:
            file_names = file_names[self.list_id]
        
        #img_paths = [os.path.join(self.img_dir, file+'.dcm') for file in file_names]
        img_paths = [os.path.join(self.img_dir, file) if file in img_names else os.path.join(self.img_dir, file + '.dcm') for file in file_names]
        mask_paths = [os.path.join(self.mask_dir, file+'.png') for file in file_names]
        
        return img_paths, mask_paths, file_names
                              
                         
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

        
        try:
            # GE data
            x1, x0, y1, y0 = self._get_coord(img_ds)
        except:
            # SSI data
            x1, x0, y1, y0 = 1200, 290, 850, 240
        
        img = img_ds.pixel_array
        if len(img.shape) == 2:
            img = gray2rgb(img)
        img = img[y0:y1, x0:x1, 0]
        
        while True:
            try:
                mask = imageio.imread(self.mask_paths[idx])
                break
            except:
                print("Errors during reading mask %s !" % self.mask_paths[idx])
        mask = mask[y0:y1, x0:x1, 0]
        
        if self.transform:
            img, mask, _ = self.transform(img, mask.astype('uint8'), mask.astype('uint8'))
        if self.test:
            return img, mask, self.img_paths[idx]
        else:
            return img, mask
        

class SingleClassDataset_Gray(Dataset):
    def __init__(self, img_dir, mask_dir, list_id = None, transform = None, 
                 ori_ext = '*.png', mask_ext = '*.png', mask_appendix = '', test=False):
        self.ori_ext = ori_ext
        self.mask_ext = mask_ext
        self.mask_appendix = mask_appendix
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.list_id = list_id
        self.transform = transform
        self.img_paths, self.mask_paths, self.file_names = self._get_img_list()
        self.test = test
        
        #print(self.file_names)
        
    def _get_img_list(self):
        img_paths = glob.glob(os.path.join(self.img_dir, self.ori_ext))
        mask_paths = glob.glob(os.path.join(self.mask_dir, self.mask_ext))
        
        #linux
        img_names = set([file.split('/')[-1][:-(len(self.ori_ext)-1)] for file in img_paths])
        mask_names = set([file.split('/')[-1][:-(len(self.mask_ext)-1+len(self.mask_appendix))] for file in mask_paths])
        
        file_names = np.array(list(img_names.intersection(mask_names)))
        file_names.sort()
        

        if self.list_id:
            file_names = file_names[self.list_id]
        
        img_paths = [os.path.join(self.img_dir, file + self.ori_ext[1:]) for file in file_names]
        mask_paths = [os.path.join(self.mask_dir, file + self.mask_appendix + self.mask_ext[1:]) for file in file_names]
        
        return img_paths, mask_paths, file_names
                              
                         
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        while True:
            try:
                img = imageio.imread(self.img_paths[idx])
                break
            except:
                print("Errors during reading image %s !" % self.img_paths[idx])

           
        if len(img.shape) == 2:
            img = gray2rgb(img)
        
        while True:
            try:
                mask = imageio.imread(self.mask_paths[idx])
                break
            except:
                print("Errors during reading mask %s !" % self.mask_paths[idx])
        if len(mask.shape) == 3:    
            mask = mask[:, :, 0]
        
        if self.transform:
            img, mask, _ = self.transform(img[:,:,0], mask.astype('uint8'), mask.astype('uint8'))
        if self.test:
            return img, mask, self.img_paths[idx]
        else:
            return img, mask

    
    
class SingleClassDataset_RGB(Dataset):
    def __init__(self, img_dir, mask_dir, list_id = None, transform = None, gpu='0',
                 ori_ext = '*.png', mask_ext = '*.png', mask_appendix = '', test=False):
        self.ori_ext = ori_ext
        self.mask_ext = mask_ext
        self.mask_appendix = mask_appendix
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.list_id = list_id
        self.transform = transform
        self.img_paths, self.mask_paths, self.file_names = self._get_img_list()
        self.test = test
        
    def _get_img_list(self):
        
        img_paths = glob.glob(os.path.join(self.img_dir, self.ori_ext))
        mask_paths = glob.glob(os.path.join(self.mask_dir, self.mask_ext))
        
        #linux
        img_names = set([file.split('/')[-1][:-(len(self.ori_ext)-1)] for file in img_paths])
        mask_names = set([file.split('/')[-1][:-(len(self.mask_ext)-1+len(self.mask_appendix))] for file in mask_paths])
        
        file_names = np.array(list(img_names.intersection(mask_names)))
        file_names.sort()
        
        if self.list_id:
            file_names = file_names[self.list_id]
        
        img_paths = [os.path.join(self.img_dir, file + self.ori_ext[1:]) for file in file_names]
        mask_paths = [os.path.join(self.mask_dir, file + self.mask_appendix + self.mask_ext[1:]) for file in file_names]
        
        return img_paths, mask_paths, file_names
                              
                         
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        while True:
            try:
                img = imageio.imread(self.img_paths[idx])
                break
            except:
                print("Errors during reading image %s !" % self.img_paths[idx])

           
        if len(img.shape) == 2:
            img = gray2rgb(img)
        
        while True:
            try:
                mask = imageio.imread(self.mask_paths[idx])
                break
            except:
                print("Errors during reading mask %s !" % self.mask_paths[idx])
        if len(mask.shape) == 3:
            mask = mask[:,:,0]
        
        if self.transform:
            img, mask, _ = self.transform(img, mask.astype('uint8'), mask.astype('uint8'))
        if self.test:
            return img, mask, self.img_paths[idx]
        else:
            return img, mask

