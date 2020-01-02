import sys
import os
import glob
import pandas as pd
import argparse
import configparser
import numpy as np
import torch
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from collections import OrderedDict

from model import *
from lib.preprocessing import *
from lib.dataloading import *
from lib.loss_functions import *
from lib.evaluation import *
from torchvision import transforms
from torchvision.utils import save_image
from torch import optim, nn
#from torch.utils.tensorboard import SummaryWriter

class NetworkPredicter(object):

    def __init__(self, img_dir, 
                 kidney_mask_dir = None, 
                 liver_mask_dir = None, 
                 network = 'unet',                 
                 batch_size = 1,                 
                 weight_path = None,
                 gpu = '0',
                 save = True,
                 output_dir = '../01_Rawdata_for_annotation/Test/predictions/'):
        
        ## Set the default information
        self.info = OrderedDict()
        self.set_info(network, batch_size, weight_path)
        self.device = torch.device("cuda:%s" % gpu if torch.cuda.is_available() else "cpu")
        self.save_imgs = save
        
        ## Dataset
        self.img_dir = img_dir
        self.kidney_mask_dir = kidney_mask_dir
        self.liver_mask_dir = liver_mask_dir 
        if not self.kidney_mask_dir and not self.liver_mask_dir:
            self.has_mask = False
        else:
            self.has_mask = True
                
        self.dataset = self.load_dataset()
        self.file_names = [file.split('.')[0] for file in self.dataset.file_names]
        
        ## Output folder path 
        exp_name = weight_path.split('/')[-2]
        self.output_dir = os.path.join(output_dir, exp_name)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            os.mkdir(os.path.join(self.output_dir, 'liver'))
            os.mkdir(os.path.join(self.output_dir, 'kidney'))            
        self.output_dir_liver = os.path.join(self.output_dir, 'liver')       
        self.output_dir_kidney = os.path.join(self.output_dir, 'kidney')
        
        
        ## initialize results dictionary
        self.results = self.intialize_results_dict()
        
        # Network
        self.network = self.get_network()                        
        self.load_weights(weight_path)
        self.transform = self.get_transform()
        self.test_loader = self.get_data_loader()
        
        print("Network initizliation:")        
        print("Test Number: %s" % (len(self.test_loader) * self.info['batch_size']))
        
    def set_info(self, network, batch_size, weight_path):
        self.info['network'] = network        
        self.info['batch_size'] = batch_size
        self.info['weight_path'] = weight_path
        
    
    def update_info(key, value):
        self.info[key] = value    
    
    def intialize_results_dict(self):
        results = OrderedDict()                
        results['dice'] = []
        results['liver_dice'] = []
        results['kidney_dice'] = []
        return results
    
        
    def load_dataset(self, list_id = None, transform = None):       
        if self.has_mask:
            return LiverHRIDatasetMulti_simple(img_dir = self.img_dir, kidney_mask_dir = self.kidney_mask_dir, liver_mask_dir = self.liver_mask_dir, list_id = list_id, transform = transform) 
        else:
            return LiverHRIDatasetMulti_inference(img_dir = self.img_dir, transform = transform)
                            
    def get_data_loader(self):        
        test_dataset = self.load_dataset(transform = self.transform['val'])        
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.info['batch_size'], num_workers = 4)       
        return test_loader
        
    def get_network(self):
        if self.info['network'] == "unet":
            net = UNet(n_channels = 1, n_classes = 2)
        elif self.info['network'] == "unet-light":
            net = UNetLight(n_channels = 1, n_classes = 1)
        elif self.info['network'] == "vggnet":
            net = VGG16UNet(pretrained = True, freeze = args.freeze)
        elif self.info['network'] == 'res-unet':
            net = ResidualUNet(in_channels = 1, out_channels=2, num_hidden_features = [64,128,256,512,1024], n_resblocks = 1, num_dilated_convs = 0)
        elif self.info['network'] == 'r2u-unet':
            net = R2U_Net(img_ch = 1, output_ch = 2)
        
        return net
    
    def get_transform(self):
        return get_transformer()
    
    
    def predict(self):
        self.network = self.network.to(self.device)                        
        with torch.set_grad_enabled(False):
            self.network.eval()                            
            for i, img in enumerate(self.test_loader):
            # Transfer to GPU
                imgs = img.to(self.device)
                masks_pred = self.network(imgs)
                masks_pred = (masks_pred > 0.5).float()
                
                if self.save_imgs:
                    save_image(masks_pred[0,0,:,:], os.path.join(self.output_dir_liver, self.file_names[i] + '.png'))
                    save_image(masks_pred[0,1,:,:], os.path.join(self.output_dir_kidney, self.file_names[i] + '.png'))                
        
    def evaluate(self):
        assert self.has_mask, 'No ground-truth masks -- please use predict function instead'
            
        self.network = self.network.to(self.device)                
        tot_dice = np.array([0,0])
        with torch.set_grad_enabled(False):
            self.network.eval()                            
            for i, (img, mask) in enumerate(self.test_loader):
            # Transfer to GPU
                imgs, masks = img.to(self.device), mask.to(self.device)

                masks_pred = self.network(imgs)
                masks_pred = (masks_pred > 0.5).float()

                dice_liver, dice_kidney = dice(masks_pred, masks).detach().cpu().numpy()                
                self.results['liver_dice'].append(dice_liver)
                self.results['kidney_dice'].append(dice_kidney)
                self.results['dice'].append(np.mean([dice_liver, dice_kidney]))

                if self.save_imgs:
                    save_image(masks_pred[0,0,:,:], os.path.join(self.output_dir_liver, self.file_names[i] + '.png'))
                    save_image(masks_pred[0,1,:,:], os.path.join(self.output_dir_kidney, self.file_names[i] + '.png'))
                                    
                                   
        
        
        self.results['avg_dice'] = np.mean(self.results['dice'])
        self.results['avg_liver_dice'] = np.mean(self.results['liver_dice'])
        self.results['avg_kidney_dice'] = np.mean(self.results['kidney_dice'])
   

        print("Finish Inference")
        print("Mean Liver Dice on the Test set:", self.results['avg_liver_dice'])
        print("Mean Kidney Dice on the Test set:", self.results['avg_kidney_dice'])
        self.save_results()
  
    def save_results(self):
        config = configparser.ConfigParser()        
        config['INFO'] = self.info
        config['RESULTS'] = {'avg_dice': self.results['avg_dice'],
                            'avg_liver_dice': self.results['avg_liver_dice'],
                            'avg_kidney_dice': self.results['avg_kidney_dice']}
                        
        
        all_dice = pd.DataFrame({'file': self.file_names,
                                'dice': self.results['dice'],
                                'liver_dice': self.results['liver_dice'],
                                'kidney_dice': self.results['kidney_dice']})
        
        with open(os.path.join(self.output_dir, 'exp.ini'), 'w') as configfile:
            config.write(configfile)
            
        all_dice.to_csv(os.path.join(self.output_dir, 'dices.csv'))
        
    def load_weights(self, weight_path):
        self.network.load_state_dict(torch.load(weight_path))
        
        
        
        

            
            
            
            
            
        
        
    