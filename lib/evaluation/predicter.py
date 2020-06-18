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
from lib.preprocessing.joint_transforms import get_transformer_norm
from lib.dataloading import *
from lib.loss_functions import *
from lib.evaluation import *
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch import optim, nn
#from torch.utils.tensorboard import SummaryWriter

class NetworkPredicter(object):
    '''The class to handle the inference
    parameters:
        - task:  the task to predict
        - network: the model architecture
        - batch_size: the inference batch size
        - weight_path: pretrained weights
        - gpu: the gpu used
        - bootstrap_n: if greater than 0, perform bootstrap on the test set
        - output_dir: the directory the put the results
    '''

    def __init__(self, 
                 task,
                 network = 'vgg-ce-unet',                                  
                 batch_size = 1,
                 weight_path = None,
                 gpu = '0',
                 save = True,                 
                 bootstrap_n = 0,
                 output_dir = '/mnt/Liver/GE_study_hri/ContextEncoder/results/downstream/test/'):
        
        ## Set the default information
        self.info = OrderedDict()
        self.set_info(network, batch_size, weight_path, task)
        self.device = torch.device("cuda:%s" % gpu if torch.cuda.is_available() else "cpu")
        self.save_imgs = save              
        if bootstrap_n == 0:
            self.do_bootstrap = False
        else:
            self.do_bootstrap = True
            self.bootstrap_n = bootstrap_n
        
        ## Output folder path 
        self.exp_name = weight_path.split('/')[-2]
        self.output_dir = os.path.join(output_dir, task, self.exp_name)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            
    def bootstrap(self):
        pass
               
                                
    def set_info(self, network, batch_size, weight_path, task):
        self.info['network'] = network        
        self.info['batch_size'] = batch_size
        self.info['weight_path'] = weight_path
        self.info['task'] = task        
    
    def initalize_bootstrap_results_dict(self):
        pass
    
    def intialize_results_dict(self):
        pass
    
        
    def load_dataset(self, list_id = None, transform = None):       
        pass       
    
    def get_data_loader(self, bootstrap = False):        
        test_dataset = self.load_dataset(transform = self.transform['val']) 
        if not self.do_bootstrap:            
            test_loader = DataLoader(test_dataset, batch_size=self.info['batch_size'], num_workers = 4)       
        else:
            sampler = RandomSampler(test_dataset, replacement = True, num_samples = len(test_dataset))
            test_loader = DataLoader(test_dataset, batch_size=self.info['batch_size'], num_workers = 4, sampler = sampler)
        return test_loader
        
    def get_network(self):         
        net = VGGCEUNet(num_classes = self.info['num_classes'], activation = 'sigmoid')
        return net
    
    def get_transform(self):
        return get_transformer_norm()
        
    def predict(self):
        pass
        
    def evaluate(self):
        pass
    
    def save_results(self):
        pass
        
    def load_weights(self, weight_path):
        print(f'load weights from {weight_path}')
        self.network.load_state_dict(torch.load(weight_path, map_location = torch.device(self.device)))
        
        
        
class HRINetworkPredicter(NetworkPredicter):

    def __init__(self, task, img_dir, liver_mask_dir = None, kidney_mask_dir = None, 
                 network = 'vgg-ce-unet',                                  
                 batch_size = 1,
                 weight_path = None,
                 gpu = '0',
                 save = True,
                 bootstrap_n = 0,
                 output_dir = '/mnt/Liver/GE_study_hri/ContextEncoder/results/downstream/test/'):
                        
        super().__init__(task,network, batch_size,weight_path, gpu, save, bootstrap_n, output_dir)
        
        ## Dataset
        self.img_dir = img_dir
        self.kidney_mask_dir = kidney_mask_dir
        self.liver_mask_dir = liver_mask_dir 
        if not self.kidney_mask_dir and not self.liver_mask_dir:
            self.has_mask = False
        else:
            self.has_mask = True
        self.info['num_classes'] = 2
            
                
        self.dataset = self.load_dataset()
        self.file_names = [file.split('.')[0] for file in self.dataset.file_names]
        
        ## Output folder path 
        if not os.path.exists(os.path.join(self.output_dir, 'liver')):
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
        
    def initalize_bootstrap_results_dict(self):
        b_results = OrderedDict()
        b_results['bootstrap_n'] = []
        b_results['dice'] = []
        b_results['dice_std'] = []
        return b_results
        
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

    def bootstrap(self):
        self.bootstrap_result = self.initalize_bootstrap_results_dict()
                
        for i in range(self.bootstrap_n):            
            print(f"Bootstrap for exp {self.exp_name}, #{i+1}")
            self.results = self.intialize_results_dict()
            
            self.test_dataset = self.get_data_loader(bootstrap = True)            
            self.evaluate(exp_postfix = f'_b{i+1}')                    
            
            self.bootstrap_result['bootstrap_n'].append(i+1)
            self.bootstrap_result['dice'].append(self.results['avg_dice'])
            self.bootstrap_result['dice_std'].append(np.std(self.results['dice']))
            
            print(f"Average Dice Score: {self.results['avg_dice']}, +/- {np.std(self.results['dice'])}")
            
        bootstrap_results = pd.DataFrame({'bootsrap': self.bootstrap_result['bootstrap_n'],'dice': self.bootstrap_result['dice'], 'dice_std': self.bootstrap_result['dice_std']})
        bootstrap_results.to_csv(os.path.join(self.output_dir,'bootstrap_dice.csv'))
        
    
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
        
    def evaluate(self, exp_postfix = ''):
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
        if not self.do_bootstrap:
            self.save_results()
        else:
            self.save_results(exp_postfix)
            
  
    def save_results(self, postfix = ''):
        config = configparser.ConfigParser()        
        config['INFO'] = self.info
        config['RESULTS'] = {'avg_dice': self.results['avg_dice'],
                            'avg_liver_dice': self.results['avg_liver_dice'],
                            'avg_kidney_dice': self.results['avg_kidney_dice']}
                                
        
        all_dice = pd.DataFrame({'file': self.file_names,
                                'dice': self.results['dice'],
                                'liver_dice': self.results['liver_dice'],
                                'kidney_dice': self.results['kidney_dice']})
        
        with open(os.path.join(self.output_dir, f'exp{postfix}.ini'), 'w') as configfile:
            config.write(configfile)
            
        all_dice.to_csv(os.path.join(self.output_dir, f'dices{postfix}.csv'))
        
 
            
class SingleNetworkPredicter(NetworkPredicter):

    def __init__(self, task, img_dir, mask_dir, ext,
                 network = 'vgg-ce-unet',                                  
                 batch_size = 1,
                 weight_path = None,
                 gpu = '0',
                 save = True,
                 bootstrap_n = 0,                 
                 output_dir = '/mnt/Liver/GE_study_hri/ContextEncoder/results/downstream/test/'):
                        
        super().__init__(task,network, batch_size,weight_path, gpu, save, bootstrap_n, output_dir)
        
        ## Dataset
        self.img_dir = img_dir        
        self.mask_dir = mask_dir 
        self.ext = ext        
        self.info['num_classes'] = 1         
                
        self.dataset = self.load_dataset()        
        self.file_names = [f'image{i}' for i in range(len(self.dataset))]
        
        ## Output folder path 
        if not os.path.exists(os.path.join(self.output_dir, 'mask')):
            os.mkdir(os.path.join(self.output_dir, 'mask'))            
        self.output_dir_mask = os.path.join(self.output_dir, 'mask')       
        
                
        ## initialize results dictionary
        self.results = self.intialize_results_dict()
        
        # Network
        self.network = self.get_network()                        
        self.load_weights(weight_path)
        self.transform = self.get_transform()
        self.test_loader = self.get_data_loader()
        
        print("Network initizliation:")        
        print("Test Number: %s" % (len(self.test_loader) * self.info['batch_size']))
        
    def initalize_bootstrap_results_dict(self):
        b_results = OrderedDict()
        b_results['bootstrap_n'] = []
        b_results['dice'] = []
        b_results['dice_std'] = []
        return b_results
        
    def intialize_results_dict(self):
        results = OrderedDict()                
        results['dice'] = []        
        return results
    
        
        
    def load_dataset(self, list_id = None, transform = None):    
        return SingleClassDataset_Gray(img_dir = self.img_dir, mask_dir = self.mask_dir, list_id = list_id, 
                                           transform = transform, ori_ext = self.ext, mask_ext = self.ext)         

    def bootstrap(self):
        self.bootstrap_result = self.initalize_bootstrap_results_dict()
                
        for i in range(self.bootstrap_n):            
            print(f"Bootstrap for exp {self.exp_name}, #{i+1}")
            self.results = self.intialize_results_dict()
            
            self.test_dataset = self.get_data_loader(bootstrap = True)            
            self.evaluate(exp_postfix = f'_b{i+1}')                    
            
            self.bootstrap_result['bootstrap_n'].append(i+1)
            self.bootstrap_result['dice'].append(self.results['avg_dice'])
            self.bootstrap_result['dice_std'].append(np.std(self.results['dice']))
            
            print(f"Average Dice Score: {self.results['avg_dice']}, +/- {np.std(self.results['dice'])}")
                        
            
        bootstrap_results = pd.DataFrame({'bootsrap': self.bootstrap_result['bootstrap_n'],'dice': self.bootstrap_result['dice'], 'dice_std': self.bootstrap_result['dice_std']})
        bootstrap_results.to_csv(os.path.join(self.output_dir,'bootstrap_dice.csv'))
        
    
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
        
    def evaluate(self, exp_postfix = ''):                    
        self.network = self.network.to(self.device)                
        tot_dice = np.array([0,0])
        with torch.set_grad_enabled(False):
            self.network.eval()                            
            for i, (img, mask) in enumerate(self.test_loader):
            # Transfer to GPU
                imgs, masks = img.to(self.device), mask.to(self.device)

                masks_pred = self.network(imgs)
                masks_pred = (masks_pred > 0.5).float()

                dice_score = dice(masks_pred, masks).detach().cpu().numpy()                                
                self.results['dice'].append(dice_score)                

                if self.save_imgs:
                    save_image(masks_pred[0,0,:,:], os.path.join(self.output_dir_mask, self.file_names[i] + '.png'))
                            
        self.results['avg_dice'] = np.mean(self.results['dice'])        
   

        print("Finish Inference")        
        print("Mean Dice on the Test set:", self.results['avg_dice'])
        if not self.do_bootstrap:
            self.save_results()
        else:
            self.save_results(exp_postfix)
            
  
    def save_results(self, postfix = ''):
        config = configparser.ConfigParser()        
        config['INFO'] = self.info
        config['RESULTS'] = {'avg_dice': self.results['avg_dice']}
                                
        
        all_dice = pd.DataFrame({'file': self.file_names,
                                'dice': self.results['dice']})
        
        with open(os.path.join(self.output_dir, f'exp{postfix}.ini'), 'w') as configfile:
            config.write(configfile)
            
        all_dice.to_csv(os.path.join(self.output_dir, f'dices{postfix}.csv'))  
            
            
        
        
    
