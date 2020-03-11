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
from torch import optim, nn
#from torch.utils.tensorboard import SummaryWriter

class SegNetworkTrainer(object):

    def __init__(self,
                 network = 'unet', 
                 opt = 'adam', 
                 lr = 0.001,
                 reg = 0, 
                 loss_fx = 'DL',
                 batch_size = 4,
                 epochs = 10,                  
                 pretrain = False, 
                 self_train = '',
                 freeze = False,
                 shrink = 1,
                 experiment = 'TEST',
                 gpu = '0'):
        
        ## Set the default information
        self.info = OrderedDict()
        self.set_info(network, opt, lr, reg, loss_fx, batch_size, epochs, pretrain, self_train, freeze, shrink, experiment)
        self.device = torch.device("cuda:%s" % gpu if torch.cuda.is_available() else "cpu")
         
        ## initialize results dictionary
        self.results = self.intialize_results_dict()
                        
        
    def set_info(self, network, opt, lr, reg, loss_fx, batch_size, epochs, pretrain, self_train, freeze, shrink, experiment):
        self.info['network'] = network
        self.info['optimizer'] = opt
        self.info['learning_rate'] = lr
        self.info['regularization_weights'] = reg
        self.info['batch_size'] = batch_size
        self.info['loss_function'] = loss_fx
        self.info['epochs'] = epochs
        self.info['pretrain'] = pretrain
        self.info['self_train'] = self_train
        self.info['freeze'] = freeze
        self.info['shrink'] = shrink
        self.info['experiment'] = experiment
    
    def update_info(self, key, value):
        self.info[key] = value    
    
    def intialize_results_dict(self):
        results = OrderedDict()
        results['training_loss'] = []
        results['validation_loss'] = []
        results['training_dice'] = []
        results['validation_dice'] = []
        
        results['best_dice'] = 0
        return results
    
        
    def data_split(self, validation_split = 0.2, random_seed = 123, shuffle_dataset = True, shrink = 1):
        dataset_size = len(self.all_dataset)  
        indices = list(range(dataset_size))

        split = int(np.floor(validation_split * len(indices)))
        if shuffle_dataset :
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        if shrink < 1:
            train_indices = random.sample(train_indices, int(np.floor(shrink * len(train_indices))))
        return train_indices, val_indices
        
    def get_data_loader(self):
        train_indices, val_indices = self.data_split(shrink = self.info['shrink'])
        train_dataset = self.load_dataset(list_id = train_indices, transform = self.transform['train'])
        val_dataset = self.load_dataset(list_id = val_indices, transform = self.transform['val'])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.info['batch_size'], num_workers = 4)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.info['batch_size'], num_workers = 4)       
        return train_loader, val_loader
    
    def get_optimizer(self):
        opt, lr, reg = self.info['optimizer'], self.info['learning_rate'], self.info['regularization_weights']
        if opt == 'adam':
            return optim.Adam(self.network.parameters(), lr=lr, weight_decay = reg)
        elif opt == 'rmsprop':
            return optim.RMSprop(self.network.parameters(), lr = lr, weight_decay = reg)
        else:
            return optim.SGD(self.network.parameters(), lr = lr, weight_decay = reg)
    
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
        elif self.info['network'] == 'vgg-ce-unet':
            net = VGGCEUNet(num_classes = self.info['num_classes'], pretrained = self.info['pretrain'], self_trained = self.info['self_train'], freeze = self.info['freeze'], activation = 'sigmoid')
        
        return net
    
    def get_loss_fx(self):
        loss_fx = self.info['loss_function']
        if loss_fx == "BCE":
            criterion = nn.BCELoss()
        elif loss_fx == "DL":
            criterion = SoftDiceLoss(batch_dice=True)
        elif loss_fx == "GDL":
            criterion = GDL(batch_dice=True)
        
        return criterion
    
    def get_transform(self):
        return get_transformer_norm()
    
    def train(self):
        # Customize for each subclass
        pass            
        
    def evaluate(self, data_loader, epoch):
        # Customize for each subclass
        pass            

    def evaluate_train(self):                
        pass
     
    def plot_training():
        pass
    
    def save_results(self):
        config = configparser.ConfigParser()        
        config['INFO'] = self.info

        if init:
            with open(os.path.join(self.output_dir, 'exp.ini'), 'w') as configfile:
                config.write(configfile)
            return
        
        config['BEST RESULTS'] = {'val_dice': self.results['best_dice'],                             
                                  'best_epoch': self.results['best_epoch']}        
        
        with open(os.path.join(self.output_dir, 'exp.ini'), 'w') as configfile:
            config.write(configfile)
                        
        loss_history = pd.DataFrame({'generator_training_loss': self.results['G_training_loss'],
                                    'discriminator_loss': self.results['D_training_loss']})
        loss_history.to_csv(os.path.join(self.output_dir, 'loss_history.csv'))
        
        torch.save(self.generator.state_dict(), os.path.join(self.output_dir, "epoch_last.pth"))
        
    def load_weights(self, weight_path):
        self.network.load_state_dict(torch.load(weight_path), map_location = self.device)
        
        

class HRISegNetworkTrainer(SegNetworkTrainer):

    def __init__(self,
                 img_dir, kidney_mask_dir, liver_mask_dir,
                 network = 'unet', 
                 opt = 'adam', 
                 lr = 0.001,
                 reg = 0, 
                 loss_fx = 'DL',
                 batch_size = 4,
                 epochs = 10,                  
                 pretrain = False, 
                 self_train = '',
                 freeze = False,
                 shrink = 1,
                 experiment = 'TEST',
                 gpu = '0'):
        
        super().__init__(network, opt, lr, reg, loss_fx, batch_size, epochs, pretrain, self_train, freeze, shrink, experiment, gpu)
        
        ## Set default parameters for the HRI segmentation
        self.update_info('num_classes', 2)
        
        ## Dataset
        self.img_dir = img_dir
        self.kidney_mask_dir = kidney_mask_dir
        self.liver_mask_dir = liver_mask_dir        
        self.all_dataset = self.load_dataset()
        
        ## Output folder path 
        self.output_dir = os.path.join("/mnt/Liver/GE_study_hri/ContextEncoder/results/downstream/hri/", experiment)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
                                 
        ## initialize results dictionary
        self.results = self.intialize_results_dict()
        
        # Network        
        self.network = self.get_network()        
        self.optimizer = self.get_optimizer()
        self.criterion = self.get_loss_fx()
        self.transform = self.get_transform()
        self.train_loader, self.val_loader = self.get_data_loader()                
        
        print("Network initizliation:")
        print("Training Number: %s" % (len(self.train_loader) * self.info['batch_size']))
        print("Validation Number: %s" % (len(self.val_loader) * self.info['batch_size']))
        
        self.save_results(init=True)
        
    def intialize_results_dict(self):
        results = OrderedDict()
        results['training_loss'] = []
        results['validation_loss'] = []
        results['training_dice'] = []
        results['validation_dice'] = []                
        
        results['best_dice'] = 0
        results['liver_best_dice'] = 0
        results['kidney_best_dice'] = 0
        
        results['train_dice'] = 0
        results['train_liver_dice'] = 0
        results['train_kidney_dice'] = 0                
        
        return results
    
    def load_dataset(self, list_id = None, transform = None):
        return LiverHRIDatasetMulti_simple(img_dir = self.img_dir, kidney_mask_dir = self.kidney_mask_dir, liver_mask_dir = self.liver_mask_dir, list_id = list_id, transform = transform) 

    def train(self):        
        self.network = self.network.to(self.device)
        
        epochs = self.info['epochs']
        for epoch in range(epochs):
            print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
            self.network.train()
            epoch_loss = 0            
            for i, (img, mask) in enumerate(self.train_loader):
                #print('***',img.shape)
                imgs, masks = img.to(self.device), mask.to(self.device)
                masks_pred = self.network(imgs)

                loss = self.criterion(masks_pred, masks)

                epoch_loss += loss.item()

                #print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))
                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()
                if (i+1) % 5 == 0:
                    print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f ' %(epoch+1, epochs, i+1, len(self.train_loader), loss.item()))
                    
            epoch_loss /= i+1
            self.results['training_loss'].append(epoch_loss)
            print('Epoch finished ! Loss: {}'.format(epoch_loss))
                        
            # Validation
            with torch.set_grad_enabled(False):
                self.network.eval()                            
                val_loss, (val_dice_liver, val_dice_kidney) = self.evaluate(self.val_loader, by_class = True)                
                val_avg_dice = (val_dice_liver + val_dice_kidney)/2
                                
                self.results['validation_loss'].append(val_loss)                
                self.results['validation_dice'].append(val_avg_dice)
        
                print('Validation Liver Dice Coeff: {}'.format(val_dice_liver))
                print('Validation Kidney Dice Coeff: {}'.format(val_dice_kidney))

                
                if val_avg_dice > self.results['best_dice']:
                    self.results['best_dice'] = val_avg_dice
                    self.results['liver_best_dice'] = val_dice_liver
                    self.results['kidney_best_dice'] = val_dice_kidney                
                    self.results['best_epoch'] = epoch+1
                    torch.save(self.network.state_dict(), os.path.join(self.output_dir, "epoch{}.pth".format(epoch+1)))                   
                    print("Best validation dice score improved!")                
                    print('Best Average Dice Coeff: {}'.format(val_avg_dice))
                    print('Best Liver Dice Coeff: {}'.format(val_dice_liver))
                    print('Best Kidney Dice Coeff: {}'.format(val_dice_kidney))

        # Save the training dice score using the best weights
        self.evaluate_train()
        
    def evaluate(self, data_loader, by_class = False):
        """Evaluation without the densecrf with the dice coefficient"""
        self.network.eval()
    
        tot_dice = 0
        tot_loss = 0
        for i, (img, mask) in enumerate(data_loader):
            # Transfer to GPU
            imgs, masks = img.to(self.device), mask.to(self.device)
        
            # Model computations        
            val_masks_pred = self.network(imgs)
            val_true_masks = masks
            loss = self.criterion(val_masks_pred, val_true_masks)
            tot_loss += loss.item()

            val_masks_pred = (val_masks_pred > 0.5).float()
        
            if by_class:
                dc = dice(val_masks_pred, val_true_masks).detach().cpu().numpy()
            else:
                dc = dice(val_masks_pred, val_true_masks, return_mean = True).item()

            tot_dice += dc
        return tot_loss/(i+1), tot_dice/(i+1)
        
    
    def evaluate_train(self):                
        weight_path =  os.path.join(self.output_dir, "epoch{}.pth".format(self.results['best_epoch']))
        self.network.load_state_dict(torch.load(weight_path))
        
        with torch.set_grad_enabled(False):
            self.network.eval()                            
            train_loss, (train_dice_liver, train_dice_kidney) = self.evaluate(self.train_loader, by_class = True)  
            train_avg_dice = (train_dice_liver + train_dice_kidney)/2
            self.results['train_dice'] = train_avg_dice
            self.results['train_liver_dice'] = train_dice_liver
            self.results['train_kidney_dice'] = train_dice_kidney
            
                    
    def plot_training():
        pass
    def save_results(self, init=False):
        config = configparser.ConfigParser()        
        config['INFO'] = self.info                   

        if init:
            with open(os.path.join(self.output_dir, 'exp.ini'), 'w') as configfile:
                config.write(configfile)
            return
        
                                
        config['BEST RESULTS'] = {'val_dice': self.results['best_dice'],
                             'val_liver_dice': self.results['liver_best_dice'],
                             'val_kidney_dice': self.results['kidney_best_dice'],
                              'best_epoch': self.results['best_epoch']}
        config['TRAIN RESULTS'] = {'train_dice': self.results['train_dice'],                            
                             'train_liver_dice': self.results['train_liver_dice'],                             
                             'train_kidney_dice': self.results['train_kidney_dice'],
                            }
                
        
        with open(os.path.join(self.output_dir, 'exp.ini'), 'w') as configfile:
            config.write(configfile)
            
            
        loss_history = pd.DataFrame({'training_loss': self.results['training_loss'],
                                    'validation_loss': self.results['validation_loss']})
        loss_history.to_csv(os.path.join(self.output_dir, 'loss_history.csv'))
        
        

            
            

class SingleSegNetworkTrainer(SegNetworkTrainer):

    def __init__(self,
                 img_dir, mask_dir, ext, task,
                 network = 'unet', 
                 opt = 'adam', 
                 lr = 0.001,
                 reg = 0, 
                 loss_fx = 'DL',
                 batch_size = 4,
                 epochs = 10,                  
                 pretrain = False, 
                 self_train = '',
                 freeze = False,
                 shrink = 1,
                 experiment = 'TEST',                 
                 gpu = '0'):
        
        super().__init__(network, opt, lr, reg, loss_fx, batch_size, epochs, pretrain, self_train, freeze, shrink, experiment, gpu)
        
        ## Set default parameters for the HRI segmentation
        self.update_info('num_classes', 1)
        
        ## Dataset
        self.img_dir = img_dir        
        self.mask_dir = mask_dir        
        self.ext = ext
        self.all_dataset = self.load_dataset()
        
        ## Output folder path 
        self.output_dir = os.path.join("/mnt/Liver/GE_study_hri/ContextEncoder/results/downstream/",task, experiment)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
                                 
        ## initialize results dictionary
        self.results = self.intialize_results_dict()
        
        # Network        
        self.network = self.get_network()        
        self.optimizer = self.get_optimizer()
        self.criterion = self.get_loss_fx()
        self.transform = self.get_transform()
        self.train_loader, self.val_loader = self.get_data_loader()                
        
        print("Network initizliation:")
        print("Training Number: %s" % (len(self.train_loader) * self.info['batch_size']))
        print("Validation Number: %s" % (len(self.val_loader) * self.info['batch_size']))
        
        self.save_results(init=True)
        
    def intialize_results_dict(self):
        results = OrderedDict()
        results['training_loss'] = []
        results['validation_loss'] = []
        results['training_dice'] = []
        results['validation_dice'] = []                
        
        results['best_dice'] = 0        
        
        results['train_dice'] = 0        
        
        return results
    
    def load_dataset(self, list_id = None, transform = None):
        return SingleClassDataset_Gray(img_dir = self.img_dir, mask_dir = self.mask_dir, list_id = list_id, 
                                           transform = transform, ori_ext = self.ext, mask_ext = self.ext) 

    def train(self):        
        self.network = self.network.to(self.device)
        
        epochs = self.info['epochs']
        for epoch in range(epochs):
            print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
            self.network.train()
            epoch_loss = 0            
            for i, (img, mask) in enumerate(self.train_loader):
                #print('***',img.shape)
                imgs, masks = img.to(self.device), mask.to(self.device)
                masks_pred = self.network(imgs)

                loss = self.criterion(masks_pred, masks)

                epoch_loss += loss.item()

                #print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))
                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()
                if (i+1) % 5 == 0:
                    print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f ' %(epoch+1, epochs, i+1, len(self.train_loader), loss.item()))
                    
            epoch_loss /= i+1
            self.results['training_loss'].append(epoch_loss)
            print('Epoch finished ! Loss: {}'.format(epoch_loss))
                        
            # Validation
            with torch.set_grad_enabled(False):
                self.network.eval()                            
                val_loss, val_dice = self.evaluate(self.val_loader)                                
                                
                self.results['validation_loss'].append(val_loss)                
                self.results['validation_dice'].append(val_dice)
                        
                print('Validation dice Coeff: {}'.format(val_dice))
                
                if val_dice > self.results['best_dice']:
                    self.results['best_dice'] = val_dice                    
                    self.results['best_epoch'] = epoch+1
                    torch.save(self.network.state_dict(), os.path.join(self.output_dir, "epoch{}.pth".format(epoch+1)))                   
                    print("Best validation dice score improved!")                
                    print('Best Dice Coeff: {}'.format(val_dice))                    

        # Save the training dice score using the best weights
        self.evaluate_train()
        
    def evaluate(self, data_loader):
        """Evaluation without the densecrf with the dice coefficient"""
        self.network.eval()
    
        tot_dice = 0
        tot_loss = 0
        for i, (img, mask) in enumerate(data_loader):
            # Transfer to GPU
            imgs, masks = img.to(self.device), mask.to(self.device)
        
            # Model computations        
            val_masks_pred = self.network(imgs)
            val_true_masks = masks
            loss = self.criterion(val_masks_pred, val_true_masks)
            tot_loss += loss.item()

            val_masks_pred = (val_masks_pred > 0.5).float()            
                    
            dc = dice(val_masks_pred, val_true_masks, return_mean = True).item()

            tot_dice += dc
        return tot_loss/(i+1), tot_dice/(i+1)
        
    
    def evaluate_train(self):                
        weight_path =  os.path.join(self.output_dir, "epoch{}.pth".format(self.results['best_epoch']))
        self.network.load_state_dict(torch.load(weight_path))
        
        with torch.set_grad_enabled(False):
            self.network.eval()                            
            train_loss, train_dice = self.evaluate(self.train_loader)              
            self.results['train_dice'] = train_dice            
                    
    def plot_training():
        pass
    def save_results(self, init=False):
        config = configparser.ConfigParser()        
        config['INFO'] = self.info                   

        if init:
            with open(os.path.join(self.output_dir, 'exp.ini'), 'w') as configfile:
                config.write(configfile)
            return
        
                                
        config['BEST RESULTS'] = {'val_dice': self.results['best_dice'],                             
                              'best_epoch': self.results['best_epoch']}
        config['TRAIN RESULTS'] = {'train_dice': self.results['train_dice']                             
                            }
                
        
        with open(os.path.join(self.output_dir, 'exp.ini'), 'w') as configfile:
            config.write(configfile)
            
            
        loss_history = pd.DataFrame({'training_loss': self.results['training_loss'],
                                    'validation_loss': self.results['validation_loss']})
        loss_history.to_csv(os.path.join(self.output_dir, 'loss_history.csv'))
        
            
            
        
        
    
