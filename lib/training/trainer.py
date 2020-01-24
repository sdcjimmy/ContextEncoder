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
from lib.utilities.image_sampler import ImageSampler
from torchvision import transforms
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch import optim, nn
#from torch.utils.tensorboard import SummaryWriter

class NetworkTrainer(object):

    def __init__(self,                  
                 opt = 'adam', 
                 network = 'ce-net',
                 lr = 0.001,                 
                 batch_size = 4,
                 epochs = 10, 
                 dcm_loss = True, 
                 padding_center = False,                 
                 center_distribution = None, 
                 experiment = 'TEST',                 
                 gpu = '0',
                 ):
        
        ## Set the default information
        self.info = OrderedDict()
        self.set_info(opt, network, lr, batch_size, epochs, dcm_loss, padding_center, center_distribution, experiment)
        self.set_default_info()
        self.device = torch.device("cuda:%s" % gpu if torch.cuda.is_available() else "cpu")                
        
        
        ## Dataset              
        if self.info['experiment'] == 'TEST':
            self.all_dataset = self.load_dataset(list_id = range(101))
        else:
            self.all_dataset = self.load_dataset()        
        
        ## Output folder path 
        self.output_dir = os.path.join("./results", experiment)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            
        self.writer = SummaryWriter(log_dir = self.output_dir)
        
        ## initialize results dictionary
        self.results = self.intialize_results_dict()
        
        # Network
        self.generator, self.discriminator = self.get_network(self.info['network'])
        #self.optimizer = self.get_optimizer()                
        self.criteriaMSE = self.get_loss_fx('MSE')
        self.criteriaBCE = self.get_loss_fx('BCE')
        self.transform = self.get_transform()
        self.train_loader, self.val_loader = self.get_data_loader()
        
        print("Network initizliation:")
        print("Training Number: %s" % (len(self.train_loader) * self.info['batch_size']))
        print("Validation Number: %s" % (len(self.val_loader) * self.info['batch_size']))

        self.save_results(init = True)
        
    def set_info(self, opt, network, lr, batch_size, epochs, dcm_loss, padding_center, center_distribution, experiment):        
        self.info['optimizer'] = opt
        self.info['network'] = network
        self.info['learning_rate'] = lr
        self.info['batch_size'] = batch_size
        self.info['epochs'] = epochs
        self.info['dcm_loss'] = dcm_loss
        self.info['padding_center'] = padding_center
        self.info['center_distribution'] = center_distribution
        self.info['experiment'] = experiment
        
    def set_default_info(self):
        self.info['Generator_adv_loss'] = 0.01
        self.info['Generator_mse_loss'] = 0.99
        self.info['Discriminator_adv_loss'] = 1
        self.info['Discriminator_dcm_loss'] = 0
        self.info['sample_interval'] = 5
        self.info['output_resize'] = False
        
        if self.info['dcm_loss']:
            self.info['n_dcm_labels'] = 9
        else:
            self.info['n_dcm_labels'] = 0
    
    def update_info(key, value):
        self.info[key] = value    
    
    def intialize_results_dict(self):
        results = OrderedDict()
        results['G_training_loss'] = []
        results['D_training_loss'] = []
        results['validation_loss'] = []        
        
        results['validation_mse_loss'] = []                        
        results['validation_adv_loss'] = []
        results['best_loss'] = float('inf')
        results['best_MSE'] = float('inf')
        
        return results
    
        
    def load_dataset(self, list_id = None, transform = None, inpaint = False, rand = None):
        return SSIDataset(list_id = list_id, transform = transform, inpaint = inpaint, rand = rand, output_resize = self.info['output_resize']) 
        
        
    def data_split(self, validation_split = 0.2, random_seed = 123, shuffle_dataset = True):
        dataset_size = len(self.all_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset :
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        return train_indices, val_indices
        
    def get_data_loader(self):
        train_indices, val_indices = self.data_split()
        train_dataset = self.load_dataset(list_id = train_indices, transform = self.transform['train'], inpaint = True, rand = self.info['center_distribution'])
        val_dataset = self.load_dataset(list_id = val_indices, transform = self.transform['val'], inpaint = True, rand = self.info['center_distribution'])
        
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
    
    def get_network(self, net = 'ce-net'):                
        if net == 'ce-net':
            return CENet(), Discriminator(n_classes = self.info['n_dcm_labels'] + 1)
        elif net == 'vgg-unet':
            self.info['output_resize'] = True
            return VGGCEUNet(), Discriminator(n_classes = self.info['n_dcm_labels'] + 1, n_block = 5)
        
    
    def get_transform(self):
        return get_transformer_norm()
    
    
    def get_loss_fx(self, loss_fx):
        if loss_fx == 'MSE':
            return nn.MSELoss()
        elif loss_fx == 'BCE':
            return nn.BCELoss()        
        
        
    def _get_output_labels(self, output):        
        if self.info['dcm_loss']:
            # Discrimation labels
            pred_dlabel = output[:,0]                
            # DICOM Labels
            pred_dicom = output[:,1:]
        else:
            pred_dlabel = output
            pred_dicom = None
        
        return pred_dlabel, pred_dicom
    
    def padding_center(self, imgs, centers):
        new_img = imgs.clone()
        new_img[:,:,64:192, 100:300] = centers
        return new_img
        
    def get_disc_input(self, imgs, centers):
        if self.info['padding_center']:
            disc_input = self.padding_center(imgs, centers)
        else:
            disc_input = centers
        return disc_input

    def sample_images(self, imgs, centers, pred_centers, epoch):
        true = self.padding_center(imgs, centers)        
        true = make_grid(true, normalize= True, scale_each = True)
        self.writer.add_image('true_images', true, epoch)
                
        pred = self.padding_center(imgs, pred_centers)
        pred = make_grid(pred, normalize= True, scale_each = True)
        self.writer.add_image('pred_images', pred, epoch)
        
    
    def evaluate(self, dataloader, epoch):
        self.generator.eval()
        self.discriminator.eval()  
        
        sample = False
        if (epoch+1) % self.info['sample_interval'] == 0:
            sampler = ImageSampler()
            sample = True
        
        #dlabel = torch.FloatTensor(self.info['batch_size'])
        MSE_loss = 0 
        Adv_loss = 0
        
        for i, (imgs, centers, dcm_labels) in enumerate(dataloader):               
            batch_size = dcm_labels.clone().size(0)
            
            imgs, centers, dcm_labels = imgs.to(self.device), centers.to(self.device), dcm_labels.to(self.device)
                        
            pred_centers = self.generator(imgs)
            
            if sample:
                sampler.sample(imgs, centers, pred_centers)                
                
            # Advasarial loss            
            # Fake Image succesfully fool the discriminator
            dlabel = torch.FloatTensor(batch_size).fill_(1).to(self.device)
            disc_input = self.get_disc_input(imgs, pred_centers)
            output = self.discriminator(disc_input)
            pred_dlabel, _ = self._get_output_labels(output)                               
                    
            lossAdv_Encoder = self.criteriaBCE(pred_dlabel, dlabel)

            # MSE Loss           
            lossMSE_Encoder = self.criteriaMSE(pred_centers, centers)                             
            
            MSE_loss += lossMSE_Encoder.item()
            Adv_loss += lossAdv_Encoder.item()
        
        if sample:
            true, pred = sampler.make_grid()
            self.writer.add_image('true_images', true, epoch)
            self.writer.add_image('pred_images', pred, epoch)
        
        MSE_loss /= i
        Adv_loss /= i
        
        return MSE_loss, Adv_loss
    
    
    def train(self):
        self.generator = self.generator.to(self.device)        
        self.discriminator = self.discriminator.to(self.device)
        
        epochs = self.info['epochs']                                            
        n_iter = 0 
        
        # Define the optimizer for the network
        optG = optim.Adam(self.generator.parameters(), lr = self.info['learning_rate'])         
        optD = optim.Adam(self.discriminator.parameters(), lr = self.info['learning_rate'])
        
        for epoch in range(epochs):
            print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
            self.generator.train()
            self.discriminator.train()
            
            G_epoch_loss = 0         
            D_epoch_loss = 0
            
            dlabel = torch.FloatTensor(self.info['batch_size']).to(self.device)
            
            for i, (imgs, centers, dcm_labels) in enumerate(self.train_loader):                     
                batch_size = imgs.size(0)
                imgs, centers, dcm_labels = imgs.to(self.device), centers.to(self.device), dcm_labels.to(self.device)           
                
                # -----------------------
                # Train Generator (Encoder)
                # -----------------------
                optG.zero_grad()       
                
                pred_centers = self.generator(imgs)
                
                # Advasarial loss                
                #dlabel.data.resize_(batch_size).fill_(1)            
                dlabel = torch.FloatTensor(batch_size).fill_(1).to(self.device)
                output = self.discriminator(pred_centers)
                pred_dlabel, _ = self._get_output_labels(output)            
                
                lossAdv_Encoder = self.criteriaBCE(pred_dlabel, dlabel)

                # MSE Loss
                lossMSE_Encoder = self.criteriaMSE(pred_centers, centers)

                                        
                lossG = self.info['Generator_adv_loss'] * lossAdv_Encoder + self.info['Generator_mse_loss']* lossMSE_Encoder
                                                                
                G_epoch_loss += lossG.item()
                
                # Write the loss to summary writer
                self.writer.add_scalar('Loss/train_ADV_G', lossAdv_Encoder.item(), n_iter)
                self.writer.add_scalar('Loss/train_MSE_G', lossMSE_Encoder.item(), n_iter)
                self.writer.add_scalar('Loss/train_G', lossG.item(), n_iter)
                                
                
                lossG.backward()
                
                optG.step()
                
                
                # -----------------------
                # Train Discriminator
                # -----------------------
                
                
                # Discriminator - Train with real
                optD.zero_grad()
                #dlabel.data.resize_(batch_size).fill_(1)
                dlabel = torch.FloatTensor(batch_size).fill_(1).to(self.device)
                
                # Padding the original context as the input for discriminator
                if self.info['padding_center']:
                    disc_input = self.padding_center(imgs, centers)
                else:
                    disc_input = centers
                
                output = self.discriminator(disc_input)
                
                # Get the output labels for discriminator
                pred_dlabel, pred_dicom = self._get_output_labels(output)                            

                lossAdv_real = self.criteriaBCE(pred_dlabel, dlabel)        
                
                if self.info['dcm_loss']:
                    lossDCM = self.criteriaBCE(pred_dicom, dcm_labels.float())
                else:
                    lossDCM = torch.Tensor([0]).to(self.device)

                # Discriminator - Train with fake
                
                pred_centers = self.generator(imgs)
                #dlabel.data.resize_(batch_size).fill_(0)
                dlabel = torch.FloatTensor(batch_size).fill_(0).to(self.device)

                if self.info['padding_center']:
                    disc_input = self.padding_center(imgs, pred_centers)
                else:
                    disc_input = pred_centers
                                
                output = self.discriminator(pred_centers)
                pred_dlabel = output[:, 0]
                                
                lossAdv_fake = self.criteriaBCE(pred_dlabel, dlabel)
                
                
                lossD = self.info['Discriminator_adv_loss'] * (lossAdv_real + lossAdv_fake) + self.info['Discriminator_dcm_loss'] * lossDCM
                
                D_epoch_loss += lossD.item()
                self.writer.add_scalar('Loss/train_ADV_D', (lossAdv_real + lossAdv_fake).item(), n_iter)
                self.writer.add_scalar('Loss/train_DCM_D', lossDCM.item(), n_iter)
                self.writer.add_scalar('Loss/train_D', lossD.item(), n_iter)
                
                n_iter += 1
                
                lossD.backward()
                optD.step()

                if i % 100 == 0:
                    print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                          % (epoch, epochs, i, len(self.train_loader),
                             lossD.item(), lossG.item()))    

                                
            D_epoch_loss /= i+1
            G_epoch_loss /= i+1
            self.results['G_training_loss'].append(D_epoch_loss)
            self.results['D_training_loss'].append(G_epoch_loss)
            print(f'Epoch finished ! D_Loss: {D_epoch_loss}, G_Loss: {G_epoch_loss}' )
                        
            # Validation
            with torch.set_grad_enabled(False):
                self.generator.eval()                                            
                MSE_loss, Adv_loss  = self.evaluate(self.val_loader, epoch = epoch)
                
                self.results['validation_mse_loss'].append(MSE_loss)
                self.results['validation_adv_loss'].append(Adv_loss)
                                                
                print('Validation MSE Loss: {}'.format(MSE_loss))
                print('Validation Adv Loss: {}'.format(Adv_loss))
                                
                if MSE_loss + Adv_loss < self.results['best_loss']:
                    self.results['best_loss'] = MSE_loss + Adv_loss
                    self.results['best_MSE'] = MSE_loss                                        
                    self.results['best_epoch'] = epoch + 1
                    torch.save(self.generator.state_dict(), os.path.join(self.output_dir, "epoch{}.pth".format(epoch+1)))                   
                    print("Best Validation MSE improved!")                                    
                    
                elif (epoch+1) % self.info['sample_interval'] == 0:
                    torch.save(self.generator.state_dict(), os.path.join(self.output_dir, "epoch{}.pth".format(epoch+1)))       

        # Save the training dice score using the best weights
        #self.evaluate_train()

    def evaluate_train(self):                
        weight_path =  os.path.join(self.output_dir, "epoch{}.pth".format(self.results['best_epoch']))
        self.network.load_state_dict(torch.load(weight_path))
        
        with torch.set_grad_enabled(False):
            self.network.eval()      
            train_loss, train_acc, train_precision, train_recall  = eval_net(self.network, self.train_loader, self.criterion, self.device)   
                        
            self.results['train_accuracy'] = train_acc
            self.results['train_precision'] = train_precision
            self.results['train_recall'] = train_recall
            
            
        
    def plot_training():
        pass
    def save_results(self, init = False):
        config = configparser.ConfigParser()        
        config['INFO'] = self.info

        if init:
            with open(os.path.join(self.output_dir, 'exp.ini'), 'w') as configfile:
                config.write(configfile)
            return
        
        config['BEST RESULTS'] = {'val_mse': self.results['best_MSE'],                             
                                  'best_epoch': self.results['best_epoch']}        
        
        with open(os.path.join(self.output_dir, 'exp.ini'), 'w') as configfile:
            config.write(configfile)
                        
        loss_history = pd.DataFrame({'generator_training_loss': self.results['G_training_loss'],
                                    'discriminator_loss': self.results['D_training_loss']})
        loss_history.to_csv(os.path.join(self.output_dir, 'loss_history.csv'))
        
        torch.save(self.generator.state_dict(), os.path.join(self.output_dir, "epoch_last.pth"))
        
    def load_weights(self, weight_path):
        self.network.load_state_dict(torch.load(weight_path))
        
        
        
        

            
            
            
            
            
        
        
    
