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
from torch.nn.functional import interpolate
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch import optim, nn

from .trainer import NetworkTrainer
#from torch.utils.tensorboard import SummaryWriter

class LinearProjectNetworkTrainer(NetworkTrainer):

    def __init__(self,                  
                 opt = 'adam', 
                 network = 'ce-net',
                 lr = 0.001,                 
                 batch_size = 4,
                 epochs = 10, 
                 dcm_loss = True, 
                 loss_type = 'hinge',
                 padding_center = False,                 
                 center_distribution = None, 
                 experiment = 'TEST',                 
                 gpu = '0',
                 cluster = False,                 
                 ):
        
        super().__init__(opt, network, lr, batch_size, epochs, dcm_loss, padding_center, center_distribution, experiment, gpu, cluster)
                        
        
        self.update_info('loss_type', loss_type)
        self.set_default_info()
        
        # Update the network
        self.generator, self.discriminator = self.get_network(net = network)
        self.train_loader, self.val_loader = self.get_data_loader()
        self.criteriaGENloss, self.criteriaDISloss = self.get_gan_loss(loss_type)
        
        self.save_results(init = True)
                        
    def set_default_info(self):
        self.info['Generator_adv_loss'] = 0.05
        self.info['Generator_mse_loss'] = 0.95
        self.info['Discriminator_adv_loss'] = 1.0
        self.info['Discriminator_dcm_loss'] = 0.0
        self.info['sample_interval'] = 10
        self.info['output_resize'] = False
        
        if self.info['dcm_loss']:
            self.info['n_dcm_labels'] = 9       
        else:
            self.info['n_dcm_labels'] = 0
    
    def get_network(self, net = 'ce-net'):                
        if net == 'ce-net':                        
            return CENet(), LinearDiscriminator(n_input = 3, n_classes = self.info['n_dcm_labels'])
        elif net == 'vgg-unet':
            self.info['output_resize'] = True
            return VGGCEUNet(num_classes = 3, activation = 'tanh'), LinearDiscriminator(n_input = 3, n_classes = self.info['n_dcm_labels'])                
        
    
    def get_gan_loss(self, loss_type):
        print(f'Loss function type: {loss_type}')
        return GenLoss(loss_type = loss_type), DisLoss(loss_type = loss_type)
    
    def get_disc_input(self, imgs, centers):
        if self.info['padding_center']:
            disc_input = self.padding_center(imgs, centers)
        else:
            disc_input = centers
        return disc_input

    
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
                if self.info['network'] == 'vgg-unet':
                    centers = interpolate(centers, scale_factor = 0.5)
                    pred_centers = interpolate(pred_centers, scale_factor = 0.5)
                sampler.sample(imgs, centers, pred_centers)                
                
            # Advasarial loss            
            # Fake Image succesfully fool the discriminator            
            disc_input = self.get_disc_input(imgs, pred_centers)
            dis_fake = self.discriminator(disc_input, dcm_labels)
                                                        
            lossAdv_Encoder = self.criteriaGENloss(dis_fake)
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
        optD = optim.Adam(self.discriminator.parameters(), lr = self.info['learning_rate']/10)
        
        for epoch in range(epochs):
            print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
            self.generator.train()
            self.discriminator.train()
            
            G_epoch_loss = 0         
            D_epoch_loss = 0
            
            dlabel = torch.FloatTensor(self.info['batch_size']).to(self.device)
            
            for i, (imgs, centers, dcm_labels) in enumerate(self.train_loader):     
                print(f'center size:{centers.size()}')
                batch_size = imgs.size(0)
                imgs, centers, dcm_labels = imgs.to(self.device), centers.to(self.device), dcm_labels.to(self.device)           
                
                # -----------------------
                # Train Generator (Encoder)
                # -----------------------
                optG.zero_grad()       
                
                pred_centers = self.generator(imgs)
                                
                # Advasarial loss                                                
                disc_input = self.get_disc_input(imgs, pred_centers)
                dis_fake = self.discriminator(disc_input, dcm_labels)
                lossAdv_Encoder = self.criteriaGENloss(dis_fake)
                                
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
                                                
                # Padding the original context as the input for discriminator
                disc_input = self.get_disc_input(imgs, centers)      
                dis_real = self.discriminator(disc_input, dcm_labels)                                                

                # Discriminator - Train with fake                                
                pred_centers = self.generator(imgs)                
                disc_input = self.get_disc_input(imgs, pred_centers)           
                dis_fake = self.discriminator(disc_input, dcm_labels)
                
                lossAdv_Discriminator = self.criteriaDISloss(dis_fake = dis_fake, dis_real = dis_real)                                                                         
                lossD = self.info['Discriminator_adv_loss'] * lossAdv_Discriminator
                
                D_epoch_loss += lossD.item()
                self.writer.add_scalar('Loss/train_ADV_D', lossAdv_Discriminator.item(), n_iter)                
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
                                
                if MSE_loss  < self.results['best_MSE']:
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
        
        
        
        

            
            
            
            
            
        
        
    
