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
from lib.preprocessing import single_transforms
from lib.dataloading import *
from lib.loss_functions import *
from lib.evaluation import *
from lib.utilities.image_sampler import ImageSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torchvision import transforms
import torchvision.models as models
from torch.nn.functional import interpolate
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch import optim, nn

class ClassNetworkTrainer(object):

    def __init__(self,                  
                 opt = 'adam', 
                 network = 'dicom-resnet',
                 lr = 0.001,                 
                 batch_size = 4,
                 epochs = 10, 
                 regularizer = 0.0001,
                 experiment = 'TEST',                 
                 gpu = '0',
                 cluster = False,
                 ):
        
        ## Set the default information
        self.info = OrderedDict()
        self.set_info(opt, network, lr, batch_size, epochs, regularizer, experiment)
        self.set_default_info()
        self.device = torch.device("cuda:%s" % gpu if torch.cuda.is_available() else "cpu")                

        

        ## Define the base directory 
        if cluster:
            self.data_folder = os.environ['SLURM_JOB_SCRATCHDIR']
            self.output_dir = os.path.join(self.data_folder, 'results')
            if not os.path.exists(self.output_dir):
                os.mkdir(self.output_dir)

        else:
            self.data_folder = '/media/jimmy/224CCF8B4CCF57E5/Data'
            self.output_dir = '/mnt/Liver/GE_study_hri/ContextEncoder/results'
        
        self.data_path = os.path.join(self.data_folder, 'SSI')
        self.img_file = os.path.join(self.data_folder, 'SSI', 'ssi.csv')
    

        ## Output folder path 
        self.output_dir = os.path.join(self.output_dir, experiment)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            
        self.writer = SummaryWriter(log_dir = self.output_dir)
        
        ## Dataset              
        if self.info['experiment'] == 'TEST':
            self.all_dataset = self.load_dataset(list_id = range(101))
        else:
            self.all_dataset = self.load_dataset()        
        
        ## initialize results dictionary
        self.results = self.intialize_results_dict()
        
        # Network
        self.network = self.get_network(self.info['network'])
        self.optimizer = self.get_optimizer()                
        self.criterion = self.get_loss_fx('BCE')
        self.transform = self.get_transform()
        self.train_loader, self.val_loader = self.get_data_loader()
        
        print("Network initizliation:")
        print("Training Number: %s" % (len(self.train_loader) * self.info['batch_size']))
        print("Validation Number: %s" % (len(self.val_loader) * self.info['batch_size']))

        self.save_results(init = True)
        
    def set_info(self, opt, network, lr, batch_size, epochs , regularizer, experiment):        
        self.info['optimizer'] = opt
        self.info['network'] = network
        self.info['learning_rate'] = lr
        self.info['batch_size'] = batch_size
        self.info['epochs'] = epochs     
        self.info['regularization_weights'] = regularizer
        self.info['experiment'] = experiment
        
    def set_default_info(self):
        self.info['sample_interval'] = 10
        self.info['output_resize'] = False        
        self.info['n_dcm_labels'] = 9       
        
    def update_info(self, key, value):
        self.info[key] = value    
    
    def intialize_results_dict(self):
        results = OrderedDict()
        results['training_loss'] = []
        results['validation_loss'] = []

        results['validation_accuracy'] = []
        results['validation_precision'] = []
        results['validation_recall'] = []

        results['best_accuracy'] = 0
        results['best_precision'] = 0
        results['best_recall'] = 0
        return results

    
        
    def load_dataset(self, list_id = None, transform = None):
        return SSIDataset(img_file = self.img_file, rel_path = self.data_path, list_id = list_id, transform = transform, inpaint = False, classification = True) 
        
        
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
    
    def get_network(self, net = 'dicom-resnet'):                
        if net == 'dicom-resnet':
            net = DicomNet(backbone = 'resnet', n_classes = self.info['n_dcm_labels'])
        elif net == 'dicom-vggnet':
            net = DicomNet(backbone = 'vgg', n_classes = self.info['n_dcm_labels'])
        return net
    
    def get_transform(self):
        if self.info['network'] == 'resnet':
            return single_transforms.get_transformer_norm(resize = (256,384))
        else:
            return single_transforms.get_transformer_norm()
    
    
    def get_loss_fx(self, loss_fx):
        if loss_fx == 'MSE':
            return nn.MSELoss()
        elif loss_fx == 'BCE':
            return nn.BCELoss()        

    
    def evaluate(self, data_loader, threshold = 0.5):
        """Evaluation without the densecrf with the dice coefficient"""
        with torch.set_grad_enabled(False):
            self.network.eval()
            pred_labels = np.array([])
            true_labels = np.array([])
            tot_loss = 0
            for i, (imgs, labels) in enumerate(data_loader):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                pred = self.network(imgs)
                loss = self.criterion(pred.squeeze(dim=1), labels.float())
                tot_loss += loss.item()
                pred_labels = np.append(pred_labels, np.sum(pred.cpu().numpy() > threshold, axis = 1))
                true_labels = np.append(true_labels, np.sum(labels.cpu().numpy(), axis = 1))

            accuracy = accuracy_score(true_labels, pred_labels)

        return tot_loss/(i+1), accuracy
    
    
    def train(self):
        self.network = self.network.to(self.device)    
        self.criterion = self.criterion.to(self.device)
        epochs = self.info['epochs']
        for epoch in range(epochs):
            print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
            self.network.train()
            epoch_loss = 0
            for i, (imgs, labels) in enumerate(self.train_loader):
                #print('***',img.shape)
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                pred = self.network(imgs)
                loss = self.criterion(pred.squeeze(dim=1), labels.float())

                epoch_loss += loss.item()

                #print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))
                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()
                if (i+1) % round(len(self.train_loader)/10) == 0:
                    print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f ' %(epoch+1, epochs, i+1, len(self.train_loader), loss.item()))

            epoch_loss /= i+1
            self.results['training_loss'].append(epoch_loss)
            print('Epoch finished ! Loss: {}'.format(epoch_loss))

            # Validation
            val_loss, val_acc, val_precision, val_recall = self.evaluate(self.val_loader)

            self.results['validation_loss'].append(val_loss)
            self.results['validation_accuracy'].append(val_acc)
            self.results['validation_precision'].append(val_precision)            
            self.results['validation_recall'].append(val_recall)
            

            print('Validation Accuracy: {}'.format(val_acc))

            if val_acc > self.results['best_accuracy']:
                self.results['best_accuracy'] = val_acc
                self.results['best_epoch'] = epoch + 1
                torch.save(self.network.state_dict(), os.path.join(self.output_dir, "epoch{}.pth".format(epoch+1)))
                print("Best Validation accuracy improved!")

        # Save the training dice score using the best weights
        self.evaluate_train()     

        # Save the training dice score using the best weights
        #self.evaluate_train()

    def evaluate(self, data_loader, threshold = 0.5):
        """Evaluation without the densecrf with the dice coefficient"""
        with torch.set_grad_enabled(False):
            self.network.eval()
            pred_labels = np.array([])
            true_labels = np.array([])
            tot_loss = 0
            for i, (imgs, labels) in enumerate(data_loader):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                pred = self.network(imgs)
                pred = (pred > 0).float()
                loss = self.criterion(pred.squeeze(dim=1), labels.float())
                tot_loss += loss.item()
                pred_labels = np.append(pred_labels, pred.cpu().numpy())
                true_labels = np.append(true_labels, labels.cpu().numpy())
    
            accuracy = accuracy_score(true_labels, pred_labels)
            precision = precision_score(true_labels, pred_labels)
            recall = recall_score(true_labels, pred_labels)
            

        return tot_loss/(i+1), accuracy, precision, recall
    
        
    def evaluate_train(self):
        weight_path =  os.path.join(self.output_dir, "epoch{}.pth".format(self.results['best_epoch']))
        self.network.load_state_dict(torch.load(weight_path))

        with torch.set_grad_enabled(False):
            self.network.eval()
            train_loss, train_acc, train_precision, train_recall  = self.evaluate(self.train_loader)

            self.results['train_accuracy'] = train_acc
            self.results['train_precision'] = train_precision
            self.results['train_recall'] = train_recall
            
            
        
    def plot_training():
        pass
    def save_results(self, init = False):
        config = configparser.ConfigParser()
        config['INFO'] = self.info
        
        if not init:
            config['BEST RESULTS'] = {'val_accuracy': self.results['best_accuracy'],
                                 'val_precision': self.results['best_precision'],
                                 'val_recall': self.results['best_recall'],
                                  'best_epoch': self.results['best_epoch']}
            config['TRAIN RESULTS'] = {'train_accuracy': self.results['train_accuracy'],
                                 'train_precision': self.results['train_precision'],
                                 'train_recall': self.results['train_recall'],
                                }

        with open(os.path.join(self.output_dir, 'exp.ini'), 'w') as configfile:
            config.write(configfile)


        loss_history = pd.DataFrame({'training_loss': self.results['training_loss'],
                                    'validation_loss': self.results['validation_loss']})
        loss_history.to_csv(os.path.join(self.output_dir, 'loss_history.csv'))
        
    def load_weights(self, weight_path):
        self.network.load_state_dict(torch.load(weight_path))
        
        
        
        

            
            
            
            
            
        
        
    
