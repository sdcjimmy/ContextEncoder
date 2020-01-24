import sys
import os
import glob
import pandas as pd
import argparse
import configparser
import numpy as np
import torch

from model import *
from lib.training import *
from lib.preprocessing import *
from lib.dataloading import *
from lib.loss_functions import *
from lib.evaluation import *
from torchvision import transforms
from torch import optim, nn
#from torch.utils.tensorboard import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', type = str, help = "The name of the experiement", default = 'TEST_sw')
    parser.add_argument('-m', '--model', type = str, choices = ['ce-net', 'vgg-unet'], help = "The name of the model", default = 'ce-net')
    parser.add_argument('-n', '--epochs', type = int, help = "The number of epochs", default = 10)
    parser.add_argument('-b', '--batch_size', type = int, help = "The batch size", default = 4)
    parser.add_argument('-l', '--learning_rate', type = float, help = "The learning rate", default = 0.02)
    parser.add_argument('-s', '--validation_split', type = float,  help = "The validation split percentage", default = 0.2)
    parser.add_argument('-o', '--optimizer', type = str,  help = "The optimizer", default = 'adam')
    parser.add_argument('-r', '--regularizer', type = float,  help = "The coefficient for regularizer(weight decay)", default = 0.001)
    parser.add_argument('-g', '--gpu', type = str,  choices = ['0', '1'], help = "The gpu used", default = '0')
    parser.add_argument('-cd', '--center_distribution', type = str, help = "The inital distribution for the center image", default = 'uniform')
    parser.add_argument('-d', '--dcm_loss', action = 'store_false', default = True) 
    parser.add_argument('-p', '--padding_center', action = 'store_true', default = False) 


    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
        
    trainer = NetworkTrainer(opt = args.optimizer,
                             network = args.model, 
                            lr = args.learning_rate,
                            batch_size = args.batch_size,
                            epochs = args.epochs,
                            dcm_loss = args.dcm_loss,
                            padding_center = args.padding_center,
                            center_distribution = args.center_distribution,
                            experiment = args.experiment,
                            gpu = args.gpu)
    
    try:
        trainer.train()
        trainer.save_results()
    except KeyboardInterrupt:
        print('Keyboard interrupted! Saving results ...l')
        trainer.save_results()

    
    
                
                
