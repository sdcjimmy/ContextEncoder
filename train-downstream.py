import sys
import os
import glob
import pandas as pd
import argparse
import configparser
import numpy as np
import torch

from model import *
from lib.training.trainer_seg import *
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
    parser.add_argument('-n', '--epochs', type = int, help = "The number of epochs", default = 10)
    parser.add_argument('-b', '--batch_size', type = int, help = "The batch size", default = 4)
    parser.add_argument('-l', '--learning_rate', type = float, help = "The learning rate", default = 0.02)
    parser.add_argument('-ls', '--loss', type = str, help = "The loss function", default = "BCE")
    parser.add_argument('-s', '--validation_split', type = float,  help = "The validation split percentage", default = 0.2)
    parser.add_argument('-o', '--optimizer', type = str,  help = "The optimizer", default = 'adam')
    parser.add_argument('-r', '--regularizer', type = float,  help = "The coefficient for regularizer(weight decay)", default = 0.001)
    parser.add_argument('-p', '--pretrained', action = 'store_true', default = False)
    parser.add_argument('-f', '--freeze', action = 'store_true', default = False)
    parser.add_argument('-pd', '--pretrain_dict', type = str,  help = "The path to the pre-train model", default = '')
    parser.add_argument('-g', '--gpu', type = str,  choices = ['0', '1'], help = "The gpu used", default = '0')
    parser.add_argument('-m', '--model', type = str,  choices = ['unet', 'unet-light', 'vggnet', 'res-unet','r2u-unet', 'vgg-ce-unet'], help = "The model used", default = 'vgg-ce-unet')
    parser.add_argument('-sk', '--shrink', type = float, help = "shrink size", default = 1.0)
    parser.add_argument('-t', '--task', type = str,  choices = ['hri', 'nerve', 'quality'], help = "The downstream task", default = 'hri')



    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    if args.task == 'hri':
        img_dir = '/mnt/Liver/GE_study_hri/01_Rawdata_for_annotation/Collections_sw/img/'
        kidney_mask_dir = '/mnt/Liver/GE_study_hri/01_Rawdata_for_annotation/Collections_sw/mask_kidney/'
        liver_mask_dir = '/mnt/Liver/GE_study_hri/01_Rawdata_for_annotation/Collections_sw/mask_liver/'
        
        trainer = HRISegNetworkTrainer(img_dir = img_dir, kidney_mask_dir = kidney_mask_dir, liver_mask_dir = liver_mask_dir,
                            network = args.model,
                            opt = args.optimizer,
                            lr = args.learning_rate,
                            reg = args.regularizer,
                            loss_fx = args.loss,
                            batch_size = args.batch_size,
                            epochs = args.epochs,
                            pretrain = args.pretrained,
                            self_train = args.pretrain_dict,
                            freeze = args.freeze,
                            shrink = args.shrink,
                            experiment = args.experiment,
                            gpu = args.gpu)
        
    elif args.task == 'nerve':
        print('Training the ultrasound nerve segmentation')
        img_dir = '/mnt/DL_swRESTORED/Data/ultrasound-nerve-segmentation/train_ori/'
        mask_dir = '/mnt/DL_swRESTORED/Data/ultrasound-nerve-segmentation/train_mask/'
        trainer = NerveSegNetworkTrainer(img_dir = img_dir, mask_dir = mask_dir,
                            network = args.model,
                            opt = args.optimizer,
                            lr = args.learning_rate,
                            reg = args.regularizer,
                            loss_fx = args.loss,
                            batch_size = args.batch_size,
                            epochs = args.epochs,
                            pretrain = args.pretrained,
                            self_train = args.pretrain_dict,
                            freeze = args.freeze,
                            shrink = args.shrink,
                            experiment = args.experiment,
                            gpu = args.gpu)
        
        
    
    try:
        trainer.train()
        trainer.save_results()
    except KeyboardInterrupt:
        print('Keyboard interrupted! Saving results ...l')
        trainer.save_results()

    
    
                
                
