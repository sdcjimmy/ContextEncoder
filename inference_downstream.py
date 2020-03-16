import sys
sys.path.append('../../')
import os
import glob
import pandas as pd
import torch
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, RandomSampler

from model import *
from lib.preprocessing import *
from lib.dataloading import *
from lib.loss_functions import *
from lib.evaluation import *
from torchvision import transforms
import torchvision.models as models
from torch import optim, nn
from sklearn.metrics import accuracy_score
import configparser



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', type = str, help = "The name of the experiement", default = 'TEST')
    parser.add_argument('-m', '--model', type = str, choices = ['vgg-ce-unet', 'vgg-unet'], help = "The name of the model", default = 'vgg-ce-unet')    
    parser.add_argument('-b', '--bootstrap', action = 'store_true')    
    parser.add_argument('-g', '--gpu', type = str, help = "The gpu used", default = '0')        
    parser.add_argument('-t', '--task', type = str,  choices = ['hri', 'nerve', 'quality', 'thyroid'], help = "The downstream task", default = 'hri')
    
    
    return parser.parse_args()


def get_weight_paths(root):

    print(os.listdir(root))
    all_exp = [exp for exp in os.listdir(root) if os.path.isdir(os.path.join(root,exp)) and 'exp.ini' in os.listdir(os.path.join(root, exp)) and exp != 'TEST']
    print(f'Target experiements: {all_exp}')
    weight_paths = []
    for exp in all_exp:
        config = configparser.ConfigParser()
        try:
            config.read(os.path.join(root, exp, 'exp.ini'))
            epoch = config['BEST RESULTS']['best_epoch']
        except:
            continue
        weight_path = os.path.join(root, exp, f'epoch{epoch}.pth')
        weight_paths.append(weight_path)

    for weight_path in weight_paths:
        print(f'valid weight path: {weight_path}')

    return weight_paths




if __name__ == '__main__':
    args = get_args()
    
    
    if args.task == 'hri':
        img_dir = '/mnt/Liver/GE_study_hri/01_Rawdata_for_annotation/Collections_sw/img_test'        
        liver_dir = '/mnt/Liver/GE_study_hri/01_Rawdata_for_annotation/Collections_sw/mask_liver_test_Xiaohong'
        kidney_dir = '/mnt/Liver/GE_study_hri/01_Rawdata_for_annotation/Collections_sw/mask_kidney_test_Xiaohong'
        
         
        #weight_path = '/mnt/Liver/GE_study_hri/ContextEncoder/results/downstream/hri/dt-hri-nop-d-best-freeze-100/epoch184.pth'
        weight_paths = get_weight_paths('/mnt/Liver/GE_study_hri/ContextEncoder/results/downstream/hri/')
        if args.bootstrap:
            for weight_path in weight_paths:
                predictor = HRINetworkPredicter(task = 'hri', img_dir = img_dir, liver_mask_dir = liver_dir, kidney_mask_dir = kidney_dir, network = args.model, weight_path = weight_path, gpu = args.gpu, save = False, bootstrap_n = 10)
                predictor.bootstrap()
        else:
            pass
            
    if args.task == 'thyroid':
        img_dir = '/mnt/DL_swRESTORED/Data/thyroid-ultrasound/Image-slc-train/'
        mask_dir = '/mnt/DL_swRESTORED/Data/thyroid-ultrasound/Mask-slc-train/'
        
        weight_paths = get_weight_paths('/mnt/Liver/GE_study_hri/ContextEncoder/results/downstream/thyroid/')
        if args.bootstrap:
            for weight_path in weight_paths:
                predictor = SingleNetworkPredicter(task = 'thyroid', img_dir = img_dir, mask_dir = mask_dir, network = args.model, weight_path = weight_path, gpu = args.gpu, save = False, bootstrap_n = 10, ext = '*.png')
                predictor.bootstrap()
        else:
            pass
        





"""
    all_exp = [exp for exp in os.listdir('./') if 'scoring' in exp]

    transform = get_transformer_norm()['val']
    dataset = HRIViewScoringDataset(img_file='../../data/enc_info_test.csv', transform = transform)



    all_accuracy = []
    all_exp_list = []

    for exp in all_exp:    
        config = configparser.ConfigParser()
        try:
            config.read(os.path.join('./', exp, 'exp.ini'))
            epoch = config['BEST RESULTS']['best_epoch']
            weight_path = os.path.join('./', exp, f'epoch{epoch}.pth')
        except:
            continue

        if '01' not in exp:
            continue

        print(f'prediciting experiment {exp}')        
        net = VGG16Scoring()
        device = 'cuda:0'
        net.to(device)

        net.load_state_dict(torch.load(weight_path))            
        net.eval()

        criterion = nn.BCELoss()

        # Bootstrap for 10 times
        for boot in range(10):
            sampler = RandomSampler(dataset, replacement=True, num_samples= len(dataset))
            test_loader = DataLoader(dataset, batch_size=4, sampler = sampler)


            with torch.set_grad_enabled(False):   
                net.eval()
                pred_labels = np.array([])    
                true_labels = np.array([])
                tot_loss = 0
                for i, (imgs, labels) in enumerate(test_loader):               

                    imgs, labels = imgs.to(device), labels.to(device)              
                    pred = net(imgs)                
                    loss = criterion(pred.squeeze(dim=1), labels.float())
                    tot_loss += loss.item()
                    pred_labels = np.append(pred_labels, np.sum(pred.cpu().numpy() > 0.5, axis = 1))
                    true_labels = np.append(true_labels, np.sum(labels.cpu().numpy(), axis = 1))

                accuracy = accuracy_score(true_labels, pred_labels)    
                print(f'exp: {exp}, bootsrap: #{boot+1} -- accuracy: {accuracy}')
                all_accuracy.append(accuracy)
                all_exp_list.append(exp)

    res = pd.DataFrame({'exp': all_exp_list, 'accuracy': all_accuracy})
    res = res.sort_values("accuracy", ascending = False)
    print(res)
    res.to_csv('./results_summary_bootstrap_01.csv', index = None)

"""
