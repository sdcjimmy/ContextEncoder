import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from .dice_score import dice



def eval_net(network, data_loader, criterion, device, threshold = 0.5):
    """Evaluation the model performance
    """
    network.eval()
    pred_labels = np.array([])    
    true_labels = np.array([])
    tot_loss = 0
    for i, (imgs, labels) in enumerate(data_loader):               
        imgs, labels = imgs.to(device), labels.to(device)              
        pred = network(imgs)                
        loss = criterion(pred.squeeze(dim=1), labels.float())
        tot_loss += loss.item()
        pred_labels = np.append(pred_labels, pred.cpu().numpy() > threshold)
        true_labels = np.append(true_labels, labels.cpu().numpy())
        
    
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
        
    return tot_loss/(i+1), accuracy, precision, recall


def eval_net_mclass(net, data_loader, device, loss_fx):
    """"deprecated function, can be ignored"""
    net.eval()
    tot_liver = 0
    tot_kidney = 0
    for i, (img, mask, low_ref, deep_ref) in enumerate(data_loader):
    #for i, (img, mask) in enumerate(data_loader):
        
        # Transfer to GPU
        imgs, masks = img.to(device), mask.to(device)
        
        low_refs = [e.to(device) for e in low_ref]
        deep_refs = [e.to(device) for e in deep_ref]
        # Model computations        
        val_masks_pred = net(x=imgs, low_ref=low_refs, deep_ref=deep_refs)
        
        #val_masks_pred = net(x=imgs)

        val_true_masks = masks


       
        val_masks_pred = (val_masks_pred > 0.5).float()
        tot_liver += dice_coeff(val_masks_pred[:,0,:,:], val_true_masks[:,0,:,:], device).item()
        tot_kidney += dice_coeff(val_masks_pred[:,1,:,:], val_true_masks[:,1,:,:], device).item()
    
    return  tot_liver/(i+1), tot_kidney/(i+1)
