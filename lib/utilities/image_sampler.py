import torch
import numpy as np
from torchvision.utils import make_grid

class ImageSampler(object):
    def __init__(self, max_images = 16, sample_p = 0.05):
        self.n_images = 0
        self.max_images = max_images
        self.image_tensors = []
        self.pred_tensors = []
        self.sample_p = sample_p
        
    def sample(self, imgs, centers, pred_centers):
        n_batchs = imgs.size()[0]
        for i in range(n_batchs):
            if np.random.uniform(0,1) < self.sample_p and self.n_images < self.max_images:
                true = self.padding_center(imgs[i:i+1, :, : ,:], centers[i:i+1, :, : ,:])
                pred = self.padding_center(imgs[i:i+1, :, : ,:], pred_centers[i:i+1, :, : ,:])
                
                self.image_tensors.append(true)
                self.pred_tensors.append(pred)                
                self.n_images += 1                                
    def make_grid(self):
        true = torch.cat(self.image_tensors)
        true = make_grid(true, normalize= True, scale_each = True)
        
        pred = torch.cat(self.pred_tensors)
        pred = make_grid(pred, normalize= True, scale_each = True)
        return true, pred
        
    def padding_center(self, imgs, centers):
        b,c,w,h = imgs.size()
        new_img = imgs.clone()
        new_img[:,:,w//4:3*w//4, h//4:3*h//4] = centers
        return new_img     
