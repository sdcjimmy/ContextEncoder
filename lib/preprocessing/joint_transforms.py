from PIL import Image, ImageOps
from torchvision.transforms import functional as F
import numpy as np
import random

"""
The code was adapted from
https://github.com/zijundeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py
"""

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, liver_mask, kidney_mask):
        # assert img.size == mask.size, ("image size: %s, mask size: %s" % (img.size, mask.size))
        for t in self.transforms:
            img, liver_mask, kidney_mask = t(img, liver_mask, kidney_mask)
        return img, liver_mask, kidney_mask


class RandomHorizontallyFlip(object):
    def __call__(self, img, liver_mask, kidney_mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), liver_mask.transpose(Image.FLIP_LEFT_RIGHT), kidney_mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, liver_mask, kidney_mask

class Equalization(object):
    def __call__(self, img, liver_mask, kidney_mask):
        return ImageOps.equalize(img), liver_mask, kidney_mask

    
class GaussianNoise(object):
    def __call__(self, img, liver_mask, kidney_mask):
        return img + torch.randn_like(img), liver_mask, kidney_mask  
    
class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, liver_mask, kidney_mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), liver_mask.rotate(rotate_degree, Image.NEAREST),kidney_mask.rotate(rotate_degree, Image.NEAREST)
        '''
        if random.random() < 0.8:
            rotate_degree = random.random() * 2 * self.degree - self.degree
            return img.rotate(rotate_degree, Image.BILINEAR), liver_mask.rotate(rotate_degree, Image.NEAREST),kidney_mask.rotate(rotate_degree, Image.NEAREST)
        return img, liver_mask, kidney_mask
        '''
class RandomLightVar(object):
    def __call__(self, img, liver_mask, kidney_mask):
        return (img+random.random()*64-32).astype('uint8'), liver_mask, kidney_mask

class RandomLightRevert(object):
    def __call__(self, img, liver_mask, kidney_mask):
        if random.random() < 0.5:
            return 255-img, liver_mask, kidney_mask
        else:
            return img, liver_mask, kidney_mask
        
    
class Normalization(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace
    def __call__(self, img, liver_mask, kidney_mask):
        return F.normalize(img, self.mean, self.std, self.inplace), liver_mask, kidney_mask
        
class Resize(object):
    def __init__(self, size, interpolation = Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
    def __call__(self, img, liver_mask, kidney_mask):
        return F.resize(img, self.size, self.interpolation), F.resize(liver_mask, self.size, self.interpolation), F.resize(kidney_mask, self.size, self.interpolation)
 
    
class ToTensor(object):
    def __call__(self, img, liver_mask, kidney_mask):
        return F.to_tensor(img), F.to_tensor(liver_mask), F.to_tensor(kidney_mask)

class ToPILImage(object):
    def __call__(self, img, liver_mask, kidney_mask):
        return F.to_pil_image(img), F.to_pil_image(liver_mask), F.to_pil_image(kidney_mask)


def get_horizontal_transformer(normalize = True):
    transform = {
    'train': Compose([
        ToPILImage(),
        GaussianNoise,
        Equalization(),
        Resize((192,320)),
        RandomHorizontallyFlip(),
        ToTensor(),
        Normalization([0.367],[0.384])
    ]),
    'val': Compose([
        ToPILImage(),
        Equalization(),
        Resize((192,320)),
        ToTensor(),
        Normalization([0.367],[0.384])
    ])
    }
    return transform


def get_transformer(normalize = True):
    transform = {
    'train': Compose([
        RandomLightVar(),
        ToPILImage(),
        RandomHorizontallyFlip(),
        RandomRotate(45),
        #Resize((192,320)),
        Resize((256,400)),
        Equalization(),
        ToTensor(),
        Normalization([0.367],[0.384])
    ]),
    'val': Compose([
        ToPILImage(),
        #Resize((192,320)),
        Resize((256,400)),
        Equalization(),
        ToTensor(),
        Normalization([0.367],[0.384])
    ])
    }
    return transform


def get_nonorm_transformer():
    return {'val':Compose([ToPILImage(),
                    Resize((192,320)),
                    ToTensor()])}

def get_norm_transformer():
    return {'val': Compose([
            ToPILImage(),
            #Resize((192,320)),
            Resize((256,400)),
            Equalization(),
            ToTensor(),
            Normalization([0.367],[0.384])
            ])
           }

'''
def get_norm_transformer():
    return Compose([ToPILImage(),
                    Equalization(),
                    Resize((192,320)),
                    ToTensor()])
'''
