from __future__ import print_function, division
import os
import torch
import pandas as pd
#from skimage import io, transform
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
import math
import os 
import imgaug.augmenters as iaa

import argparse

from pdb import set_trace 




# data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
seq = iaa.Sequential([
    iaa.Add(value=(-40,40), per_channel=True), # Add color 
    iaa.GammaContrast(gamma=(0.5,1.5)) # GammaContrast with a gamma of 0.5 to 1.5
])





# array
class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.01, sh = 0.05, r1 = 0.5, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, sample):
        img, spoofing_label = sample['image_x'], sample['spoofing_label']
        
        if random.uniform(0, 1) < self.probability:
            attempts = np.random.randint(1, 3)
            for attempt in range(attempts):
                area = img.shape[0] * img.shape[1]
           
                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1/self.r1)
    
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
    
                if w < img.shape[1] and h < img.shape[0]:
                    x1 = random.randint(0, img.shape[0] - h)
                    y1 = random.randint(0, img.shape[1] - w)

                    img[x1:x1+h, y1:y1+w, 0] = self.mean[0]
                    img[x1:x1+h, y1:y1+w, 1] = self.mean[1]
                    img[x1:x1+h, y1:y1+w, 2] = self.mean[2]
                    
        return {'image_x': img, 'spoofing_label': spoofing_label}


# Tensor
class Cutout(object):
    def __init__(self, length=50):
        self.length = length

    def __call__(self, sample):
        img,  spoofing_label = sample['image_x'], sample['spoofing_label']
        h, w = img.shape[1], img.shape[2]    # Tensor [1][2],  nparray [0][1]
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        length_new = np.random.randint(1, self.length)
        
        y1 = np.clip(y - length_new // 2, 0, h)
        y2 = np.clip(y + length_new // 2, 0, h)
        x1 = np.clip(x - length_new // 2, 0, w)
        x2 = np.clip(x + length_new // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        
        return {'image_x': img, 'spoofing_label': spoofing_label}


class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    # def __init__(self, mean, std):

    def __call__(self, sample):
        image_x, spoofing_label = sample['image_x'], sample['spoofing_label']
        
        new_image_x = (image_x - 127.5)/128     # [-1,1]
    
        return {'image_x': new_image_x, 'spoofing_label': spoofing_label}



class RandomHorizontalFlip(object):
    """Horizontally flip the given Image randomly with a probability of 0.5."""
    def __call__(self, sample):
        image_x,  spoofing_label = sample['image_x'], sample['spoofing_label']
        
        new_image_x = np.zeros((256, 256, 3))
        

        p = random.random()
        if p < 0.5:
            #print('Flip')

            new_image_x = cv2.flip(image_x, 1)       
                
            return {'image_x': new_image_x, 'spoofing_label': spoofing_label}
        else:
            #print('no Flip')
            return {'image_x': image_x, 'spoofing_label': spoofing_label}

class Resize(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """
    def __init__(self, size=256):
        self.size = size

    def __call__(self, sample):
        image_x, spoofing_label = sample['image_x'],sample['spoofing_label']
        # set_trace()
        image_x = cv2.resize(image_x, (self.size, self.size))
        

        return {'image_x': image_x, 'spoofing_label': spoofing_label}

class CenterCrop(object):
    def __init__(self, size=224):
        self.size=size

    def __call__(self, sample):
        image_x, spoofing_label = sample['image_x'], sample['spoofing_label']
        # set_trace()

        image_width, image_height, _ = image_x.shape
        crop_height, crop_width = self.size, self.size
        
        start_width = np.random.randint(0, high=image_width-crop_width)
        end_width = start_width + self.size

        start_height = np.random.randint(0, high=image_height-crop_height)
        end_height = start_height + self.size

        image_x = image_x[start_height: end_height, start_width: end_width, :]

        # set_trace()
        return {'image_x': image_x, 'spoofing_label': spoofing_label}


class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        image_x, spoofing_label = sample['image_x'], sample['spoofing_label']
        
        # swap color axis because
        # numpy image: (batch_size) x H x W x C
        # torch image: (batch_size) x C X H X W
        image_x = image_x[:,:,::-1].transpose((2, 0, 1))
        image_x = np.array(image_x)
                        
        spoofing_label_np = np.array([0],dtype=np.long)
        spoofing_label_np[0] = spoofing_label
        
        
        return {'image_x': torch.from_numpy(image_x.astype(np.float)).float(), 'spoofing_label': torch.from_numpy(spoofing_label_np.astype(np.float)).float()}



class oulu_dataset_train(Dataset):

    def __init__(self, info_list, root_dir,  transform=None):

        self.landmarks_frame = pd.read_csv(info_list, delimiter=',', header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    
    def __getitem__(self, idx):
        #print(self.landmarks_frame.iloc[idx, 0])
        videoname = str(self.landmarks_frame.iloc[idx, 1])
        image_path = os.path.join(self.root_dir, "Train_files")
        image_path = os.path.join(image_path, videoname)
        image_path = image_path + ".avi"
        
        image_x = self.get_single_image_x(image_path)
            
        spoofing_label = self.landmarks_frame.iloc[idx, 0]
        
        sample = {'image_x': image_x, 'spoofing_label': spoofing_label}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_single_image_x(self, image_path):
        
        
        image_x = np.zeros((256, 256, 3))
 
        cap = cv2.VideoCapture(image_path)
        
        
        while (cap.isOpened()):

            ret, frame = cap.read()

            if ret:
                image_x_temp = frame
                break

        #set_trace()
        image_x = cv2.resize(image_x_temp, (256, 256))
        
        image_x_aug = seq.augment_image(image_x) 
   
        return image_x_aug




