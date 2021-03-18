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

from pdb import set_trace


frames_total = 8    # each video 8 uniform samples



class Resize_val(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """
    def __init__(self, size=256):
        self.size = size

    def __call__(self, sample):
        image_x,  spoofing_label = sample['image_x'], sample['spoofing_label']


        # set_trace()
        image_x = cv2.resize(image_x, (self.size, self.size))
    

        return {'image_x': image_x, 'spoofing_label': spoofing_label}

class CenterCrop_val(object):
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

class Normaliztion_valtest(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        image_x = sample['image_x']
        spoofing_label = sample['spoofing_label']

        new_image_x = (image_x - 127.5)/128     # [-1,1]
        
        return {'image_x': new_image_x,'spoofing_label': spoofing_label}


class ToTensor_valtest(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        image_x = sample['image_x']
        
        spoofing_label = sample['spoofing_label']
        # swap color axis because    BGR2RGB
        # numpy image: (batch_size) x T x H x W x C
        # torch image: (batch_size) x T x C X H X W
        image_x = image_x[:,:,::-1].transpose((2, 0, 1))
        image_x = np.array(image_x)
         
        return {'image_x': torch.from_numpy(image_x.astype(np.float)).float(), 'spoofing_label': spoofing_label} 



class casia_fasd_dataset_val(Dataset):

    def __init__(self, info_list, root_dir,  transform=None):

        self.landmarks_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    
    def __getitem__(self, idx):

        videoname = str(self.landmarks_frame.iloc[idx, 0])
        image_path = os.path.join(self.root_dir, videoname)
             
        image_x = self.get_single_image_x(image_path, videoname)
		
        # set_trace()
        spoofing_label = self.landmarks_frame.iloc[idx, 1]
        if spoofing_label == 1:
            spoofing_label = 1            # real
        else:
            spoofing_label = 0            # fake
            binary_mask = np.zeros((32, 32))    
        
        
        #frequency_label = self.landmarks_frame.iloc[idx, 2:2+50].values  
            
        sample = {'image_x': image_x,'spoofing_label': spoofing_label}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_single_image_x(self, image_path, videoname):

        image_x = np.zeros((frames_total, 256, 256, 3))
    

        cap = cv2.VideoCapture(image_path)
        
        
        while (cap.isOpened()):

            ret, frame = cap.read()

            if ret:
                image_x_temp = frame
                break

                       
        image_x = cv2.resize(image_x_temp, (256, 256))
        
        return image_x







if __name__ == '__main__':
    # usage
    # MAHNOB
    root_list = '/wrk/yuzitong/DONOTREMOVE/BioVid_Pain/data/cropped_frm/'
    trainval_list = '/wrk/yuzitong/DONOTREMOVE/BioVid_Pain/data/ImageSet_5fold/trainval_zitong_fold1.txt'
    

    BioVid_train = BioVid(trainval_list, root_list, transform=transforms.Compose([Normaliztion(), Rescale((133,108)),RandomCrop((125,100)),RandomHorizontalFlip(),  ToTensor()]))
    
    dataloader = DataLoader(BioVid_train, batch_size=1, shuffle=True, num_workers=8)
    
    # print first batch for evaluation
    for i_batch, sample_batched in enumerate(dataloader):
        #print(i_batch, sample_batched['image_x'].size(), sample_batched['video_label'].size())
        print(i_batch, sample_batched['image_x'], sample_batched['pain_label'], sample_batched['ecg'])
        pdb.set_trace()
        break

            
 


