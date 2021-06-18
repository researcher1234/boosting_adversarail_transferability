import math, random, os, sys, argparse, csv, logging, shutil
from datetime import datetime
import numpy as np
import scipy.stats as st
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

def load_ground_truth(csv_filename):
    image_id_list = []
    label_ori_list = []
    label_tar_list = []

    with open(csv_filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            image_id_list.append( row['ImageId'] )
            label_ori_list.append( int(row['TrueLabel'])-1 )
            label_tar_list.append( int(row['TargetClass'])-1 )

    return image_id_list,label_ori_list,label_tar_list

# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
    def forward(self, x):
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]

##define TI
def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

##define DI
def DI(X_in):
    rnd=np.random.randint(299, 330,size=1)[0]
    h_rem = 330 - rnd
    w_rem = 330 - rnd
    pad_top = np.random.randint(0, h_rem,size=1)[0]
    pad_bottom = h_rem - pad_top
    pad_left = np.random.randint(0, w_rem,size=1)[0]
    pad_right = w_rem - pad_left

    c=np.random.rand(1)
    if c<=0.7:
        X_out=F.pad(F.interpolate(X_in, size=(rnd,rnd)),(pad_left,pad_top,pad_right,pad_bottom),mode='constant', value=0)
        return  X_out 
    else:
        return  X_in
    
def get_logger(path):
    logger = logging.getLogger('logbuch')
    logger.setLevel(level=logging.DEBUG)
    
    # Stream handler
    sh = logging.StreamHandler()
    sh.setLevel(level=logging.DEBUG)
    sh_formatter = logging.Formatter('%(message)s')
    sh.setFormatter(sh_formatter)
    
    # File handler
    fh = logging.FileHandler(os.path.join(path, "log.txt"))
    fh.setLevel(level=logging.DEBUG)
    fh_formatter = logging.Formatter('%(message)s')
    fh.setFormatter(fh_formatter)

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger

def get_timestamp():
    ISOTIMEFORMAT='%Y%m%d_%H%M%S_%f'
    timestamp = '{}'.format(datetime.utcnow().strftime( ISOTIMEFORMAT)[:-3])
    return timestamp

def one_hot(class_labels, num_classes):
    class_labels = class_labels.cpu()
    return torch.zeros(len(class_labels), num_classes).scatter_(1, class_labels.unsqueeze(1), 1.).cuda()