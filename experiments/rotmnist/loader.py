import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler, SubsetRandomSampler

import torchvision
import torchvision.transforms as transforms

from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt

import os
import sys
import time
import numpy as np
import random, math
from datetime import timedelta

from texnist import TexNIST

def load_data(folder, mean=None, std=None, inp_size=47, fraction=1.0, shift=False):
    if mean is None or std is None:
        R = []
        G = []
        B = []

        for c in range(10):
            print('Starting on class {0}'.format(c + 1))
            for filename in os.listdir(folder + str(c) + "/"):
                if filename.endswith(".PNG"):
                    file = folder + str(c) + '/' + filename
                    img = cv.imread(file)
                    R.append(np.mean(img[:, :, 0]))
                    G.append(np.mean(img[:, :, 1]))
                    B.append(np.mean(img[:, :, 2]))

        R = np.multiply(R, 1/255)
        G = np.multiply(G, 1/255)
        B = np.multiply(B, 1/255)

        R_avg = np.mean(R); R_std = np.std(R)
        G_avg = np.mean(G); G_std = np.std(G)
        B_avg = np.mean(B); B_std = np.std(B)
        
        del(R); del(G); del(B)

        mean = [R_avg, G_avg, B_avg]
        std = [R_std, G_std, B_std]
        print('mean', mean)
        print('std', std)

    data_transform = transforms.Compose([
            transforms.Resize((inp_size, inp_size)),
            transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ])
    if shift:
        data_transform = transforms.Compose([data_transform,
                                             transforms.RandomCrop((inp_size, inp_size),
                                                                   padding=1, padding_mode='edge')])
    return TexNIST(root=folder, transform=data_transform, fraction=fraction)

def load(dataset, inp_size, fraction=1.0, shift=False):
    sets = ['RotMNIST_nonrot',
            'RotMNIST_nonrot_validation',
            'RotMNIST_scaled',
            'RotMNIST_scaled_validation']
    means_stds = {'RotMNIST_nonrot': ([0.13083715981299926, 0.13083715981299926, 0.13083715981299926], [0.043476788322023646, 0.043476788322023646, 0.043476788322023646]),
                  'RotMNIST_nonrot_validation': ([0.13083715981299926, 0.13083715981299926, 0.13083715981299926], [0.043476788322023646, 0.043476788322023646, 0.043476788322023646]),
                  'RotMNIST_scaled': ([0.13083715981299926, 0.13083715981299926, 0.13083715981299926], [0.043476788322023646, 0.043476788322023646, 0.043476788322023646]),
                  'RotMNIST_scaled_validation': ([0.13083715981299926, 0.13083715981299926, 0.13083715981299926], [0.043476788322023646, 0.043476788322023646, 0.043476788322023646])}
    if dataset not in sets:
        print('Dataset not found...')
        return None
    return load_data('./data/{0}/'.format(dataset), mean=means_stds[dataset][0], std=means_stds[dataset][1],
                     inp_size=inp_size, fraction=fraction, shift=shift)