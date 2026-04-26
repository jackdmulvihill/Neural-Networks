#!/usr/bin/env python
# coding: utf-8

# # VGGNet Implementation for Glaucoma (Finetuned from CIFAR 10 Training)

# ## Imported Libraries

# In[1]:


import os
import random
import numpy
import matplotlib.pyplot
import sklearn.model_selection
import sklearn.metrics

import torch
import torchvision
import collections

"""
Implementation based on: https://arxiv.org/pdf/1409.1556

See Table 1 Column D

Added batch normalization before each activation function
to address the exploding/vanishing gradient issue
"""

class VGG16D(torch.nn.Module):
    def __init__(self, class_count):
        super(VGG16D, self).__init__()
        
        # Block 1: 2 conv3-64 layers, normalize, then pool
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 2: 2 conv3-128 layers, normalize, then pool
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 3: 3 conv3-256 layers, normalize, then pool
        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 4: 3 conv3-512 layers, normalize, then pool
        self.block4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
            
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 5: 3 conv3-512 layers, normalize, then pool
        self.block5 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512 * 7 * 7, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, class_count)
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x