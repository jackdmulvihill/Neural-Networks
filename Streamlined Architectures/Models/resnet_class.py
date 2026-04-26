#!/usr/bin/env python
# coding: utf-8

# # ResNet Implementation for Glaucoma (Finetuned from CIFAR-10 Training)

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
Implementation based on: https://arxiv.org/pdf/1512.03385

See Table 1 18-layer Column
"""

class BasicBlock(torch.nn.Module):
    
    # Processes data and adds the skip connection
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        # 3 x 3 convolution that processes input
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                                      stride=stride, padding=1, bias=False)
        
        # Batch normalization and activation function
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        
        # 3 x 3 convolution that processes input
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                      stride=1, padding=1, bias=False)
        
        # Batch normalization
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        
        # Saves the downsample (reduce image quality) layer if provided to resize the skip connection
        self.downsample = downsample
    
    def forward(self, x):
        
        # Saves the original input
        identity = x
        
        # First conv layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second conv layer
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection
        # Adjust shortcut if needed
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Adds the identity shortcut, original input (of the last layer) to the processed output
        out += identity
        out = self.relu(out)
        
        return out


class ResNet18(torch.nn.Module):
    def __init__(self, class_count):
        super(ResNet18, self).__init__()
        
        # Initial convolution layer
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Layer 1: 2 residual blocks, 64 channels
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        
        # Layer 2: 2 residual blocks, 128 channels
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        
        # Layer 3: 2 residual blocks, 256 channels
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        
        # Layer 4: 2 residual blocks, 512 channels
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        
        # Global average pooling and fully connected layer
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(512, class_count)
    
    # Creates a sequence of BasicBlocks
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        
        # Create a layer with multiple residual blocks
        downsample = None
        
        # If dimensions change, need to downsample the skip connection
        # Use a 1 x 1 convolution to resize the shortcut
        if stride != 1 or in_channels != out_channels:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                               stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        
        # First block may need to downsample
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        
        # Remaining blocks keep the same dimensions
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        
        return torch.nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Fully connected layer
        x = self.fc(x)
        
        return x