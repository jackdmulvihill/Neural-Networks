#!/usr/bin/env python
# coding: utf-8

# # Xception Implementation and Training from Scratch with CIFAR 10

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
Implementation based on: https://arxiv.org/pdf/1610.02357

See Figure 5
"""

# depth-wise (or channel-wise) separable convolution
class DepthWiseSepConv2d(torch.nn.Module):
    
    # Creates a sequence of depth-wise convolutions
    def __init__(self, in_filter_count, out_filter_count):
        super().__init__()
        
        # Performs a 3 x 3 depth-wise convolution
        self.depthwise = torch.nn.Conv2d(in_filter_count, in_filter_count, \
                                         kernel_size=3, stride=1, padding=1, groups=in_filter_count)
        
        # Performs the point-wise (1 x 1) convolution on each output after the depth-wise convolution
        self.pointwise = torch.nn.Conv2d(in_filter_count, out_filter_count, \
                                         kernel_size=1, stride=1, padding=0, groups=1)
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
# Xception block
class XceptionBlock(torch.nn.Module):
    def __init__(self, in_filter_count, out_filter_count, dwsconv_repeat_count=1):
        super().__init__()
        
        # depth-wise separated convolution block, with "dwsconv_repeat_count" repetitions
        layers = []
        layers.append(DepthWiseSepConv2d(in_filter_count, out_filter_count))
        layers.append(torch.nn.BatchNorm2d(out_filter_count))
        layers.append(torch.nn.ReLU(inplace=True))
        
        # The depth-wise convolution will always be repeated at least once
        # Repeat as necessary from the current convolutional layer i
        for i in range(dwsconv_repeat_count):
            layers.append(DepthWiseSepConv2d(out_filter_count, out_filter_count))
            layers.append(torch.nn.BatchNorm2d(out_filter_count))
            layers.append(torch.nn.ReLU(inplace=True))
            
        self.block = torch.nn.Sequential(*layers) # this is the block
        
        # input downsampling and max-pooling application if "out_filter_count > in_filter_count"
        self.max_pool = None
        self.input_downsample = None
        
        # Determines if the skip connection needs to rescale the image
        # (i.e. some of the skip connections perform 1 x 1 convolutions to upsample)
        if out_filter_count > in_filter_count:
            self.max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.input_downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_filter_count, out_filter_count, kernel_size=1, stride=2),
                torch.nn.BatchNorm2d(out_filter_count),
            )
            
    def forward(self, x):
        output = self.block(x)
        if self.max_pool is not None:
            output = self.max_pool(output)
        if self.input_downsample is not None:
            x = self.input_downsample(x)
        return output + x # "+ x": skip connection
    
class Xception(torch.nn.Module):
    def __init__(self, class_count):
        super(Xception, self).__init__()

        # stem
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),

            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )

        # entry flow
        self.entry = torch.nn.Sequential(
            
            # paper original: XceptionBlock(256, 728)
            XceptionBlock(64, 128),
            XceptionBlock(128, 256),
            XceptionBlock(256, 512) 
        )

        # middle flow
        self.middle = torch.nn.Sequential(
            
            # paper original: XceptionBlock(728, 728) x8
            XceptionBlock(512, 512, dwsconv_repeat_count=2), 
            XceptionBlock(512, 512, dwsconv_repeat_count=2),
            XceptionBlock(512, 512, dwsconv_repeat_count=2),
            XceptionBlock(512, 512, dwsconv_repeat_count=2), 
            XceptionBlock(512, 512, dwsconv_repeat_count=2),
            XceptionBlock(512, 512, dwsconv_repeat_count=2),
            XceptionBlock(512, 512, dwsconv_repeat_count=2),
            XceptionBlock(512, 512, dwsconv_repeat_count=2),
        )

        # exit flow
        self.exit = torch.nn.Sequential(
            
            # paper original: XceptionBlock(728, 1024)
            XceptionBlock(512, 768, dwsconv_repeat_count=2),

            # paper original: DepthWiseSepConv2d(1024, 1536)
            DepthWiseSepConv2d(768, 1024), 

            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(inplace=True),

            # paper original: DepthWiseSepConv2d(1536, 2048)
            DepthWiseSepConv2d(1024, 2048), 

            torch.nn.BatchNorm2d(2048),
            torch.nn.ReLU(inplace=True),

            torch.nn.AdaptiveAvgPool2d((1, 1))
        )

        # fc layer
        self.last = torch.nn.Linear(2048, class_count)

    def forward(self, x):
        x = self.stem(x)
        x = self.entry(x)
        x = self.middle(x)
        x = self.exit(x)
        x = torch.flatten(x, 1)
        x = self.last(x)

        return x        