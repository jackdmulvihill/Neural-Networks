#!/usr/bin/env python
# coding: utf-8

# # AlexNet Implementation and Training from Scratch with Glaucoma 

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
Implementation based on: https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf

See Figure 2
"""
class AlexNet(torch.nn.Module):
    def __init__(self, class_count):
        super(AlexNet, self).__init__()

        # Hidden convolution layers
        self.layer1 = torch.nn.Sequential(
            # In the beginning, there are only three channels as the input
            # is represented in RGB values
            torch.nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2)
        ) # output: (96, 55, 55)

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(96, 256, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
        ) # output: (256, 27, 27)

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 384, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        ) # output: (384, 13, 13)

        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(384, 384, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        ) # output: (384, 13, 13)

        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2)
        ) # output: (256, 13, 13)

        # Fully connected "head" layer
        self.head = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256 * 6 * 6, 4096),
            torch.nn.ReLU(inplace=True),

            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
        )

        self.last = torch.nn.Linear(4096, class_count)

    def forward(self, x):
        # Pass image through layers 1-5
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        # Transforms the input image into a vector-like object
        # makes it easier for the neural network to process because
        # there are no more convolutions occuring.
        x = torch.flatten(x, 1)  # flatten all but batch dimension
        # Pass the image through the fully connected layers
        x = self.head(x)
        x = self.last(x)
        return x