#!/usr/bin/env python
# coding: utf-8

# # Xception Implementation for Glaucoma (Finetuned from CIFAR-10 Training)

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


# RNG Seed Setting

# In[2]:


"""
-The seed provides a starting point for generating a series of random numbers.
-Provides a starting point for selecting a series of filters and weights for 
 the neural network for training. This eliminates the randomness of results 
 produced by the training of the network.
-In essence, this allows others to reproduce your results exactly by utilizing
 the same seed.
"""

# set the overall seed here
rnd_seed = 2025

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

random.seed(rnd_seed)
numpy.random.seed(rnd_seed)

torch.manual_seed(rnd_seed)
torch.cuda.manual_seed(rnd_seed)
torch.cuda.manual_seed_all(rnd_seed)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

data_generator = torch.Generator().manual_seed(rnd_seed)

# if multiple python workers are eventually used...
def worker_init_fn(worker_id):
    random.seed(rnd_seed + worker_id)
    numpy.random.seed(rnd_seed + worker_id)


# ## On GPU balanced accuracy computation

# In[3]:


def torch_compute_bacc(preds: torch.Tensor, labels: torch.Tensor, class_count: int):
    recall_per_class = []

    for label in range(class_count):
        true_positive = ((preds == label) & (labels == label)).sum().float()
        total_actual = (labels == label).sum().float()

        recall = true_positive / total_actual if total_actual > 0.0 else torch.tensor(0.0)
        recall_per_class.append(recall)

    return torch.stack(recall_per_class).mean().item()


# ## Glaucoma Download and Preparation

# In[4]:


"""
-Class count: defines number of possible classes for each image to be classified.
-Batch size:  the number of images loaded into the GPU for training at a time.
-Image size:  a n x n pixel size of each input image.
-Validation split (val_split): from the original training partion of data, a decimal value
                               that takes a portion of the training partion to simulate a
                               test partion. Here, it is 0.2 so 20% of the training data
                               will be used to simulate testing. Basically like a quiz.
"""

# Glaucoma data configuration
class_count = 2 # Glaucoma or not glaucoma

batch_size = 16 #  Had to use this batch size due to memory issues
    
image_size = 224 

model_path = "glaucoma_xcep_ft.pth"

# Replace validation split with appropriate paths to the dataset as this one already had the necessary splits
train_dir = "/home/jmulvihill/glaucoma-release-crop/release-crop/train"
test_dir = "/home/jmulvihill/glaucoma-release-crop/release-crop/test"
val_dir = "/home/jmulvihill/glaucoma-release-crop/release-crop/validation"


# In[5]:


# dataset mean, stdev, and label weight computation

# transform each image to a uniform size (here, 224 x 224)
stat_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(image_size),
    torchvision.transforms.ToTensor(),
])

# Apply transformations to training dataset
stat_train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=stat_transform)
stat_train_loader = torch.utils.data.DataLoader(stat_train_dataset, batch_size=batch_size, shuffle=True)

# after resizing the image, replace the RGB values with float values that can be graphed (normalization)
label_list, mean, std, count = [], 0.0, 0.0, 0.0
for images, labels in stat_train_loader:
    batch_count = images.size(0)
    images = images.view(batch_count, 3, -1)

    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    count += batch_count

    label_list.extend(labels.numpy())

# mean and std
mean /= count
std /= count

# label weight (inverse frequency)
label_count = collections.Counter(label_list)
label_total = sum(label_count.values())
label_weights = torch.tensor([label_total / (class_count * label_count[i]) for i in range(class_count)], \
                             dtype=torch.float)

"""
Mean and standard deviation of the input image RGB values are calculated as part
of the data normalization process. RBG has a domain of [0, 255] so it can only
represent positive numbers on a very limited range. To make training more applicable
to the real world, these RBG values are normalized to float (decimal) values so a wide
range of numbers can be represented. Computed from the TRAINING PARTITION ONLY.

-Label weights: some classes may have less samples than others, so to prevent
                bias towards the dominant class, assign more weight to classes
                with less samples and less weight to classes with more samples.
"""
print('Dataset Train Partition Stats')
print(f'        Mean: {mean}')
print(f'         Std: {std}')
print('Label Weights:', label_weights)


# In[6]:


# data loading
# train and test transforms/augmentations
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(image_size),
    torchvision.transforms.RandomCrop(image_size),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=mean, std=std)
])

""" Do not apply augmentations to testing """
test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(image_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=mean, std=std)
])

# train vs. validation split, creates the "quiz partition"
train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=train_transform)
test_dataset = torchvision.datasets.ImageFolder(test_dir, transform=train_transform)

# actual data loading for each partition, training, quiz, and test
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, \
                                           generator=data_generator)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, \
                                         generator=data_generator)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, \
                                          generator=data_generator)


# ## Xception Implementation

# In[7]:


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