# Neural-Networks
## Training

All base models were trained using the CIFAR-10 dataset and all finetuned models were trained on a binary toy case glaucoma dataset local to the authors.

- **CIFAR-10:** https://www.cs.toronto.edu/~kriz/cifar.html
- **Glaucoma:** A binary toy case dataset containing X-ray images of eyes, some of patients with glaucoma, some without

## AlexNet Architecture 

Implementation of AlexNet based on the paper "ImageNet Classification with Deep Convolutional Neural Networks." The original architecture that popularized deep learning for computer vision and would go on to inspire other convolutional neural networks such as VGGNet and ResNet.

### Architecture Overview

The network consists of 5 convolutional layers followed by 3 fully connected layers:

- **Layer 1:** 11×11 conv, 96 filters, stride 4 → ReLU → 3×3 max pool, stride 2
- **Layer 2:** 5×5 conv, 256 filters → ReLU → 3×3 max pool, stride 2
- **Layer 3:** 3×3 conv, 384 filters → ReLU
- **Layer 4:** 3×3 conv, 384 filters → ReLU
- **Layer 5:** 3×3 conv, 256 filters → ReLU → 3×3 max pool, stride 2
- **Classifier:** Flatten → FC-4096 → ReLU → Dropout(0.5) → FC-4096 → ReLU → Dropout(0.5) → FC-n_classes

### Key Features

- **Large initial kernels:** 11×11 convolution in first layer captures broad patterns
- **Aggressive downsampling:** Stride of 4 in first layer reduces spatial dimensions quickly
- **ReLU activation:** First major network to use ReLU instead of tanh/sigmoid, enabling faster training
- **Dropout regularization:** 0.5 dropout rate in fully connected layers prevents overfitting
- **Deep architecture:** 8 weight layers (5 conv + 3 FC)
- **Expects 224×224 RGB input images**

### Parameters

- Input: 224×224×3 RGB images
- Output: n-classes (configurable)
- Total parameters: ~62M (varies with number of output classes)

## VGG-16 Architecture (Configuration D)

Implementation of VGG-16 based on the paper "Very Deep Convolutional Networks for Large-Scale Image Recognition" **with batch normalization added** (not included in the original) for improved training stability.

### Architecture Overview

The network consists of 13 convolutional layers organized into 5 blocks, followed by 3 fully connected layers:

- **Block 1:** 2 × conv3-64 → max pool
- **Block 2:** 2 × conv3-128 → max pool
- **Block 3:** 3 × conv3-256 → max pool
- **Block 4:** 3 × conv3-512 → max pool
- **Block 5:** 3 × conv3-512 → max pool
- **Classifier:** FC-4096 → FC-4096 → FC-n_classes

### Key Features

- All convolutional layers use 3×3 kernels with stride 1 and padding 1
- Batch normalization applied after each convolutional layer
- ReLU activation functions throughout
- Max pooling layers use 2×2 kernels with stride 2
- Dropout (0.5) in fully connected layers for regularization
- Expects 224×224 RGB input images

### Layer Count

- 13 convolutional layers + 3 fully connected layers = **16 weight layers**

### Parameters

- Input: 224×224×3 RGB images
- Output: n-classes (configurable)
- Total parameters: ~138M (varies with number of output classes)

## ResNet-18 Architecture

Implementation of the 18-layer Residual Network based on the paper "Deep Residual Learning for Image Recognition." This architecture introduces skip connections (residual connections) that allow training of very deep networks.

### Architecture Overview

The network consists of an initial convolutional layer, 4 residual layer groups with 2 BasicBlocks each, and a fully connected layer:

- **Initial Layer:** 7×7 conv, 64 filters, stride 2 → batch norm → ReLU → 3×3 max pool, stride 2
- **Layer 1 (conv2_x):** 2 BasicBlocks, 64 channels, stride 1
- **Layer 2 (conv3_x):** 2 BasicBlocks, 128 channels, stride 2
- **Layer 3 (conv4_x):** 2 BasicBlocks, 256 channels, stride 2
- **Layer 4 (conv5_x):** 2 BasicBlocks, 512 channels, stride 2
- **Output:** Global average pooling → FC-n_classes

### BasicBlock Structure

Each BasicBlock contains:
- 3×3 conv → batch norm → ReLU
- 3×3 conv → batch norm
- Skip connection (identity shortcut or 1×1 conv projection when dimensions change)
- ReLU activation after addition

### Key Features

- Skip connections enable training of deeper networks by addressing vanishing gradient problem
- Batch normalization after each convolutional layer
- Downsampling via strided convolutions (stride 2) in layers 2-4
- Global average pooling eliminates need for large fully connected layers
- Identity shortcuts when dimensions match, projection shortcuts (1×1 conv) when they don't
- Expects 224×224 RGB input images

### Layer Count

- 1 initial conv + 8 conv layers in BasicBlocks (4 layers × 2 blocks × 2 convs) + 1 FC = **18 weight layers**

### Parameters

- Input: 224×224×3 RGB images
- Output: n_classes (configurable)
- Total parameters: ~11M

## Xception Architecture

Implementation of Xception (Extreme Inception) based on the paper "Xception: Deep Learning with Depthwise Separable Convolutions." This architecture replaces standard Inception modules with depthwise separable convolutions for improved efficiency.

### Architecture Overview

The network consists of three main flows preceded by an entry stem:

- **Stem:** 2 standard convolutions (3→32→64 channels)
- **Entry Flow:** 3 XceptionBlocks with increasing channels (64→128→256→512)
- **Middle Flow:** 8 XceptionBlocks maintaining 512 channels with repeated depthwise separable convolutions
- **Exit Flow:** Final XceptionBlock (512→768) followed by depthwise separable convolutions (768→1024→2048)
- **Classifier:** Global average pooling → FC-n_classes

### Key Components

**DepthWiseSepConv2d (Depthwise Separable Convolution):**
- Depthwise convolution: 3×3 convolution applied independently to each input channel
- Pointwise convolution: 1×1 convolution to combine channel outputs
- Significantly reduces parameters compared to standard convolutions

**XceptionBlock:**
- Multiple depthwise separable convolutions with batch normalization and ReLU
- Skip connections (residual connections) similar to ResNet
- Optional downsampling via max pooling and 1×1 convolutions when channels increase
- Configurable repetition count for depthwise separable convolutions

### Architecture Details

**Entry Flow:**
- Block 1: 64 → 128 channels (downsample)
- Block 2: 128 → 256 channels (downsample)
- Block 3: 256 → 512 channels (downsample)

**Middle Flow:**
- 8 identical blocks maintaining 512 channels
- Each block contains 3 depthwise separable convolutions (1 initial + 2 repeated)
- No downsampling, focuses on feature refinement

**Exit Flow:**
- Block: 512 → 768 channels with 3 depthwise separable convolutions
- Depthwise separable conv: 768 → 1024
- Depthwise separable conv: 1024 → 2048
- Global average pooling

### Key Features

- Depthwise separable convolutions reduce computational cost by ~8-9× compared to standard convolutions
- Skip connections enable training of very deep networks
- Batch normalization after each convolution for training stability
- ReLU activations throughout (except final layer)
- Global average pooling eliminates large fully connected layers
- Expects 299x299 RGB input images

### Parameters

- Input: 299x299x3 RGB images
- Output: n-classes (configurable)
- Total parameters: ~23M

## Streamlined Architectures (Coming Soon)

Recommended for deployment, identical to the base models but have the dataset statistics and models saved in individual files. This ensures that if one changes, the changes will be applied everywhere since the dataset statistics and models are imported from said files.

## EnsembleNet (See Streamlined Architectures) (Coming Soon)

Combines the classification power of all aforementioned architectures into an ensemble-learning type classifier with an implementation of both hard and soft majority voting.

### Architecture Overview
- Loads each model
- Stores the predictions to an array with truth labels stored in a separate array
- Perform hard or soft majority voting with each model's predictions as an ensemble classifier

### Parameters

- Input: 224x224x3 RGB images
- Output: n-classes (configurable)
