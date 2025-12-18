# ResNet-18 from Scratch on MNIST

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)

This repository contains a full implementation of the **ResNet-18** architecture built from scratch using PyTorch. The model is specifically adapted to train on the **MNIST dataset** (grayscale handwritten digits).

## 📌 Project Overview

**ResNet (Residual Network)**, introduced in the 2016 paper *"Deep Residual Learning for Image Recognition"*, won the ILSVRC 2015 competition. It solved the "vanishing gradient" problem in deep networks by introducing **Skip Connections** (or Shortcut Connections), allowing gradients to flow through the network more easily.

This implementation uses the **Basic Block** structure found in ResNet-18 and ResNet-34.

### ⚙️ Adaptations for MNIST
The standard ResNet is designed for ImageNet ($224 \times 224$ inputs). To make it effective for MNIST ($28 \times 28$), we modified the **Initial Stem**:

1.  **Input Channels:** Changed from 3 (RGB) to **1 (Grayscale)**.
2.  **Initial Convolution:**
    * *Original:* $7 \times 7$ kernel, stride 2, followed by MaxPool stride 2.
    * *Modified:* **$3 \times 3$ kernel, stride 1**, removed MaxPool.
    * *Reason:* The original stem reduces the image size by $4\times$ instantly (from $224 \to 56$). If applied to MNIST, the image would drop from $28 \to 7$ before entering the first layer, destroying spatial information. Our modification preserves the $28 \times 28$ (or $32 \times 32$) resolution entering Layer 1.
3.  **Input Resize:** Images are resized to **$32 \times 32$** to facilitate clean downsampling factors ($32 \to 16 \to 8 \to 4 \to 2$).

## 🏗️ Architecture Details

### The Residual Block


The core component is the `BasicBlock`, formulated as:
$$y = \mathcal{F}(x, \{W_i\}) + x$$
Where $x$ is the input, and $\mathcal{F}$ represents the stacked convolution layers. The addition $+ x$ is the skip connection.

### Network Configuration (ResNet-18)

| Layer Name | Configuration | Output Channels | Output Size (Input $32\times32$) |
| :--- | :--- | :--- | :--- |
| **Stem** | Conv $3\times3$, stride 1 | 64 | $32 \times 32$ |
| **Layer 1** | [2 Basic Blocks] | 64 | $32 \times 32$ |
| **Layer 2** | [2 Basic Blocks] | 128 | $16 \times 16$ |
| **Layer 3** | [2 Basic Blocks] | 256 | $8 \times 8$ |
| **Layer 4** | [2 Basic Blocks] | 512 | $4 \times 4$ |
| **Classifier** | AvgPool $\to$ FC | 10 | $1 \times 1$ |

## 🚀 Getting Started

### Prerequisites
* Python 3.x
* PyTorch
* Torchvision

### Installation
Clone the repository:
```bash
git clone [https://github.com/your-username/resnet-mnist-scratch.git](https://github.com/your-username/resnet-mnist-scratch.git)
cd resnet-mnist-scratch
pip install torch torchvision
