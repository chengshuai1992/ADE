# Learning to Hash based on Angularly Discriminative Embedding-pytorch
This repository is reimplementation of [Learning to Hash based on Angularly Discriminative Embedding]
Original repo is [here](https://github.com/sudalvxin/ADE.git).

# Requirements
- Python 2.7
- PyTorch 0.4
- torchvision 0.3.0
- numpy 1.16.2

# How to run
Prepare dataset

Running experiments CIFAR10 

```python
$ python CIFAR10.py
```
# Note:
# [Different data (hash codes, and metrics) generally need different parameters.£©
We suggest that \lambda takes a larger value when code length $k$ and class number $c$ take larger value:
For example: 48bit, 64 bit
For CIFAR10 or SVHN~(mAP) \lambda = 0.1; CIFAR10 or SVHN~(Pre@2) \lambda = 0.3 
For Imagenet or CIFAR100~(mAP) \lambda = 0.3; Imagenet or CIFAR100~(Pre@2) \lambda = 1 

