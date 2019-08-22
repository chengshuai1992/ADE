
import torch
import torch.nn as nn
import torchvision
import numpy as np
from torchvision import models
from torch.autograd import Function
from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
import math
import pdb


class hash_CNN(nn.Module):
    def __init__(self, encode_length, num_classes):
        super(hash_CNN, self).__init__()
        self.alex = torchvision.models.alexnet(pretrained=True)
        self.alex.classifier = nn.Sequential(*list(self.alex.classifier.children())[:6])
        self.fc_plus = nn.Linear(4096, encode_length)

    def forward(self, x):
        x = self.alex.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.alex.classifier(x)
        x = self.fc_plus(x)                         #the learned features
        code = torch.sign(x)
        return x, code


class hash_SOFTCNN(nn.Module):
    def __init__(self, encode_length, num_classes):
        super(hash_SOFTCNN, self).__init__()
        self.alex = torchvision.models.alexnet(pretrained=True)
        self.alex.classifier = nn.Sequential(*list(self.alex.classifier.children())[:6])
        self.fc_plus = nn.Linear(4096, encode_length)
        self.fc = nn.Linear(encode_length, num_classes)
    def forward(self, x):
        x = self.alex.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.alex.classifier(x)
        x = self.fc_plus(x)  # the learned features
        code = torch.sign(x)
        out = self.fc(x)
        return x, code, out


