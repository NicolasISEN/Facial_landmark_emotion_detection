import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net32(nn.Module):

    def __init__(self, input_size:int):
        super(Net32, self).__init__()

        # 1 input image channel, 6 output channels, 5x5 square convolution
        
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(input_size, 32, kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d((2,2)))
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d((2,2)))
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d((2,2)))
        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.MaxPool2d((2,2)))
        self.conv_layer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ELU(),
            nn.MaxPool2d((2,2)))

        # an affine operation: y = Wx + b

        self.linear_layer1 = nn.Sequential(
            nn.Linear(512*2*2, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(),
            nn.Dropout(p=0.2))
        self.linear_layer2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(),
            nn.Dropout(p=0.2))
        self.linear_layer3 = nn.Sequential(
            nn.Linear(1024, 7),
            nn.Softmax())
        

    def forward(self, x):
        
        # Max pooling over a (2, 2) window
        x = x.float()
        x = x.cuda()
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.conv_layer5(x)
        x = x.view(-1,512*2*2)
        x = self.linear_layer1(x)
        x = self.linear_layer2(x)
        x = self.linear_layer3(x)
        return x

class Net256(nn.Module):

    def __init__(self, input_size:int):
        super(Net256, self).__init__()

        # 1 input image channel, 6 output channels, 5x5 square convolution
        
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(input_size, 256, kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.MaxPool2d((2,2)))
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.MaxPool2d((2,2)))
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.MaxPool2d((2,2)))
        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.MaxPool2d((2,2)))
        self.conv_layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.MaxPool2d((2,2)))

        # an affine operation: y = Wx + b

        self.linear_layer1 = nn.Sequential(
            nn.Linear(256*2*2, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(),
            nn.Dropout(p=0.2))
        self.linear_layer2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(),
            nn.Dropout(p=0.2))
        self.linear_layer3 = nn.Sequential(
            nn.Linear(1024, 7),
            nn.Softmax())
        

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = x.float()
        x = x.cuda()
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.conv_layer5(x)
        x = x.view(-1,256*2*2)
        x = self.linear_layer1(x)
        x = self.linear_layer2(x)
        x = self.linear_layer3(x)
        return x   