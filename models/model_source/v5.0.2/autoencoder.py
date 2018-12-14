import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AutoEncoderLandmark(nn.Module):

    def __init__(self, input_size:int):
        super(AutoEncoderLandmark, self).__init__()
        # AutoEncoder

        # Encoder
        self.conv_layer1_e = nn.Sequential(
            nn.Conv2d(input_size, 256, kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.MaxPool2d((2,2)))
        self.conv_layer2_e = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.MaxPool2d((2,2)))
        self.conv_layer3_e = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.MaxPool2d((2,2)))
        self.conv_layer4_e = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.MaxPool2d((2,2)))
        self.conv_layer5_e = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ELU())

        # Decoder 
        self.conv_layer5_d = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ELU())
            
        
        self.conv_layer4_d = nn.Sequential(
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(256, 256, kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ELU())
            

        self.conv_layer3_d = nn.Sequential(
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(256, 256, kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ELU())
            

        self.conv_layer2_d = nn.Sequential(
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(256, 256, kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ELU())
           

        self.conv_layer1_d = nn.Sequential(
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(256, 256, kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ELU())
           

        self.conv_link_layer1 = nn.Conv2d(256, 256, kernel_size=(1,1),stride=1, padding=0)
        self.conv_link_layer2 = nn.Conv2d(256, 68, kernel_size=(1,1),stride=1,padding=0)
         
        
        
        

    def forward(self, x):
        #x = x.float()
        #x = x.cuda()
        images= x

        #Encoder
        x = self.conv_layer1_e(x)
        x = self.conv_layer2_e(x)
        x = self.conv_layer3_e(x)
        x = self.conv_layer4_e(x)
        x = self.conv_layer5_e(x)

        #Decoder
        x = self.conv_layer5_d(x)
        x = self.conv_layer4_d(x)
        x = self.conv_layer3_d(x)
        x = self.conv_layer2_d(x)
        x = self.conv_layer1_d(x)

        #Link
        x = self.conv_link_layer1(x)
        x = self.conv_link_layer2(x)
       
        return x