import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AutoEncoder(nn.Module):

    def __init__(self, input_size:int):
        super(AutoEncoder, self).__init__()
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
           

        self.conv_link_ch1_layer1 = nn.Conv2d(256, 256, kernel_size=(1,1),stride=1, padding=0)
        self.conv_link_ch1_layer2 = nn.Conv2d(256, 256, kernel_size=(1,1),stride=1,padding=0)
        self.conv_link_ch2_layer1 = nn.Conv2d(256, 68, kernel_size=(1,1),stride=1,padding=0)
        self.conv_link_ch2_layer2 = nn.Conv2d(68, 256, kernel_size=(1,1),stride=1,padding=0)


        #Emotions
        self.emotion_conv_layer1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.MaxPool2d((2,2)))

        self.emotion_conv_layer2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.MaxPool2d((2,2)))

        self.emotion_conv_layer3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.MaxPool2d((2,2)))

        self.emotion_conv_layer4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.MaxPool2d((2,2)))

        self.emotion_conv_layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.MaxPool2d((2,2)))

        self.emotion_linear_layer1 = nn.Sequential(
            nn.Linear(256*2*2, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Dropout(p=0.2))
        
        self.emotion_linear_layer2 = nn.Sequential(
            nn.Linear(512, 7),
            nn.Softmax())
            
        
        
        

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
        x1 = self.conv_link_ch1_layer1(x)
        x1 = self.conv_link_ch1_layer2(x1)

        x2 = self.conv_link_ch2_layer1(x)
        #Landmarks
        landmarks = x2
        x2 = self.conv_link_ch2_layer2(x2)

        #Sum Tensor
        x = x1+x2
        #x = torch.cat((x,images),1)

        #Emotion
        x = self.emotion_conv_layer1(x)
        x = self.emotion_conv_layer2(x)
        x = self.emotion_conv_layer3(x)
        x = self.emotion_conv_layer4(x)
        x = self.emotion_conv_layer5(x)
        x = x.view(-1,256*2*2)
        x = self.emotion_linear_layer1(x)
        x = self.emotion_linear_layer2(x)

       
        return landmarks,x
