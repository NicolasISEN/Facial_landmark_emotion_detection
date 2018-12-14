import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()
        # AutoEncoder

        # Encoder
        self.conv_layer1_e = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(3,3),stride=1,padding=1),
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
           
        # Landmarks
        self.landmark_conv_layer1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ELU())

        self.landmark_conv_layer2 = nn.Sequential(
            nn.Conv2d(128, 68, kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(68),
            nn.ELU())

        # Emotions
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(257, 256, kernel_size=(3,3),padding=1),
            nn.Conv2d(256, 256, kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Dropout(p=0.2))
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3,3),padding=1),
            nn.Conv2d(256, 256, kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Dropout(p=0.2))
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3,3),padding=1),
            nn.Conv2d(256, 256, kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Dropout(p=0.2))
        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3,3),padding=1),
            nn.Conv2d(512, 512, kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Dropout(p=0.2))


        self.emotion_linear_layer1 = nn.Sequential(
            nn.Linear(8*8*512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU())

        self.emotion_linear_layer2 = nn.Sequential(
            nn.Linear(256, 5),
            nn.Softmax())
        
            
        
        
        

    def forward(self, x):
        x = x.float()
        image = x.data

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
        
        #Landmark
        landmark = x.data
        #print(landmark.shape)
        #landmark=landmark.view(landmark.size(0),-1)
        #print(landmark.shape)
        landmark = self.landmark_conv_layer1(landmark)
        landmark = self.landmark_conv_layer2(landmark)


        

        #Emotion
        emotion = x.data
        emotion.require_grad = True
        emotion = torch.cat((x,image),1)
        emotion = self.conv_layer1(emotion)
        emotion = self.conv_layer2(emotion)
        emotion = self.conv_layer3(emotion)
        emotion = self.conv_layer4(emotion)
        #print(emotion.shape)
        emotion = emotion.view(emotion.size(0),-1)
        #print(emotion.shape)
        emotion = self.emotion_linear_layer1(emotion)
        emotion = self.emotion_linear_layer2(emotion)

       
        return emotion,landmark
