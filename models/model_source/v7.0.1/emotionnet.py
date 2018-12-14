import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EmotionNet(nn.Module):

    def __init__(self):
        super(EmotionNet, self).__init__()
        # EmotionNet

        # Emotions
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(3,3),padding=1),
            nn.Conv2d(128, 128, kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Dropout(p=0.2))
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3,3),padding=1),
            nn.Conv2d(128, 128, kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Dropout(p=0.2))
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3,3),padding=1),
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
        
            
        
        
        

    def forward(self, emotion):
        emotion = emotion.float()
        
        emotion = self.conv_layer1(emotion)
        emotion = self.conv_layer2(emotion)
        emotion = self.conv_layer3(emotion)
        emotion = self.conv_layer4(emotion)
        #print(emotion.shape)
        emotion = emotion.view(emotion.size(0),-1)
        #print(emotion.shape)
        emotion = self.emotion_linear_layer1(emotion)
        emotion = self.emotion_linear_layer2(emotion)

       
        return emotion