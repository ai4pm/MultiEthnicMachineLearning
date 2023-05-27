# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 21:34:22 2022

@author: tsharma2
"""

import torch
import torch.nn as nn
#import torch.nn.functional as F
#from models.BasicModule import BasicModule

class DCD(nn.Module):
    def __init__(self,h_features=64,input_features=128):
        super(DCD,self).__init__()
        self.d0=nn.Dropout(0.5)
        self.fc1=nn.Linear(input_features,h_features)
        self.d1=nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(h_features)
        self.fc2=nn.Linear(h_features,32)
        self.d2=nn.Dropout(0.5)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3=nn.Linear(32,4)

    def forward(self,inputs):
        out=torch.relu(self.fc1(self.d0(inputs)))
        out=self.d1(out)
        out=self.bn1(out)
        out=torch.relu(self.fc2(out))
        out=self.d2(out)
        out=self.bn2(out)
        # return torch.softmax((out),dim=1)
        # return torch.softmax(self.fc3(out),dim=1)
        return torch.softmax(self.fc3(out),dim=1)
        #return self.fc3(out)

class Network(nn.Module):
    def __init__(self,in_features_data=189,nb_classes=2,dropout=0.5,hiddenLayers=[128, 64]):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_features_data, hiddenLayers[0]),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hiddenLayers[0], hiddenLayers[1]),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            #nn.Dropout(dropout),
            nn.Linear(hiddenLayers[1], nb_classes),
        )
    def forward(self, x):
        feature = self.feature(x)                           # Sequential_1
        pred = torch.sigmoid(self.classifier(feature))      # Dropout -> Dense -> Activation
        return pred, feature


