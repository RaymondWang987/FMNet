#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun June 28 2020
@author: zouhongwei
"""


import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torchvision


__all__ = ['resnext101_wsl']



class ResNeXt(nn.Module):
    def __init__(self, model_name = 'resnext101_32x8d_wsl'):
        super(ResNeXt, self).__init__()
        wsl_resnext101 = model = torch.hub.load('facebookresearch/WSL-Images', model_name)


        self.conv1 = wsl_resnext101.conv1
        self.bn1 = wsl_resnext101.bn1
        self.relu = wsl_resnext101.relu
        self.maxpool = wsl_resnext101.maxpool
        self.layer1 = wsl_resnext101.layer1
        self.layer2 = wsl_resnext101.layer2
        self.layer3 = wsl_resnext101.layer3
        self.layer4 = wsl_resnext101.layer4

    def forward(self, x):
        features = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        #print(x.shape)
        features.append(x)

        x = self.layer2(x)
        #print(x.shape)
        features.append(x)

        x = self.layer3(x)
        #print(x.shape)
        features.append(x)

        x = self.layer4(x)
        #print(x.shape)
        features.append(x)
        #exit(1)

        return features

def resnext101_wsl():
    model = ResNeXt(model_name = 'resnext101_32x8d_wsl')

    return model

if __name__ == '__main__':
    img_seq = torch.ones(16,3,192,256)
    model = resnext101_wsl()
    out = model(img_seq)
    print(out[0].shape,out[1].shape,out[2].shape,out[3].shape)