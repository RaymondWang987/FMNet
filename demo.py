import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import argparse
import cv2
from os.path import join as pjoin
import json
import math 
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import matplotlib.pyplot as plt
import random
import os
from torchvision import transforms, utils
import scipy.io as io
import resnext101_wsl
from DepthNet4 import *
from ConvTrans import *    
from DepthNet4 import Decoder
from DepthNet4 import Decoder_noAO
import matplotlib.pyplot as plt
from torch.nn import DataParallel
import h5py
import glob
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True   

def img_loader(path):
    
    image = cv2.imread(path)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

    return image


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

seq_len= 12
small_seq_len = 2

devices = [f'cuda:{c}' for c in range(torch.cuda.device_count())]

device_cnn_encoder = torch.device(devices[0])
device_fmnet_encoder = torch.device(devices[0])
device_fmnet_decoder = torch.device(devices[-1])
device_cnn_decoder = torch.device(devices[-1])



cnn_encoder = resnext101_wsl.resnext101_wsl()  #resnet.resnet50() #resnext101_wsl.resnext101_wsl()    
fmnet_Encoder = fmnet_encoder(num_hidden = [2048],batch_size = 1 ,img_width = 20,img_height = 15 ,input_length = seq_len ,small_length = small_seq_len ,encoder_depth = 6, device = device_fmnet_encoder)
fmnet_Decoder = fmnet_decoder(num_hidden = [2048],batch_size = 1 ,img_width = 20,img_height = 15 ,input_length = seq_len ,small_length = small_seq_len ,encoder_depth = 1, device = device_fmnet_decoder)
cnn_decoder = Decoder(inchannels=[256, 512, 1024, 2048], midchannels=[256, 256, 256, 512], upfactors=[2, 2, 2, 2], outchannels=1)
  
checkpoint = torch.load('./checkpoint/nyu_epoch_20.pth',map_location = 'cpu')    #18
cnn_encoder.load_state_dict(checkpoint['cnn_encoder']) 
cnn_decoder.load_state_dict(checkpoint['cnn_decoder'])   
fmnet_Encoder.load_state_dict(checkpoint['Mae_Encoder'])
fmnet_Decoder.load_state_dict(checkpoint['Mae_Decoder'])

cnn_encoder.to(device_cnn_encoder)
fmnet_Encoder.to(device_fmnet_encoder)
fmnet_Decoder.to(device_fmnet_decoder)
cnn_decoder.to(device_cnn_decoder)


base_dir = './demo/'
image_dir = base_dir + '/rgb/'
image_seq = glob.glob(image_dir+'*.png')
image_seq.sort()

for j in range(seq_len):
    img = img_loader(image_seq[j])
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    img = torch.Tensor(np.ascontiguousarray(img).astype(np.float32))
    img = img.unsqueeze(0)
    if j==0:
        img_seq = img
    else:
        img_seq=torch.cat([img_seq,img],dim=0)

with torch.no_grad():
    img_seq = img_seq.to(device_cnn_encoder)
    img_seq = img_seq.squeeze(0)
    spatial_feat0, spatial_feat1, spatial_feat2,outputs = cnn_encoder(img_seq)
    outputs = outputs.to(device_fmnet_encoder)
    outputs, choose_frames = fmnet_Encoder(outputs)
    outputs = outputs.to(device_fmnet_decoder)
    outputsall = fmnet_Decoder(outputs, choose_frames)
    spatial_feat0 = spatial_feat0.to(device_cnn_decoder)
    spatial_feat1 = spatial_feat1.to(device_cnn_decoder)
    spatial_feat2 = spatial_feat2.to(device_cnn_decoder)
    outputsall = outputsall.to(device_cnn_decoder)
    outputsall = cnn_decoder([spatial_feat0, spatial_feat1, spatial_feat2, outputsall])[0]
    outputsall = F.relu(outputsall).squeeze()
        
        
for k in range(outputsall.shape[0]):
    vis = outputsall[k].detach().cpu().numpy().squeeze()
    vis[vis!=0] = 1.0 / vis[vis!=0]
    plt.imsave('./demo/results/'+str(k+1)+'.png',vis, cmap='inferno',vmin =np.min(vis) , vmax = np.max(vis))

        
