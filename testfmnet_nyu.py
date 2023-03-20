import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
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


def gt_png_loader(path): 

    depth = cv2.imread(path, -1)

    depth = depth / 1000.0
    #real_depth[real_depth!=0] = 1.0 / real_depth[real_depth!=0]

    #disp_norm = real_depth / np.max(real_depth)
    #valid_mask = real_depth != 0

    return depth.astype(np.float32)


def compute_errors(pred, gt):
    
    #pred = torch.clamp(pred, 1e-6, 9.9955)
    pred = pred[gt>0] 
    gt = gt[gt>0]

    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy()

    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()
    
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())
    
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    
    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - 0.85*np.mean(err) ** 2) * 10
    
    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)
    
    return rmse, rmse_log, silog, log10, abs_rel, sq_rel, d1, d2, d3

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

seq_len= 12
small_seq_len = 2

device_cnn_encoder = torch.device("cuda:0")
device_fmnet_encoder = torch.device("cuda:0")
device_fmnet_decoder = torch.device("cuda:1")
device_cnn_decoder = torch.device("cuda:1")



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


base_dir = './data/testnyu_data/'
alld1,alld2,alld3 = 0,0,0
allrmse, allrmse_log, allsilog, alllog10, allabs_rel, allsq_rel = 0,0,0,0,0,0
for i in range(654):

    print('Testing sequence:',i+1,'/654')

    image_dir = base_dir + str(i+1) +'/rgb/'

    gt_dir = base_dir + str(i+1) +'/gt/'

    image_seq = glob.glob(image_dir+'*.png')
    image_seq.sort(key=lambda x: int(x[-10:-4]))
    gt_pic = glob.glob(gt_dir+'*.png')[0]
    pic_num = int(gt_pic[-10:-4])

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

    gt = gt_png_loader(gt_pic)
    gt = torch.Tensor(np.ascontiguousarray(gt.astype(np.float32)))


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
        outputsall = F.relu(outputsall)
            

        pred = outputsall[pic_num].squeeze()
        pred = torch.clamp(pred, min=0.7133, max=9.9955)
        gt = gt.squeeze()
        gt = gt[44:471, 40:601]
        pred = pred[44:471, 40:601]
        rmse, rmse_log, silog, log10, abs_rel, sq_rel, d1, d2, d3 = compute_errors(pred,gt)
        print(d1,d2,d3)

        alld1 = alld1 + d1
        alld2 = alld2 + d2
        alld3 = alld3 + d3
        allrmse = allrmse + rmse
        allrmse_log = allrmse_log + rmse_log
        allsilog = allsilog + silog
        alllog10 = alllog10 + log10
        allabs_rel = allabs_rel + abs_rel
        allsq_rel = allsq_rel + sq_rel


print('d1:',alld1/654)

print('d2:',alld2/654)

print('d3:',alld3/654)

print('RMSE:',allrmse/654)

print('log10',alllog10/654)

print('Absrel:',allabs_rel/654)


        
