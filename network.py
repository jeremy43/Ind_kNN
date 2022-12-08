from __future__ import print_function, absolute_import
import os, csv
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import scipy.io as sio
from sklearn.metrics import hamming_loss
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.optim import lr_scheduler
import math


def learning_rate(init, epoch):
    step = 25000 * epoch
    # step = int(600000/config.nb_teachers)*epoch
    optim_factor = 0
    if step > 150000:
        optim_factor = 3
    elif step > 100000:
        optim_factor = 2
    elif step > 50000:
        optim_factor = 1

    return init * math.pow(0.2, optim_factor)


def predFeature(model, dataset):
    torch.manual_seed(0)
    # os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        # print("Currently using GPU {}".format(config.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(0)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    pin_memory = True if use_gpu else False

    testloader = DataLoader(dataset, batch_size=200, shuffle=False,
                            pin_memory=pin_memory, drop_last=False,
                            )

    if use_gpu:
        model = nn.DataParallel(model).cuda()
    model.eval()

    with torch.no_grad():
        pred_list, feature_list = [], []
        float_logit_list = []
        for batch_idx, (imgs, label) in enumerate(testloader):
            if use_gpu:
                imgs = imgs.cuda()
            if batch_idx == 0:
                print('image before pretrain', imgs.shape)
            if batch_idx % 50 == 0:
                print('batch {}/{}', batch_idx, len(testloader))
            features = model(imgs)
            # print('feature', features.shape)
            # predA = predA.cpu()
            # print('features shape {} predA shape'.format(features.shape, predA.shape))

            feature_list.append(features.cpu())
            # print('predAs', predicted)
        # float_logit_list = (((torch.cat(float_logit_list, 0)).float()).numpy()).tolist()
        # float_logit_list = np.array(float_logit_list)

    feature_list = (((torch.cat(feature_list, 0)).float()).numpy()).tolist()
    feature_list = np.array(feature_list)

    return feature_list
