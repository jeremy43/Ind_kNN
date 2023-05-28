import os
from torchvision import datasets, transforms
from torchvision import datasets as dataset
from datasets import load_dataset
from correct_weight_knn import PrepareData
from PIL import Image
import numpy as np
import torch
import network
from torch.utils.data import DataLoader
import utils
import sys
import os
import metrics
import argparse
from sentence_transformers import util
from utils import extract_feature
from sentence_transformers import util

def IndividualkNN(dataset, feature='resnet50', kernel_method='RBF',  num_query=1000,  nb_labels=10,  threshold_list=[0.5], seed=0, var=1., norm='centering+L2',dataset_path=None):

    private_data_list, private_label_list, query_data_list, query_label_list = PrepareData(dataset, feature, num_query, dataset_path, norm=norm, seed=seed)
    print(f'length of query list={len(query_data_list)}')
    print('shape of feature', private_data_list.shape)
    mask_idx = np.ones(len(private_data_list)) * 100
    # pointer to original idx
    ac_list = []
    sample_point = len(threshold_list)
    original_idx = np.array([x for x in range(len(private_data_list))])
    # keep_idx denote the data that is not deleted.
    #  private_data = torch.stack(private_data)
    max_clip = 0
    sum_neighbors = 0
    # predict_threshold records the predicted label when min_weight = threshold
    predict_threshold = np.zeros([ sample_point, num_query])
    predict_labels = []
    track_k_weight = []
    count_neighbor_list = np.zeros([sample_point, num_query])
    for idx in range(num_query):
        query_data = query_data_list[idx]
        if idx % 100 == 0:
            print('current query idx', idx)
        #print(f'idx is {idx}')
        #filter_private_data = private_data_list[mask_idx > 0]

        if kernel_method =='cosine':
            kernel_weight = np.dot(private_data_list, query_data)
            #print(f'max of kernel weight is {max(kernel_weight)} and min of kernel weight is {min(kernel_weight)}')
        elif kernel_method == 'RBF':
            kernel_weight = [np.exp(-np.linalg.norm(private_data_list[i] - query_data) ** 2 / var) for i in range(len(private_data_list))]
        elif kernel_method == 'student':
            kernel_weight = [(1+np.linalg.norm(private_data_list[i] - query_data) ** 2 / var)**(-(var+1)/2) for i in range(len(private_data_list))]
        kernel_weight = np.array(kernel_weight)
        print(f'max kernel weight is {max(kernel_weight)} and min of kernel weight is {min(kernel_weight)}')

        for (j,threshold) in enumerate(threshold_list):
            original_top_index_set =  original_idx[np.where(kernel_weight>threshold)]
            count_neighbor_list[j, idx] = len(original_top_index_set)
            if idx %100 == 0:
                print(f'threshold is {threshold} number of neighbors above threshold is {len(original_top_index_set)}')
            vote_count = np.zeros(nb_labels)
            if len(original_top_index_set)==0:
                predict_labels.append(0)
                continue
            for i in range(len(original_top_index_set)):
                select_idx = original_top_index_set[i]
                vote_count[private_label_list[select_idx]] +=min(mask_idx[select_idx], kernel_weight[select_idx])
                #mask_idx[select_idx]-=  kernel_weight[select_idx]
            predict_threshold[j][idx]=np.argmax(vote_count)

    for idx in range(sample_point):
        prediction = predict_threshold[idx,:]
        print('prediction', prediction.shape)
        print(f'length of prediction is {len(prediction)}')
        accuracy = 100*metrics.accuracy(prediction, query_label_list)
        ac_list.append(accuracy)

    return count_neighbor_list,  ac_list
