import os
import pickle
from hash import LSH
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import datasets as dataset
from PIL import Image
import numpy as np
from datasets import load_dataset
import torch
import network
from torch.utils.data import DataLoader
from correct_weight_knn import PrepareData
from sentence_transformers import util
import utils
import sys
import os
import metrics
from utils import extract_feature
from numpy import linalg as LA
import argparse

dataset_path = '/home/yq/dataset'



def IndividualkNN(dataset, kernel_method='RBF',  hash_method = 'knn',seed=0, min_weight = 0.2,  num_tables = 2, proj_dim=12,feature='resnet50', nb_teachers= 50, num_query=1000, nb_labels=10, ind_budget=20, noisy_scale=0.1, var=1., norm='centering+L2', dataset_path=None):
    # mask_idx masked private data that are deleted.  only train_data[mask_idx!=0] will be used for kNN.
        print('which norm', norm)
    private_data_list, private_label_list, query_data_list, query_label_list = PrepareData(dataset, feature, num_query, dataset_path, seed, norm=norm)
    print('shape of feature', private_data_list.shape)
    print(f'the second data norm is {LA.norm(private_data_list[0])}')
    # construct hash table
    print(f'num_tables is {num_tables} and proj_dim is {proj_dim}')
    hash_path = f'hash_table/{dataset}_num_tables{num_tables}_projdim{proj_dim}_norm{norm}.pkl'
    if os.path.exists(hash_path):
        with open(hash_path, 'rb') as f:
            lsh_hash = pickle.load(f)
    else:
        lsh_hash = LSH(num_tables, proj_dim, private_data_list.shape[1])
        for (idx, x) in enumerate(private_data_list):
            if idx % 10000 == 0:
                print(f'prepare hash table for idx={idx}')
            lsh_hash[x] = idx
        with open(hash_path, 'wb') as f:
            pickle.dump(lsh_hash, f)

    mask_idx = np.ones(len(private_data_list)) * ind_budget
    # pointer to original idx
    private_label_list = np.array(private_label_list)
    predict_labels = []
    sum_neighbors = 0
    for idx in range(num_query):
        query_data = query_data_list[idx]
        if idx % 100 == 0:
            print('current query idx', idx)

        # filter_private_data  = private_data_list[mask_idx>0]
        # print('len of filter_private_data', len(filter_private_data))
        hash_neighbors = lsh_hash.query_neighbor(query_data)
        # print(f'length of hash neighbors is {len(hash_neighbors)}')
        select_neighbors = [x for x in hash_neighbors if mask_idx[x] > 0]
        select_neighbors = np.array(select_neighbors, dtype=int)
        # sum_neighbors+=len(select_neighbors)
        vote_count = np.zeros(nb_labels)
        if kernel_method == 'cosine':
            kernel_weight = np.dot(private_data_list[select_neighbors], query_data)
            # print(f'kernel_weight shape is {kernel_weight.shape}')
            # temp_d = util.cos_sim(private_data_list[select_neighbors], query_data).reshape(-1)
            # kernel_weight = [np.exp(-temp_d[i] ** 2 / var) for i in range(len(select_neighbors))]
        elif kernel_method == 'RBF':
            kernel_weight = [np.exp(-np.linalg.norm(private_data_list[x] - query_data) ** 2 / var) for x in select_neighbors]
        # print(f' min of weight is {min(kernel_weight)} and max weight is {max(kernel_weight)}')
        elif kernel_method == 'student':
            kernel_weight = [(1 + np.linalg.norm(private_data_list[x] - query_data) ** 2 / var) ** (-(var + 1) / 2) for x in select_neighbors]
        normalized_weight = kernel_weight
        kernel_weight = np.array(kernel_weight)
        nb_teachers = len(kernel_weight)
        if hash_method == 'basic':
            for i in range(len(select_neighbors)):
                neighbor  = select_neighbors[i]
                vote_count[private_label_list[neighbor]]+= min(np.sqrt(mask_idx[neighbor]),normalized_weight[i])
                mask_idx[neighbor]-=  normalized_weight[i]**2
        elif hash_method == 'basic+norm':
            sum_weight = sum(normalized_weight)
            normalized_weight = [x / sum_weight for x in normalized_weight]
            for i in range(len(select_neighbors)):
                neighbor  = select_neighbors[i]
                vote_count[private_label_list[neighbor]]+= min(np.sqrt(mask_idx[neighbor]),normalized_weight[i])
                mask_idx[neighbor]-=  normalized_weight[i]**2
        elif hash_method =='knn':
            top_k_index = np.argsort(-kernel_weight)[:nb_teachers]
            for i in range(len(top_k_index)):
                neighbor  = select_neighbors[top_k_index[i]]
                vote_count[private_label_list[neighbor]]+= min(np.sqrt(mask_idx[neighbor]),normalized_weight[top_k_index[i]])
            for i in range(len(normalized_weight)):
                neighbor = select_neighbors[i]
                mask_idx[neighbor]-=  normalized_weight[i]**2
                vote_count[private_label_list[neighbor]] += min(mask_idx[neighbor], normalized_weight[i])
        elif hash_method == 'knn':
            top_k_index = np.argsort(-kernel_weight)[:nb_teachers]
            for i in range(len(top_k_index)):
                neighbor = select_neighbors[top_k_index[i]]
                vote_count[private_label_list[neighbor]] += min(mask_idx[neighbor], normalized_weight[top_k_index[i]])
            for i in range(len(normalized_weight)):
                neighbor = select_neighbors[i]
                mask_idx[neighbor] -= normalized_weight[i]**2
        elif hash_method == 'knn+norm':
            top_k_index = np.argsort(-kernel_weight)[:nb_teachers]
            sum_weight = sum(normalized_weight)
            normalized_weight = [x / sum_weight for x in normalized_weight]
            for i in range(len(top_k_index)):
                neighbor  = select_neighbors[top_k_index[i]]
                vote_count[private_label_list[neighbor]]+= min(np.sqrt(mask_idx[neighbor]),normalized_weight[top_k_index[i]])
            for i in range(len(normalized_weight)):
                neighbor = select_neighbors[i]
                mask_idx[neighbor]-=  normalized_weight[i]**2
        elif hash_method =='basic+threshold':
            keep_idx = np.where(kernel_weight>min_weight)[0]
            select_neighbors = select_neighbors[keep_idx]
            for i in range(len(select_neighbors)):
                neighbor  = select_neighbors[i]
                vote_count[private_label_list[neighbor]]+= min(np.sqrt(mask_idx[neighbor]),normalized_weight[keep_idx[i]])
                mask_idx[neighbor]-=  normalized_weight[keep_idx[i]]**2
        else:
            print('wrong approach')
        # print(f'max vote_count is{max(vote_count)} sum vote count is {sum(vote_count)}')

        sum_neighbors += len(select_neighbors)
        for i in range(nb_labels):
            vote_count[i] += np.random.normal(scale=noisy_scale)

        # sum over the number of teachers, which make it easy to compute their votings

        predict_labels.append(np.argmax(vote_count))
    predict_labels = np.array(predict_labels)
    print(f'averaged neighbors before knn is around{sum_neighbors / (len(predict_labels))}')
    accuracy = metrics.accuracy(predict_labels, query_label_list)
    return accuracy * 100


if __name__ == '__main__':
    """
    This function trains a student using predictions made by an ensemble of
    teachers. The student and teacher models are trained using the same
    neural network architecture.
    :param dataset: string corresponding to celeba
    :param nb_teachers: number of teachers (in the ensemble) to learn from
    :return: True if student training went well
    """
    # Call helper function to prepare student data using teacher predictions
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cifar10'], default='cifar10')
    parser.add_argument('--feature', choices=['resnet50', 'resnext29'], default='resnet50')
    parser.add_argument('--noisy_scale', type=float, default=0.1)
    parser.add_argument('--ind_budget', type=float, default=20.)
    parser.add_argument('--nb_teachers', type=int, default=150)
    parser.add_argument('--nb_labels', type=int, default=10)
    parser.add_argument('--num_query', type=int, default=1000)
    parser.add_argument('--var', type=float, default=8.)
    parser.add_argument('--norm', choices=['L2', 'cos'], default='L2')
    args = parser.parse_args()
    ac_labels = IndividualkNN(**vars(args))
    # return ac_labels
