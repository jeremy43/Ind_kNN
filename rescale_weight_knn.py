from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import datasets as dataset
from datasets import load_dataset
import numpy as np
import torch
import network
from torch.utils.data import DataLoader
import utils
import sys
import metrics
import argparse
from utils import extract_feature
from sentence_transformers import util
from correct_weight_knn import PrepareData


def IndividualkNN(dataset, kernel_method='rbf', feature='resnet50', nb_teachers=150, num_query=1000, nb_labels=10, ind_budget=20, min_weight=0.1, noisy_scale=0.1, sigma_1=0.1, seed=0, var=1., norm='centering+L2', dataset_path=None):
    # mask_idx masked private data that are deleted.  only train_data[mask_idx!=0] will be used for kNN.
    private_data_list, private_label_list, query_data_list, query_label_list = PrepareData(dataset, feature, num_query, dataset_path, seed, norm=norm)
    print(f'length of query list={len(query_data_list)}')
    print('shape of feature', private_data_list.shape)
    mask_idx = np.ones(len(private_data_list)) * ind_budget
    # pointer to original idx.
    original_idx = np.array([x for x in range(len(private_data_list))])
    # keep_idx denote the data that is not deleted.
    sum_neighbors = 0
    teachers_preds = np.zeros([num_query, nb_teachers])
    predict_labels = []
    num_data = []
    for idx in range(num_query):
        query_data = query_data_list[idx]
        if idx % 100 == 0:
            print('current query idx', idx)
        filter_private_data = private_data_list[mask_idx > 0]

        if kernel_method == 'cosine':
            dis = -np.dot(filter_private_data, query_data)
        elif kernel_method == 'RBF' or 'student':
            dis = np.linalg.norm(filter_private_data - query_data, axis=1)
        keep_idx = original_idx[np.where(mask_idx > 0)[0]]
        # to speed up the experiment, only keep the top 3k neighbors' prediction.
        num_data.append(len(keep_idx))
        #print(f'number of data is {len(keep_idx)}')
        keep_idx =keep_idx[np.argsort(dis)[:6000]] 
        if len(keep_idx) == 0 or len(keep_idx) == 1:
            print('private dataset is now empty')
            predict_labels.append(0)
            continue
        if kernel_method == 'cosine':
            kernel_weight = np.dot(private_data_list[keep_idx], query_data)
        elif kernel_method == 'RBF':
            kernel_weight = [np.exp(-np.linalg.norm(private_data_list[x] - query_data) ** 2 / var) for x in keep_idx]
        elif kernel_method == 'student':
            kernel_weight = [(1 + np.linalg.norm(private_data_list[i] - query_data) ** 2 / var) ** (-(var + 1) / 2) for i in keep_idx]
        if len(keep_idx) == 0 or len(keep_idx) == 1:
            print('private dataset is empty')
            predict_labels.append(0)
            continue
        kernel_weight = np.array(kernel_weight)
        # print(f' min of weight is {min(kernel_weight)} and max weight is {max(kernel_weight)}')
        normalized_weight = np.array(kernel_weight)
        keep_idx_in_normalized = np.where(normalized_weight > min_weight)[0]
        n_neighbor = len(keep_idx_in_normalized)
        n_neighbor = max(n_neighbor + np.random.normal(scale=sigma_1), 30)
        #print('number of neighbor', n_neighbor, 'true num of neighbor', len(keep_idx_in_normalized))
        rescale_noise = np.sqrt(n_neighbor) * noisy_scale
        original_top_index_set = keep_idx[keep_idx_in_normalized]
        sum_neighbors += len(original_top_index_set)
        vote_count = np.zeros(nb_labels)
        if len(original_top_index_set) == 0:
            predict_labels.append(0)
            continue
        n_neighbor_contrib = 1.0/(2*sigma_1**2)
        # count privacy loss: loss of releasing num of neighbor and the loss of releasing kernel weight
        for i in range(len(original_top_index_set)):
            select_idx = original_top_index_set[i]
            idx_normalized = keep_idx_in_normalized[i]
            mask_idx[select_idx]-=n_neighbor_contrib
            if mask_idx[select_idx]<=0:
                continue
            rescale_contrib = normalized_weight[idx_normalized] ** 2 / (2 * rescale_noise ** 2)
            vote_count[private_label_list[select_idx]] += min(np.sqrt(2 * mask_idx[select_idx] * (rescale_noise ** 2)), normalized_weight[idx_normalized])
            mask_idx[select_idx] -= rescale_contrib
        # print(f'max vote count is {max(vote_count)} and noise scale is {noisy_scale}')
        for i in range(nb_labels):
            vote_count[i] += np.random.normal(scale=rescale_noise)
        predict_labels.append(np.argmax(vote_count))
        if idx%200==0 and idx>0:
            tmp_predict_labels = np.array(predict_labels)
            accuracy = metrics.accuracy(predict_labels, query_label_list[:idx+1])
            print(f'seed is {seed} test accuracy when num of query is {idx} is {accuracy}')
            print('current dataset size is', num_data[-1])
    print('remain dataset size is', num_data[-1])
    print('averaged neighbors is {}'.format(sum_neighbors / len(teachers_preds)))
    print('answer {} queries over {}'.format(len(predict_labels), len(teachers_preds)))
    # acct.compose_poisson_subsampled_mechanisms(gaussian2, prob, coeff=len(stdnt_labels))
    predict_labels = np.array(predict_labels)
    accuracy = metrics.accuracy(predict_labels, query_label_list)
    return num_data, accuracy * 100


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
    parser.add_argument('--dataset_path', type=str, default='./dataset/')
    args = parser.parse_args()
    ac_labels = IndividualkNN(**vars(args))
    # return ac_labels
