import os
import pickle
from hash import LSH
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import datasets as dataset
from PIL import Image
import autodp
from autodp.mechanism_zoo import GaussianMechanism
from autodp.calibrator_zoo import ana_gaussian_calibrator
import numpy as np
from datasets import load_dataset
import torch
import network
from torch.utils.data import DataLoader
from sentence_transformers import util
import utils
import sys
sys.path.append('../')
import os
import metrics
from utils import extract_feature, PrepareData
from numpy import linalg as LA
import argparse


def IndKNNwithHash(dataset, kernel_method='RBF', seed=0, min_weight=0.2, num_tables=2, proj_dim=8, feature='resnet50',
                   sigma_1=0.1, num_query=1000, nb_labels=10, ind_budget=20, sigma_2=0.1, var=1., dataset_path=None):

    private_data_list, private_label_list, query_data_list, query_label_list = PrepareData(dataset, feature, num_query,
                                                                                           dataset_path, seed)
    print('shape of feature', private_data_list.shape)
    print(f'the second data norm is {LA.norm(private_data_list[0])}')
    # construct hash table
    print(f'num_tables is {num_tables} and proj_dim is {proj_dim}')
    hash_path = f'{dataset}_num_tables{num_tables}_projdim{proj_dim}.pkl'
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
    # mask_idx masked private data that are deleted.  only train_data[mask_idx!=0] will be used for kNN.
    mask_idx = np.ones(len(private_data_list)) * ind_budget
    private_label_list = np.array(private_label_list)
    predict_labels = []
    sum_neighbors = 0
    for idx in range(num_query):
        query_data = query_data_list[idx]
        if idx % 100 == 0 and idx > 0:
            print('current query idx', idx)
            print(f'remain dataset size is {len(np.where(mask_idx > 0)[0])}')
        hash_neighbors = lsh_hash.query_neighbor(query_data)
        select_neighbors = [x for x in hash_neighbors if mask_idx[x] > 0]
        select_neighbors = np.array(select_neighbors, dtype=int)
        # sum_neighbors+=len(select_neighbors)
        vote_count = np.zeros(nb_labels)
        if kernel_method == 'cosine':
            kernel_weight = np.dot(private_data_list[select_neighbors], query_data)
        elif kernel_method == 'RBF':
            kernel_weight = [np.exp(-np.linalg.norm(private_data_list[x] - query_data) ** 2 / var) for x in
                             select_neighbors]
        elif kernel_method == 'student':
            kernel_weight = [(1 + np.linalg.norm(private_data_list[x] - query_data) ** 2 / var) ** (-(var + 1) / 2) for
                             x in select_neighbors]
        normalized_weight = kernel_weight
        kernel_weight = np.array(kernel_weight)

        keep_idx = np.where(kernel_weight > min_weight)[0]
        select_neighbors = select_neighbors[keep_idx]
        n_neighbor = max(len(select_neighbors) + np.random.normal(scale=sigma_1), 30)
        rescale_noise = np.sqrt(n_neighbor) * sigma_2
        for i in range(len(select_neighbors)):
            neighbor = select_neighbors[i]
            mask_idx[neighbor] -= 1. / (2 * sigma_1 ** 2)
            vote_count[private_label_list[neighbor]] += min(np.sqrt(2 * mask_idx[neighbor]) * rescale_noise,
                                                            normalized_weight[keep_idx[i]])
            mask_idx[neighbor] -= normalized_weight[keep_idx[i]] ** 2 / (2 * rescale_noise ** 2)

        sum_neighbors += len(select_neighbors)

        for i in range(nb_labels):
            vote_count[i] += np.random.normal(scale=rescale_noise)

        # sum over the number of teachers, which make it easy to compute their votings

        predict_labels.append(np.argmax(vote_count))
    predict_labels = np.array(predict_labels)
    accuracy = metrics.accuracy(predict_labels, query_label_list)
    print(f'averaged neighbors before knn is around{sum_neighbors / (len(predict_labels))}, accuracy is {accuracy}')
    return accuracy * 100


dataset_path = '/home/yq/dataset'
if __name__ == '__main__':
    """
    The Ind-kNN algorithm with hashing extension. We provide an example when answering T=1000 queries with eps=2.0.
    """
    NUM_QUERY = 1000
    eps = 2.0
    delta = 1e-5
    ana_calibrate = ana_gaussian_calibrator()
    mech = ana_calibrate(GaussianMechanism, eps, delta)
    # Calibrate the noise multiplier such that it achieves (eps, delta)-DP.
    noise_mul = mech.params['sigma']
    ind_budget = 1.0 / (2 * noise_mul ** 2)
    # set sigma_1 = sqrt(T/6B) according to the algorithm
    sigma_1 = np.sqrt(NUM_QUERY / (6*ind_budget))
    # The best hyper-parameter of sigma_2 is usually between [0.2, 0.9]
    sigma_2 = 0.5
    parser = argparse.ArgumentParser()
    parser.add_argument('--sigma_2', type=float, default=sigma_2, help='noise level to perturb the prediction')
    parser.add_argument('--sigma_1', type=float, default=sigma_1, help='noise level to perturb the size of neighbors')
    parser.add_argument('--ind_budget', type=float, default=ind_budget,
                        help='each private data participates prediction unless budget is zero')
    parser.add_argument('--min_weight', type=float, default=0.25)
    parser.add_argument('--nb_labels', type=int, default=10)
    parser.add_argument('--num_query', type=int, default=1000)
    parser.add_argument('--num_tables', type=int, default=30, help='number of hashing tables')
    parser.add_argument('--proj_dim', type=int, default=8, help='bucket size in each hashing table')
    parser.add_argument('--var', type=float, default=np.exp(1.5),
                        help='RBF kernel bandwidth (not used in cosine kernel)')
    parser.add_argument('--dataset', choices=['cifar10', 'agnews'], default='cifar10')
    parser.add_argument('--feature', choices=['resnet50','vit'], default='vit')
    parser.add_argument('--dataset_path', default='./dataset')
    parser.add_argument('--kernel_method', choices=['RBF', 'cosine'], default='cosine')
    args = parser.parse_args()
    ac_labels = IndKNNwithHash(**vars(args))

