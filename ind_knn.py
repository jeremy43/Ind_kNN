from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import datasets as dataset
from datasets import load_dataset
import numpy as np
import torch
from torch.utils.data import DataLoader
import utils
import sys
import metrics
import argparse
from utils import extract_feature, PrepareData
from sentence_transformers import util
import autodp
from autodp.mechanism_zoo import GaussianMechanism
from autodp.calibrator_zoo import ana_gaussian_calibrator



def IndividualkNN(dataset, kernel_method='cosine', feature='resnet50', num_query=1000, nb_labels=10, ind_budget=20, min_weight=0.1, sigma_2=0.1, sigma_1=0.1, seed=0, var=1.,  opt_public=False,  dataset_path=None):
    # mask_idx masked private data that are deleted.  only train_data[mask_idx!=0] will be used for kNN.
    private_data_list, private_label_list, query_data_list, query_label_list = PrepareData(dataset, feature, num_query, dataset_path, seed)
    print(f'length of query list={len(query_data_list)}')
    print('shape of feature', private_data_list.shape)
    mask_idx = np.ones(len(private_data_list)) * ind_budget
    public_mask = np.zeros(len(query_data_list))

    # pointer to original idx.
    original_idx = np.array([x for x in range(len(private_data_list))])
    # keep_idx denote the data that is not deleted.
    sum_neighbors = 0

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
        # to speed up the experiment, only keep the top 5k neighbors' prediction.
        num_data.append(len(keep_idx))
        keep_idx =keep_idx[np.argsort(dis)[:5000]] 
        if len(keep_idx) == 0 or len(keep_idx) == 1:
            print('private dataset is now empty')
            predict_labels.append(0)
            continue
        if kernel_method == 'cosine':
            kernel_weight = np.dot(private_data_list[keep_idx], query_data)
            public_kernel_weight = np.dot(query_data_list, query_data)
        elif kernel_method == 'RBF':
            kernel_weight = [np.exp(-np.linalg.norm(private_data_list[x] - query_data) ** 2 / var) for x in keep_idx]
            public_kernel_weight = [np.exp(-np.linalg.norm(query_data_list[x] - query_data) ** 2 / var) for x in range(num_query)]
        elif kernel_method == 'student':
            kernel_weight = [(1 + np.linalg.norm(private_data_list[i] - query_data) ** 2 / var) ** (-(var + 1) / 2) for i in keep_idx]
        if len(keep_idx) == 0 or len(keep_idx) == 1:
            print('private dataset is empty')
            predict_labels.append(0)
            continue
        kernel_weight = np.array(kernel_weight)
        # print(f' min of weight is {min(kernel_weight)} and max weight is {max(kernel_weight)}')
        normalized_weight = np.array(kernel_weight)
        # when opt_public is true, we calculate kernel weight for available public data.
        public_kernel_weight = np.array(public_kernel_weight)
        keep_idx_in_normalized = np.where(normalized_weight > min_weight)[0]
        n_neighbor = len(keep_idx_in_normalized)
        n_neighbor = max(n_neighbor + np.random.normal(scale=sigma_1), 30)
        rescale_noise = np.sqrt(n_neighbor) * sigma_2
        original_top_index_set = keep_idx[keep_idx_in_normalized]
        sum_neighbors += len(original_top_index_set)
        vote_count = np.zeros(nb_labels)
        if len(original_top_index_set) == 0:
            predict_labels.append(0)
            continue
        # Calculate individual contribution of release the size of neighbor set.
        n_neighbor_contrib = 1.0/(2*sigma_1**2)
        # Count privacy loss: loss of releasing num of neighbor and the loss of releasing kernel weight
        for i in range(len(original_top_index_set)):
            select_idx = original_top_index_set[i]
            idx_normalized = keep_idx_in_normalized[i]
            mask_idx[select_idx]-=n_neighbor_contrib
            # Filter out inactive data.
            if mask_idx[select_idx]<=0:
                continue
            rescale_contrib = normalized_weight[idx_normalized] ** 2 / (2 * rescale_noise ** 2)
            vote_count[private_label_list[select_idx]] += min(np.sqrt(2 * mask_idx[select_idx] * (rescale_noise ** 2)), normalized_weight[idx_normalized])
            mask_idx[select_idx] -= rescale_contrib
        # print(f'max vote count is {max(vote_count)} and noise scale is {noisy_scale}')
        
        if opt_public is True:
            keep_idx_in_public = np.where((public_mask>0) & (public_kernel_weight>min_weight))[0]
            #if (len(keep_idx_in_public)>100):
            #    print(f'using {len(keep_idx_in_public)} public data for prediction')
            for i in keep_idx_in_public:
                vote_count[predict_labels[i]]+=public_kernel_weight[i]
                
        for i in range(nb_labels):
            vote_count[i] += np.random.normal(scale=rescale_noise)
        # Assign an infinite budget to public data.
        public_mask[idx] = 1000
        predict_labels.append(np.argmax(vote_count))
        if idx%200==0 and idx>0:
            accuracy = metrics.accuracy(predict_labels, query_label_list[:idx+1])
            print(f'seed is {seed} test accuracy when num of query is {idx} is {accuracy}')
            print('current dataset size is', num_data[-1])
    print('remain dataset size is', num_data[-1])
    print('averaged neighbors is {}'.format(sum_neighbors / num_query))
    predict_labels = np.array(predict_labels)
    accuracy = metrics.accuracy(predict_labels, query_label_list)
    print('answer {} queries over {} accuracy is {}'.format(len(predict_labels), num_query, accuracy))
    return num_data, accuracy * 100


if __name__ == '__main__':
    """
    The implementation of Ind-kNN algorithm.

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
    sigma_1 = np.sqrt(NUM_QUERY / (6 * ind_budget))
    # The best hyper-parameter of sigma_2 is usually between [0.2, 0.9]
    sigma_2 = 0.5
    min_weight = 0.26
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cifar10', 'agnews', 'fmnist', 'dbpedia'], default='cifar10')
    parser.add_argument('--feature', choices=['resnet50', 'vit',  'all-roberta-large-v1'], default='vit')
    parser.add_argument('--sigma_2', type=float, default=sigma_2, help = 'noise level to perturb the prediction')
    parser.add_argument('--sigma_1', type=float, default=sigma_1, help = 'noise level to perturb the size of neighbors')
    parser.add_argument('--ind_budget', type=float, default=ind_budget, help='each private data participates prediciton unless budget is zero')
    parser.add_argument('--min_weight', type=float, default=min_weight)
    parser.add_argument('--nb_labels', type=int, default=10)
    parser.add_argument('--num_query', type=int, default=1000)
    parser.add_argument('--var', type=float, default=np.exp(1.5), help='RBF kernel bandwidth (not used in cosine kernel')
    parser.add_argument('--opt_public', action='store_true', default=False, help='when true, re-use released predictions as public information')
    parser.add_argument('--dataset_path', type=str, default='./dataset/')
    parser.add_argument('--kernel_method', type=str, default='cosine')
    args = parser.parse_args()
    ac_labels = IndividualkNN(**vars(args))
    # return ac_labels
