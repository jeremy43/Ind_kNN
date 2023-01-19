import os
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import datasets as dataset
from correct_weight_knn import PrepareData
from PIL import Image
import numpy as np
import torch
import network
from torch.utils.data import DataLoader
import argparse
import utils
import pickle 
import sys
import os
import metrics


def PrivatekNN(dataset, feature='resnet50', nb_teachers=150, num_query=1000, nb_labels=10, noisy_scale=0.1, sample_rate=0.1, seed=1, dataset_path=None):
    private_data_list, private_label_list, query_data_list, query_label_list = PrepareData(dataset, feature, num_query, dataset_path, seed, norm='centering+L2')
    private_label_list = np.array(private_label_list)
    n = len(private_data_list)
    teachers_preds = np.zeros([num_query, nb_labels])
    predict_labels = []
    prediction_file = f'{dataset}_Private_knn_{5000}_query_seed_{seed}_sampleRate_{sample_rate}_k_{nb_teachers}.pkl'
    print(f'prediction file is {prediction_file}')
    if os.path.exists(prediction_file):
        with open(prediction_file, 'rb') as f:
            record = pickle.load(f)
            teachers_preds = record['preds']
            gt_labels = record['gt_labels']
    else:
        
        for idx in range(num_query):
            vote_count = np.zeros([nb_labels])
            query_data = query_data_list[idx]
            if idx % 100 == 0:
                print('current query idx', idx)

            subset = np.random.choice(n, int(sample_rate*n))
            dis = np.linalg.norm(private_data_list[subset] - query_data, axis=1)
            topk_index_set = subset[np.argsort(dis)[:nb_teachers]]
            #dis = np.linalg.norm(private_data_list - query_data, axis=1)
            #topk_index_set = np.argsort(dis)[:nb_teachers]
            #vote_count = np.zeros(nb_labels)
            for i in range(nb_teachers):
                select_top_k = topk_index_set[i]
                vote_count[private_label_list[select_top_k]] += 1
            teachers_preds[idx,:] = vote_count
        record = {}
        record['preds'] = teachers_preds
        record['gt_labels'] = query_label_list
        with open(prediction_file, 'wb') as f:
            pickle.dump(record, f)
            
    for idx in range(num_query):
        vote_count = teachers_preds[idx,:]
        for i in range(nb_labels):
            vote_count[i] += np.random.normal(scale=noisy_scale)
        predict_labels.append(np.argmax(vote_count))
    
    print('answer {} queries over {}'.format(len(predict_labels), len(teachers_preds)))
    # acct.compose_poisson_subsampled_mechanisms(gaussian2, prob,coeff = len(stdnt_labels))
    predict_labels = np.array(predict_labels)
    accuracy = metrics.accuracy(predict_labels, record['gt_labels'][:num_query])
    return accuracy*100


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
    args = parser.parse_args()
    ac_labels = IndividualkNN(**vars(args))
    # return ac_labels
