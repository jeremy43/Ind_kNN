import os

from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import datasets as dataset
from utils import PrepareData
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
import utils
import pickle
import sys
import os
import metrics


def PrivatekNN(dataset, feature='resnet50', nb_teachers=150, num_query=1000, nb_labels=10, noisy_scale=0.1,
               sample_rate=0.2, seed=0, dataset_path=None):
    private_data_list, private_label_list, query_data_list, query_label_list = PrepareData(dataset, feature, num_query,
                                                                                           dataset_path, seed)
    n = len(private_data_list)
    teachers_preds = np.zeros([num_query, nb_labels])
    predict_labels = []
    prediction_file = f'{dataset}_Private_knn_{num_query}_query_seed_{seed}_sampleRate_{sample_rate}_k_{nb_teachers}.pkl'
    print(f'prediction file is {prediction_file}')
    if os.path.exists(prediction_file):
        with open(prediction_file, 'rb') as f:
            record = pickle.load(f)
            teachers_preds = record['preds'][:num_query, :]
    else:

        for idx in range(num_query):
            vote_count = np.zeros([nb_labels])
            query_data = query_data_list[idx]
            subset = np.random.choice(n, int(sample_rate * n))
            # consider either L2 distance or the negative cosine as the distance function.
            # dis = np.linalg.norm(private_data_list[subset] - query_data, axis=1)
            dis = -np.dot(private_data_list[subset], query_data)
            topk_index_set = subset[np.argsort(dis)[:nb_teachers]]
            for i in range(nb_teachers):
                select_top_k = topk_index_set[i]
                vote_count[private_label_list[select_top_k]] += 1
            teachers_preds[idx, :] = vote_count
        record = {}
        record['preds'] = teachers_preds
        record['gt_labels'] = query_label_list
        with open(prediction_file, 'wb') as f:
            pickle.dump(record, f)

    for idx in range(num_query):
        vote_count = teachers_preds[idx, :]
        # print('the gap between the largest and the second largest', np.max(vote_count) - np.partition(vote_count, -2)[-2])
        for i in range(nb_labels):
            vote_count[i] += np.random.normal(scale=noisy_scale)
        predict_labels.append(np.argmax(vote_count))

    predict_labels = np.array(predict_labels)
    accuracy = metrics.accuracy(predict_labels, record['gt_labels'][:num_query])
    print('answer {} queries over {} accuracy is {}'.format(len(predict_labels), len(teachers_preds), accuracy))
    return accuracy * 100


if __name__ == '__main__':
    """
    This algorithm implements Private kNN (Private-kNN: Practical Differential Privacy for Computer Vision).
    """
    # Call helper function to prepare student data using teacher predictions
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cifar10', 'fmnist', 'agnews', 'dbpedia'], default='cifar10')
    parser.add_argument('--feature', choices=['resnet50', 'vit', 'clr', 'all-roberta-large-v1'], default='vit',
                        help='Choose the feature extractor')
    parser.add_argument('--noisy_scale', type=float, default=45.1, help='Noise level added to the prediction')
    parser.add_argument('--sample_rate', type=float, default=0.2, help='Sampling probability to select neighbors')
    parser.add_argument('--nb_teachers', type=int, default=150)
    parser.add_argument('--nb_labels', type=int, default=10)
    parser.add_argument('--num_query', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0, help='random seed to select queries')
    parser.add_argument('--dataset_path', type=str, default='./dataset', help='path to the dataset')
    args = parser.parse_args()
    ac_labels = PrivatekNN(**vars(args))
