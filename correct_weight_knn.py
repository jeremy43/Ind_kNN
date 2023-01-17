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


def PrepareData(dataset, feature, num_query, dataset_path, seed, norm=None):
    """
    Takes a dataset name and the size of the teacher ensemble and prepares
    training data for the student model, according to parameters indicated
    in flags above.
    :param dataset: string corresponding to mnist, cifar10, or svhn
    :param nb_teachers: number of teachers (in the ensemble) to learn from
    :param save: if set to True, will dump student training labels predicted by
                 the ensemble of teachers (with Laplacian noise) as npy files.
                 It also dumps the clean votes for each class (without noise) and
                 the labels assigned by teachers
    :return: pairs of (data, labels) to be used for student training and testing

    """

    # Load the dataset

    if feature == 'resnet50':
        weight = ResNet50_Weights.IMAGENET1K_V2
        preprocess = weight.transforms()
        if dataset == 'cifar10':
            train_dataset = datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transforms.Compose(
                [transforms.ToTensor(), preprocess]
            ))
            test_dataset = datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transforms.Compose(
                [transforms.ToTensor(), preprocess]
            ))
            test_labels = test_dataset.targets
            train_labels = train_dataset.targets
        elif dataset == 'fmnist':
              
            train_dataset = datasets.FashionMNIST(root=dataset_path, train=True, download=True, transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3, 1, 1)), preprocess]
            ))
            test_dataset = datasets.FashionMNIST(root=dataset_path, train=False, download=True, transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3, 1, 1)), preprocess]
            ))
            test_labels = test_dataset.targets
            train_labels = train_dataset.targets
        elif dataset == 'INaturalist':
            train_dataset = datasets.INaturalist(root=dataset_path, version='2021_train_mini', transform=transforms.Compose([transforms.ToTensor(), preprocess]
                                                                                                                            ))
            print('train_dataset', len(train_dataset))
            # test_dataset = train_dataset
            test_dataset = datasets.INaturalist(root=dataset_path + '/val', version='2021_valid', transform=transforms.Compose(
                [transforms.ToTensor(), preprocess]
            ))
            train_labels = utils.extract_label(train_dataset, 'train_mini')
            test_labels = utils.extract_label(test_dataset, 'val')
        elif dataset == 'cifar100':
            train_dataset = datasets.CIFAR100(root=dataset_path, train=True, download=True, transform=transforms.Compose(
                [transforms.ToTensor(), preprocess]
            ))
            test_dataset = datasets.CIFAR100(root=dataset_path, train=False, download=True,
                                             transform=transforms.Compose(
                                                 [transforms.ToTensor(), preprocess]
                                             ))
            test_labels = test_dataset.targets
            train_labels = train_dataset.targets
    elif feature == 'resnext29':
        normalize = transforms.Normalize(mean=[0.491, 0.482, 0.4465],
                                         std=[0.202, 0.1994, 0.2010])
        train_dataset = datasets.CIFAR10(root=dataset_path, train=True, download=True,
                                         transform=transforms.Compose(
                                             [transforms.ToTensor(), normalize]
                                         ))
        test_dataset = datasets.CIFAR10(root=dataset_path, train=False, download=True,
                                        transform=transforms.Compose(
                                            [transforms.ToTensor(), normalize]
                                        ))
        test_labels = test_dataset.targets
        train_labels = train_dataset.targets
    elif feature == 'all-roberta-large-v1':
        if dataset == 'sst2':
            ori_dataset = load_dataset('glue', 'sst2')
            train_dataset = ori_dataset['train']['sentence']
            test_dataset = ori_dataset['test']['sentence']
            train_labels = ori_dataset['train']['label']
            # test_labels = ori_dataset['validation']['label']
            test_labels = ori_dataset['test']['label']
        elif dataset == 'agnews':
            ori_dataset = load_dataset('ag_news')
            train_dataset = ori_dataset['train']['text']
            test_dataset = ori_dataset['test']['text']
            train_labels = ori_dataset['train']['label']
            test_labels = ori_dataset['test']['label']
        elif dataset == 'dbpedia':
            ori_dataset = load_dataset('dbpedia_14')
            train_dataset = ori_dataset['train']['content']
            test_dataset = ori_dataset['test']['content']
            train_labels = ori_dataset['train']['label']
            test_labels = ori_dataset['test']['label']
    elif feature == 'resnet29':
        normalize = transforms.Normalize(mean=[0.491, 0.482, 0.4465],
                                         std=[0.202, 0.1994, 0.2010])

        train_dataset = datasets.CIFAR10(root=dataset_path, train=True, download=True,
                                         transform=transforms.Compose(
                                             [transforms.ToTensor(), normalize]
                                         ))
        test_dataset = datasets.CIFAR10(root=dataset_path, train=False, download=True,
                                        transform=transforms.Compose(
                                            [transforms.ToTensor(), normalize]
                                        ))
        test_labels = test_dataset.targets
        train_labels = train_dataset.targets

    train_data, test_data = extract_feature(train_dataset, test_dataset, feature, dataset, norm=norm)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    np.random.seed(seed)
    random_index = np.random.randint(0, test_data.shape[0], num_query).astype(int)
    print('test data size', test_data.shape)
    return train_data, train_labels, test_data[random_index], test_labels[random_index]

    # return train_data, train_labels, test_data, test_labels


def IndividualkNN(dataset, kernel_method='rbf', feature='resnet50', nb_teachers=150, num_query=1000, nb_labels=10, ind_budget=20, min_weight=0.1, noisy_scale=0.1, clip=20, seed=0, var=1., norm='centering+L2', dataset_path=None):
    # mask_idx masked private data that are deleted.  only train_data[mask_idx!=0] will be used for kNN.
    private_data_list, private_label_list, query_data_list, query_label_list = PrepareData(dataset, feature, num_query, dataset_path, seed, norm=norm)
    print(f'length of query list={len(query_data_list)}')
    print('shape of feature', private_data_list.shape)
    mask_idx = np.ones(len(private_data_list)) * ind_budget
    # pointer to original idx.
    original_idx = np.array([x for x in range(len(private_data_list))])
    # keep_idx denote the data that is not deleted.
    max_clip = 0
    sum_neighbors = 0
    teachers_preds = np.zeros([num_query, nb_teachers])
    predict_labels = []
    num_data = []
    for idx in range(num_query):
        query_data = query_data_list[idx]
        if idx % 100 == 0:
            print('current query idx', idx)
        # print(f'idx is {idx}')
        filter_private_data = private_data_list[mask_idx > 0]

        if kernel_method == 'cosine':
            dis = -np.dot(filter_private_data, query_data)
            # dis = -util.cos_sim(filter_private_data, query_data).reshape(-1)
        elif kernel_method == 'RBF' or 'student':
            dis = np.linalg.norm(filter_private_data - query_data, axis=1)
        keep_idx = original_idx[np.where(mask_idx > 0)[0]]
        num_data.append(len(keep_idx))
        # to speed up the experiment, only keep the top 3k neighbors' prediction.
        keep_idx =keep_idx[np.argsort(dis)[:5000]] 
        #print(f'length of keep_idx is {len(keep_idx)}')
        if len(keep_idx) == 0 or len(keep_idx) == 1:
            print('private dataset is now empty')
            predict_labels.append(0)
            continue
        if kernel_method == 'cosine':
            kernel_weight = np.dot(private_data_list[keep_idx], query_data)
            # temp_d = util.cos_sim(private_data_list[keep_idx], query_data).reshape(-1)
            # kernel_weight = [np.exp(-temp_d[i] ** 2 / var) for i in range(len(keep_idx)) ]
        elif kernel_method == 'RBF':
            kernel_weight = [np.exp(-np.linalg.norm(private_data_list[x] - query_data) ** 2 / var) for x in keep_idx]
        # print(f'length of keep_idx is {len(keep_idx)}')
        elif kernel_method == 'student':
            kernel_weight = [(1 + np.linalg.norm(private_data_list[i] - query_data) ** 2 / var) ** (-(var + 1) / 2) for i in keep_idx]
        if len(keep_idx) == 0 or len(keep_idx) == 1:
            print('private dataset is empty')
            predict_labels.append(0)
            continue
        kernel_weight = np.array(kernel_weight)
        # normalized_weight = [x*min(1, clip/x) for x in kernel_weight]
        # track_k_weight.append(kernel_weight[10])
        # print(f' min of weight is {min(kernel_weight)} and max weight is {max(kernel_weight)}')
        # normalized_weight = [x/sum_weight for x in kernel_weight]
        normalized_weight = np.array(kernel_weight)
        keep_idx_in_normalized = np.where(normalized_weight > min_weight)[0]
        n_neighbor = len(keep_idx_in_normalized)
        rescale_noise = n_neighbor * noisy_scale
        # cur_weight = min_weight
        # while len(keep_idx_in_normalized) < 100 and cur_weight > 0:
        #     cur_weight -= 0.1
        #     keep_idx_in_normalized = np.where(normalized_weight > cur_weight)[0]
        # print(f'keep idx ={len(keep_idx_in_normalized)}')
        original_top_index_set = keep_idx[keep_idx_in_normalized]
        # print(f'length of neighbors is {len(original_top_index_set)}')
        sum_neighbors += len(original_top_index_set)
        # original_top_index_set =  keep_idx[np.where(kernel_weight>min_weight)]
        vote_count = np.zeros(nb_labels)
        if len(original_top_index_set) == 0:
            predict_labels.append(0)
            continue
        for i in range(len(original_top_index_set)):
            select_idx = original_top_index_set[i]
            idx_normalized = keep_idx_in_normalized[i]
            vote_count[private_label_list[select_idx]] += min(np.sqrt(mask_idx[select_idx]), normalized_weight[idx_normalized])
            mask_idx[select_idx] -= normalized_weight[idx_normalized]**2
        for i in range(nb_labels):
            vote_count[i] += np.random.normal(scale=noisy_scale)
        predict_labels.append(np.argmax(vote_count))
    print(f'remain dataset size is {num_data[-1]}')
    print('averaged neighbors is {}'.format(sum_neighbors / len(teachers_preds)))
    print('answer {} queries over {}'.format(len(predict_labels), len(teachers_preds)))
    # acct.compose_poisson_subsampled_mechanisms(gaussian2, prob,coeff = len(stdnt_labels))
    predict_labels = np.array(predict_labels)
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
    parser.add_argument('--dataset_path', type=str, default='./dataset/')
    args = parser.parse_args()
    ac_labels = IndividualkNN(**vars(args))
    # return ac_labels
