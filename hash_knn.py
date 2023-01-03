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
from sentence_transformers import util
import utils
import sys
import os
import metrics
from utils import extract_feature


dataset_path = '/home/yq/dataset'


def PrepareData(dataset, feature, num_query, dataset_path):
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
    if dataset == 'mnist':
        normalize = transforms.Normalize(mean=[0.485], std=[0.22])

        train_dataset = datasets.MNIST(root=dataset_path, train=False, download=True,
                                    transform=transforms.Compose(
                                        [transforms.ToTensor(), normalize]
                                    ))
        test_dataset = datasets.MNIST(root=dataset_path, train=False, download=True,
                                     transform=transforms.Compose(
                                        [transforms.ToTensor(), normalize]
                                    ))
        scattering, K, (h, w) = utils.get_scatter_transform()
        train_data, test_data = extract_feature(train_dataset, test_dataset)
        test_labels = test_dataset.targets
        train_labels = train_dataset.targets
        return train_data, train_labels, test_data, test_labels
    
    if feature == 'resnet50':
        weight = ResNet50_Weights.IMAGENET1K_V2
        preprocess = weight.transforms()
        if dataset == 'cifar10':
            train_dataset = datasets.CIFAR10(root=dataset_path, train=True, download=True,  transform=transforms.Compose(
                                        [transforms.ToTensor(), preprocess]
                                    ))
            test_dataset = datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transforms.Compose(
                                           [transforms.ToTensor(), preprocess]
                                       ))
            test_labels = test_dataset.targets
            train_labels = train_dataset.targets
        elif dataset == 'INaturalist':
            train_dataset = datasets.INaturalist(root=dataset_path, version='2021_train_mini',  transform=transforms.Compose([transforms.ToTensor(), preprocess]
                                    ))
            test_dataset = datasets.INaturalist(root=dataset_path+'/val', version='2021_valid', transform=transforms.Compose(
                
                                           [transforms.ToTensor(), preprocess]
                                       ))
            train_labels = utils.extract_label(train_dataset, 'train_mini')
            test_labels = utils.extract_label(test_dataset, 'val')

        elif dataset == 'LSUN':
            train_dataset = datasets.LSUN(root=dataset_path, classes='train', transform=transforms.Compose(
                                        [transforms.ToTensor(), preprocess]
                                    ))
            test_dataset = datasets.LSUN(root=dataset_path, classes='test', 
                                       transform=transforms.Compose(
                                           [transforms.ToTensor(), preprocess]
                                       ))
        elif dataset == 'cifar100':
            train_dataset = datasets.CIFAR100(root=dataset_path, train=True, download=True,transform=transforms.Compose(
                                        [transforms.ToTensor(), preprocess]
                                    ))
            test_dataset = datasets.CIFAR100(root=dataset_path, train=False, download=True,
                                       transform=transforms.Compose(
                                           [transforms.ToTensor(), preprocess]
                                       ))
    elif feature == 'resnext29':
        normalize = transforms.Normalize(mean=[0.491, 0.482, 0.4465],
                                         std=[0.202, 0.1994, 0.2010])
        
        train_dataset = datasets.CIFAR10(root=dataset_path, train=True, download=True,transform=transforms.Compose(
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
            #test_labels = ori_dataset['validation']['label']
            test_labels = ori_dataset['test']['label']
        elif dataset == 'agnews':
            ori_dataset = load_dataset('ag_news')
            train_dataset = ori_dataset['train']['text']
            test_dataset = ori_dataset['test']['text']
            train_labels = ori_dataset['train']['label']
            test_labels = ori_dataset['test']['label']
    train_data, test_data = extract_feature(train_dataset, test_dataset, feature, dataset) 
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    train_data = np.array(train_data)
    np.random.seed(0)
    print(f'length of test data {test_data.shape[0]}')
    random_index = np.random.randint(0, test_data.shape[0], num_query).astype(int)
    return train_data, train_labels, test_data[random_index], test_labels[random_index]


def IndividualkNN(dataset,  hash_method = 'knn', min_weight = 0.2,  num_tables = 2, proj_dim=12,feature='resnet50',clip=0.25, nb_teachers= 50, num_query=1000, nb_labels=10, ind_budget=20, noisy_scale=0.1, var=1., norm='L2', dataset_path=None):
    # mask_idx masked private data that are deleted.  only train_data[mask_idx!=0] will be used for kNN.
    print('which norm', norm)
    private_data_list, private_label_list, query_data_list, query_label_list = PrepareData(dataset, feature, num_query, dataset_path)
    print('shape of feature', private_data_list.shape)
    # construct hash table
    print(f'num_tables is {num_tables} and proj_dim is {proj_dim}')
    hash_path = f'hash_table/{dataset}_num_tables{num_tables}_projdim{proj_dim}.pkl'
    if os.path.exists(hash_path):

        with open(hash_path, 'rb') as f:
            lsh_hash =pickle.load(f)
    else:
        lsh_hash = LSH(num_tables, proj_dim, private_data_list.shape[1])
        for (idx, x) in enumerate(private_data_list):
            if idx%10000==0:
                print(f'prepare hash table for idx={idx}')
            lsh_hash[x] = idx
        with open(hash_path, 'wb') as f:
            pickle.dump(lsh_hash, f)
    
    mask_idx = np.ones(len(private_data_list)) * ind_budget
    # pointer to original idx.
    private_label_list = np.array(private_label_list)
    predict_labels = []
    sum_neighbors = 0
    for idx in range(num_query):
        query_data = query_data_list[idx]
        if idx % 100 ==0:
            print('current query idx', idx)

        #filter_private_data  = private_data_list[mask_idx>0]
        #print('len of filter_private_data', len(filter_private_data))
        hash_neighbors = lsh_hash.query_neighbor(query_data)
        #print(f'length of hash neighbors is {len(hash_neighbors)}')
        #print('select_neighbor length', len(select_neighbor))
        keep_idx = np.where(mask_idx > 0)[0]
        #print(f"argsort={keep_idx}")
        select_neighbors = [x for x in hash_neighbors if mask_idx[x]>0]
        select_neighbors = np.array(select_neighbors, dtype=int)
        sum_neighbors+=len(select_neighbors)
        vote_count = np.zeros(nb_labels)
        #select_neighbors = np.array(select_neighbors)
        #temp_d = util.cos_sim(private_data_list[select_neighbors], query_data).reshape(-1)
        #kernel_weight = [np.exp(-temp_d[i] ** 2 / var) for i in range(len(select_neighbors))]
        if dataset in {'sst2', 'agnews'}:
            temp_d = util.cos_sim(private_data_list[select_neighbors], query_data).reshape(-1)
            kernel_weight = [np.exp(-temp_d[i] ** 2 / var) for i in range(len(select_neighbors))]
        else:
            kernel_weight = [np.exp(-np.linalg.norm(private_data_list[x] - query_data) ** 2 / var) for x in select_neighbors]
        #print(f'length of kernel_weight {len(kernel_weight)}')
        normalized_weight = [x*min(1, clip/x) for x in kernel_weight]
        kernel_weight = np.array(kernel_weight)
        nb_teachers = len(kernel_weight)
        if hash_method == 'basic':
            for i in range(len(select_neighbors)):
                neighbor  = select_neighbors[i]
                vote_count[private_label_list[neighbor]]+= min(mask_idx[neighbor],normalized_weight[i])
                mask_idx[neighbor]-=  normalized_weight[i]
        elif hash_method == 'basic+norm':
            sum_weight = sum(normalized_weight)
            normalized_weight = [x/sum_weight for x in normalized_weight]
            for i in range(len(select_neighbors)):
                neighbor  = select_neighbors[i]
                vote_count[private_label_list[neighbor]]+= min(mask_idx[neighbor],normalized_weight[i])
                mask_idx[neighbor]-=  normalized_weight[i]
        elif hash_method =='knn':
            top_k_index = np.argsort(-kernel_weight)[:nb_teachers]
            for i in range(len(top_k_index)):
                neighbor  = select_neighbors[top_k_index[i]]
                vote_count[private_label_list[neighbor]]+= min(mask_idx[neighbor],normalized_weight[top_k_index[i]])
            for i in range(len(normalized_weight)):
                neighbor = select_neighbors[i]
                mask_idx[neighbor]-=  normalized_weight[i]
        elif hash_method == 'knn+norm':
            top_k_index = np.argsort(-kernel_weight)[:nb_teachers]
            sum_weight = sum(normalized_weight)
            normalized_weight = [x/sum_weight for x in normalized_weight]
            for i in range(len(top_k_index)):
                neighbor  = select_neighbors[top_k_index[i]]
                vote_count[private_label_list[neighbor]]+= min(mask_idx[neighbor],normalized_weight[top_k_index[i]])
            for i in range(len(normalized_weight)):
                neighbor = select_neighbors[i]
                mask_idx[neighbor]-=  normalized_weight[i]
        elif hash_method =='basic+threshold':
            select_neighbor = select_neighbor[np.where(kernel_weight>min_weight)[0]]
            for i in range(len(select_neighbors)):
                neighbor  = select_neighbors[i]
                vote_count[private_label_list[neighbor]]+= min(mask_idx[neighbor],normalized_weight[i])
                mask_idx[neighbor]-=  normalized_weight[i]
        else:
            print('wrong approach')
        # print(f'max vote_count is{max(vote_count)} sum vote count is {sum(vote_count)}')
        
        
        for i in range(nb_labels):
            vote_count[i]+=np.random.normal(scale=noisy_scale)
        
        # sum over the number of teachers, which make it easy to compute their votings

        predict_labels.append(np.argmax(vote_count))
    predict_labels = np.array(predict_labels)
    print(f'averaged neighbors before knn is around{sum_neighbors/(len(predict_labels))}')
    accuracy = metrics.accuracy(predict_labels, query_label_list)
    return accuracy*100



if __name__ =='__main__':
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
    ac_labels  = IndividualkNN(**vars(args))
    #return ac_labels


