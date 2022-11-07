import os
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import datasets as dataset
from PIL import Image
import numpy as np
import torch
import network
from torch.utils.data import DataLoader
import utils
import sys
import os
import metrics
from utils import extract_feature


dataset_path = '/home/yq/dataset'


def PrepareData(dataset, feature, num_query):
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
        elif dataset == 'INaturalist':
            train_dataset = datasets.INaturalist(root=dataset_path, version='2021_train_mini',   transform=transforms.Compose([transforms.ToTensor(), preprocess]
                                    ))
            print('train_dataset', len(train_dataset))
            test_dataset = train_dataset
            #test_dataset = datasets.INaturalist(root=dataset_path, version='2021_valid',download=True,  transform=transforms.Compose(
            #                               [transforms.ToTensor(), preprocess]
            #                           ))
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
        
         
    train_data, test_data = extract_feature(train_dataset, test_dataset, feature, dataset)
    idx = 0
    if dataset == 'INaturalist':
        train_labels = []
        for (img, target) in train_dataset:
            #print('target', target)
            idx+=1
            if idx%1000==0:
                print(f"idx is {idx}")
            train_labels.append(target)
        test_labels = []
        print('idx', idx)
        for (img, target) in test_dataset:
            test_labels.append(target)
    else:
        test_labels = test_dataset.targets
        #print('test_labels', test_labels[:100])
        train_labels = train_dataset.targets
    return train_data, train_labels, test_data[:num_query], test_labels[:num_query]


def IndividualkNN(dataset, feature='resnet50', nb_teachers=150, num_query=1000, nb_labels=10, ind_budget=20, noisy_scale=0.1, var=1., norm='L2'):
    # mask_idx masked private data that are deleted.  only train_data[mask_idx!=0] will be used for kNN.
    print('which norm', norm)
    private_data_list, private_label_list, query_data_list, query_label_list = PrepareData(dataset, feature, num_query)
    print('shape of feature', private_data_list.shape)
    mask_idx = np.ones(len(private_data_list)) * ind_budget
    # pointer to original idx.
    private_label_list = np.array(private_label_list)
    original_idx = np.array([x for x in range(len(private_data_list))])
    print('noisy_scale', noisy_scale)
    # keep_idx denote the data that is not deleted.
    #  private_data = torch.stack(private_data)
    teachers_preds = np.zeros([num_query, nb_teachers])
    predict_labels = []
    for idx in range(num_query):
        query_data = query_data_list[idx]
        if idx % 100 ==0:
            print('current query idx', idx)

        filter_private_data  = private_data_list[mask_idx>0]
        # Implement no sampling strategy first
        #select_teacher = np.random.choice(private_data.shape[0], int(prob * num_train))
        if norm == 'L2':
            dis = np.linalg.norm(filter_private_data - query_data, axis=1)
        else:
            # compute cos similarity 
            dis = [np.dot(x, query_data)/(np.linalg.norm(x)*np.linalg.norm(query_data)) for x in filter_private_data]
            dis = -np.array(dis)
            #dis = -np.dot(filter_private_data, query_data)/(np.linalg.norm(filter_private_data, axis=1)*np.linalg.norm(query_data))
        keep_idx = original_idx[np.where(mask_idx > 0)[0]]
        #print(f"argsort={keep_idx}")
        original_topk_index_set = keep_idx[np.argsort(dis)[:nb_teachers]]
        #print(f"original_topk_index_set={original_topk_index_set}")
        # For each data in original_tok_index, update their individual accountant.
        if norm == 'L2':
            kernel_weight = [np.exp(-np.linalg.norm(private_data_list[i]-query_data)**2/var) for i in original_topk_index_set]
        else:
            kernel_weight = [np.dot(private_data_list[i], query_data)/(np.linalg.norm(private_data_list[i])*np.linalg.norm(query_data)) for i in original_topk_index_set]
        #print('kernel weight', kernel_weight[:10])
        #copy_kernel_weight = [np.exp(-dis[i]**2/var) for i in np.argsort(dis)[:nb_teachers]]
        kernel_weight = np.asarray(kernel_weight)
        sum_kernel_weight = sum(kernel_weight)
        #print('sum_kernel_weight', sum_kernel_weight)
        normalized_weight = kernel_weight#/ sum_kernel_weight * nb_teachers
        vote_count = np.zeros(nb_labels)
        #print('normalized_weight', normalized_weight)
        #print('vote_count', vote_count)
        for i in range(len(original_topk_index_set)):
            select_top_k = original_topk_index_set[i]
            mask_idx[select_top_k]-=normalized_weight[i]
            #mask_idx[select_top_k]-= 1
            #print('norm', normalized_weight[i])
            #vote_count[private_label_list[select_top_k]]+=1
            vote_count[private_label_list[select_top_k]]+=normalized_weight[i]
        #print('vote count', vote_count)
        for i in range(nb_labels):
            vote_count[i]+=np.random.normal(scale=noisy_scale)
        
        # sum over the number of teachers, which make it easy to compute their votings

        #if len(original_topk_index_set)<nb_teachers:
        #    vote_count[0]+=nb_teachers - len(original_topk_index_set)
        #print('predict label', np.argmax(vote_count))
        #print('gt label', query_label_list[idx])
        # apply Report-Noisy-Max for each public query.
        predict_labels.append(np.argmax(vote_count))
    print('answer {} queries over {}'.format(len(predict_labels),len(teachers_preds)))
    #acct.compose_poisson_subsampled_mechanisms(gaussian2, prob,coeff = len(stdnt_labels))
    predict_labels = np.array(predict_labels)
    accuracy = metrics.accuracy(predict_labels, query_label_list)
    return accuracy



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


