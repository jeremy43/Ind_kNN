from private_knn import PrivatekNN
from hand_dp import hand_dp
from noisy_SGD import noisysgd_dp
import numpy as np
import random
import pickle
"""
Set the Parameters
"""
DATASET_PATH = 'dataset'
NUM_CLASS = 10
NUM_QUERY = 400
DATASET = 'cifar10'
FEATURE = 'resnet50'

#DATASET = 'INaturalist'
#FEATURE = 'all-roberta-large-v1'
#DATASET = 'sst2'
#DATASET = 'fmnist'
EPS_LIST = [1.3**x*0.1 for x in range(12)]
NOISE_MUL_LIST = [679.63, 533.47, 418.59, 328.3, 257.53, 201.99, 158.43, 124.30, 97.57, 76.65, 60.28, 47.48]
NOISE_MUL_LIST_01 = [68, 58.5, 46.5, 37, 30, 24.5, 20, 16.1, 12.5, 10.0, 7.7, 6.0] # for sampling ratio = 0.1
NOISE_MUL_LIST_02 = [140, 110, 89, 71, 57, 45, 36.5, 29, 23, 19, 15.1, 12.1]
num_point = 12


private_knn_file = f'{DATASET}_Private_knn_{NUM_QUERY}_query.pkl'
print(f'file_name is {private_knn_file}')
idx = 0
test_ac_list = []
best_hyper_list = []
record_eps =[]
all_record = []
"""
for idx  in range(3,7):
    best_hyper = {}
    eps = EPS_LIST[idx]
    basic_sigma = NOISE_MUL_LIST_02[idx]
    print('idx ', idx, 'current epsilon', eps)
    noise_mul = NOISE_MUL_LIST[idx]
    optimal_ac = 0
    record_eps.append(idx) 
    for k in [  500]:
        each_ac_list = []
        new_hyper = {'k':k, 'sigma':basic_sigma}
        for seed in [3]:
            print(f"dataset={DATASET}, k={k},  basic_sigma={basic_sigma}.")
            ac = PrivatekNN(dataset=DATASET, noisy_scale=basic_sigma, feature=FEATURE, num_query=NUM_QUERY, nb_teachers=k, nb_labels=NUM_CLASS,seed=seed, dataset_path=DATASET_PATH, sample_rate = 0.2)
            print(f"idx is {idx}, seed is {seed},k is{k}, accuracy={ac} \n")
            each_ac_list.append(ac)
            all_record.append([new_hyper, 'idx', idx, 'seed', seed, ac])
        each_ac_list = np.array(each_ac_list)
        print(f'mean over seed accuracy is {np.mean(each_ac_list)}')
        if np.mean(each_ac_list)>optimal_ac:
            best_hyper = new_hyper
            optimal_ac = max(optimal_ac, np.mean(each_ac_list))
    print(f'when eps is{eps} the best accuracy is {optimal_ac}')
    test_ac_list.append(optimal_ac)
    best_hyper_list.append(best_hyper)
record = {}
record['eps_list'] = record_eps
record['all_record'] = all_record
#record['noise_mul_list'] = NOISE_MUL_LIST
record['best_hyper'] = best_hyper_list
record['acc_kNN'] = test_ac_list
with open(private_knn_file, 'wb') as f:
    pickle.dump(record, f)

"""
num_query_list = [200*x for x in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16,18, 22, 24, 28, 30]]
# sample with proba 0.1
sigma_list_6000_01 = [4.0, 5.4, 6.8, 7.5,  8.2, 9, 10, 10.5, 11, 12, 13, 14, 15, 16, 17.5, 19, 20.5, 21]
# sample with prob 0.2
sigma_list_6000_02 = [7.9, 11.1, 13.5, 15.5,  17.4, 19, 20.5, 21.9, 23.3, 24.5, 26.5, 29, 31, 33, 36.2, 38.2, 40.8, 42.7]
num_query_list = [5000]
sigma_list = [1.0]
#sigma_list = sigma_list_6000_01
for idx  in range(1):
    best_hyper = {}
    num_query = num_query_list[idx]
    basic_sigma = sigma_list[idx]
    print('idx ', idx, 'current query', num_query)
    optimal_ac = 0
    record_eps.append(idx) 
    for k in [ 700]:
        each_ac_list = []
        new_hyper = {'k':k, 'sigma':basic_sigma}
        for seed in [2]:
            print(f"dataset={DATASET}, k={k},  basic_sigma={basic_sigma}.")
            ac = PrivatekNN(dataset=DATASET, noisy_scale=basic_sigma, feature=FEATURE, num_query=num_query_list[idx], nb_teachers=k, nb_labels=NUM_CLASS,seed=seed, dataset_path=DATASET_PATH, sample_rate = 0.1)
            print(f"idx is {idx}, num_query is {num_query}, seed is {seed},k is{k}, accuracy={ac} \n")
            each_ac_list.append(ac)
            all_record.append([new_hyper, 'idx', idx, 'seed', seed, ac])
        each_ac_list = np.array(each_ac_list)
        print(f'mean over seed accuracy is {np.mean(each_ac_list)}')
        if np.mean(each_ac_list)>optimal_ac:
            best_hyper = new_hyper
            optimal_ac = max(optimal_ac, np.mean(each_ac_list))
    print(f'when num of query is{num_query_list[idx]} the best accuracy is {optimal_ac}')
    test_ac_list.append(optimal_ac)
    best_hyper_list.append(best_hyper)
record = {}
record['eps_list'] = record_eps
record['all_record'] = all_record
#record['noise_mul_list'] = NOISE_MUL_LIST
record['best_hyper'] = best_hyper_list
record['acc_kNN'] = test_ac_list
with open(private_knn_file, 'wb') as f:
    pickle.dump(record, f)
