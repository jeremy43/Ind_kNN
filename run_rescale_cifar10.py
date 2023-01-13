from rescale_weight_knn import IndividualkNN
from hand_dp import hand_dp
import numpy as np
import random
import pickle
"""
Set the Parameters
"""
# dataset_path = '/home/yq/dataset'
# dataset_path = '/home/xuandong/mnt/dataset/'
DATASET_PATH = '/home/yq/dataset'
NB_TEACHERS = [100]
NUM_CLASS = 10
NUM_QUERY = 800
VARS = np.exp([  1.7])
#NOISY_SCALES = [0]  # nondp
FEATURE = 'resnet50'
#DATASET = 'INaturalist'
#FEATURE = 'all-roberta-large-v1'
#DATASET = 'sst2'
DATASET = 'cifar10'
EPS_LIST = [1.3**x*0.1 for x in range(12)]
NOISE_MUL_LIST = [30.75, 24.19, 19.03, 14.96, 11.76, 9.24, 7.26, 5.71, 4.49, 3.54, 2.79, 2.2]
NORM = 'L2'
num_point = 12

SIGMA_KERNEL =[0.7,1., 2.5, 3.5, 4]
VARS  = np.exp([1.7])
kNN_file_name = f'rescale_kNN_{DATASET}_Query_{NUM_QUERY}_record.pkl'
print(f'file_name is {kNN_file_name}')
idx = 0
kernel_method = 'RBF'
test_ac_list = []
h = 0.25 # ratio of budget for gaussian mechanism
# sigma_1 = np.sqrt(T *noise_mul**2/h)
best_hyper_list = []

for idx  in range(12):
    best_hyper = {}
    eps = EPS_LIST[idx]
    print('idx ', idx, 'current epsilon', eps)
    noise_mul  = NOISE_MUL_LIST[idx]
    optimal_ac = 0
    # alpha * ind_budget  = alpha/(2*noise_mul**2)
    ind_budget = 1.0/(2*noise_mul**2)
    for basic_sigma in SIGMA_KERNEL:
        #sigma = 0
        if idx<5 and basic_sigma<2.5:
            continue
        if idx<2 and basic_sigma<3.5:
            continue
        print('current budget is', ind_budget, 'sigma is', basic_sigma)
        for h in [4.0]:
            # h represents the ratio of budget used to release the number of neighbors
            sigma_1 = np.sqrt(NUM_QUERY/(2*ind_budget*h))
            for var in VARS:
                #for max_dis in np.exp([2.5, 2.75] ):
                for min_weight in ([0.8, 0.82]):
                    each_ac_list = []
                    new_hyper = {'min_weight':min_weight, 'sigma2':basic_sigma, 'h':h}
                    for seed in [0, 1, 2, 3, 10, 100]:
                        print(f"dataset={DATASET},h={h}, kernel_method={kernel_method}, min_weight={min_weight}, var={np.log(var)}, sigma_1 is {sigma_1},  basic_sigma={basic_sigma}, ind_budget={ind_budget},  norm={NORM}")

                        num_data, ac = IndividualkNN(dataset=DATASET, var=var, kernel_method=kernel_method, noisy_scale=basic_sigma,
                                   feature=FEATURE, num_query=NUM_QUERY, min_weight=min_weight,  sigma_1 = sigma_1, 
                                   ind_budget=ind_budget, nb_labels=NUM_CLASS,seed=seed, dataset_path=DATASET_PATH)
                        print(f"seed is {seed}, accuracy={ac} \n")
                        each_ac_list.append(ac)
                each_ac_list = np.array(each_ac_list)
                print(f'mean over seed accuracy is {np.mean(each_ac_list)}')
                if np.mean(each_ac_list)>optimal_ac:
                    best_hyper = new_hyper
                    optimal_ac = max(optimal_ac, np.mean(each_ac_list))
    print(f'when eps is{eps} the best accuracy is {optimal_ac}')
    test_ac_list.append(optimal_ac)
    best_hyper_list.append(best_hyper)
record = {}
record['eps_list'] = EPS_LIST
record['noise_mul_list'] = NOISE_MUL_LIST
record['best_hyper'] = best_hyper_list
record['acc_kNN'] = test_ac_list
with open(kNN_file_name, 'wb') as f:
    pickle.dump(record, f)
#print('indKNN acc', ac)
"""
#Result of handcrafted dp
# such that the eps is chosen from [1.3**x*0.1 for x in range(10)]

EPS_LIST = [1.3**x*0.1 for x in range(12)]
SIGMA_LIST =  [5.5, 4.4, 3.5, 2.76, 2.2, 1.8, 1.53, 1.35, 1.21, 1.11, 1.02, 0.96]
test_acc_list = []
for (mul, eps) in zip(SIGMA_LIST, EPS_LIST):
    acc = []
    for repeat in range(5):
        hand_acc = hand_dp(feature=FEATURE, batch_size=256, mini_batch_size=256,
                   lr=10, optim="SGD", momentum=0.9, nesterov=False, noise_multiplier=mul,
                   max_grad_norm=0.1, max_epsilon=None, epochs=10, logdir=None,
                   dataset=DATASET, dataset_path=DATASET_PATH, num_query=NUM_QUERY, num_class=NUM_CLASS)
    
        acc.append(hand_acc)
    acc = np.array(acc)
    print('noisy multiplier is ', mul, ' mean of Hand DP acc', np.mean(hand_acc))
    test_acc_list.append(acc)
test_acc_list = np.array(test_acc_list)
file_name = f'noisySGD_{DATASET}_Query_{NUM_QUERY}_record.pkl'
record = {}
record['eps_list'] = EPS_LIST
record['sigma_list_noisySGD'] = SIGMA_LIST
record['acc_noisySGD'] = test_acc_list
with open(file_name, 'wb') as f:
    pickle.dump(record, f)
#print('indKNN acc', ac)
"""
