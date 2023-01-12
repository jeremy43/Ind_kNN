from correct_weight_knn import IndividualkNN
from hand_dp import hand_dp
import numpy as np
import pickle

"""
Set the Parameters
"""
# dataset_path = '/home/yq/dataset'
# dataset_path = '/home/xuandong/mnt/dataset/'
DATASET_PATH = '/home/yq/dataset'
NB_TEACHERS = [100]
NUM_CLASS = 10
NUM_QUERY = 1000
VARS = np.exp([1., 2.0])
CLIPS = [0.25]
# NOISY_SCALES = [0]  # nondp

# DATASET = 'INaturalist'
FEATURE = 'all-roberta-large-v1'
# DATASET = 'sst2'
DATASET = 'agnews'
EPS_LIST = [1.3 ** x * 0.1 for x in range(12)]
NOISE_MUL_LIST = [30.75, 24.19, 19.03, 14.96, 11.76, 9.24, 7.26, 5.71, 4.49, 3.54, 2.79, 2.2]
IND_BUDGETS = [1.]
NORM = 'L2'
num_point = 12
kNN_file_name = f'kNN_{DATASET}_Query_{NUM_QUERY}_record.pkl'
print(f'file_name is {kNN_file_name}')
idx = 0
method = 'threshold'
test_ac_list = []
# for idx  in range(10, 11):
#     eps = EPS_LIST[idx]
#     print('idx ', idx, 'current epsilon', eps)
#     noise_mul  = NOISE_MUL_LIST[idx]
#     optimal_ac = 0
#     for ind_budget in IND_BUDGETS:
#         #for sigma in NOISY_SCALES:
#         sigma = ind_budget * noise_mul
#         #ind_budget = sigma/noise_mul
#         print('current budget is', ind_budget, 'sigma is', sigma)
#         for clip in CLIPS:
#             for var in VARS:
#                 #for max_dis in np.exp([2.5, 2.75] ):
#                 for min_weight in ([0.4, 0.5, 0.6]):
#                     each_ac_list = []
#                     for repeat in range(1):
#                         print(f"dataset={DATASET}, method={method}, min_weight={min_weight}, var={np.log(var)},  noise_scale={np.log(sigma)}, ind_budget={ind_budget}, clip threshold={clip} norm={NORM}")
#
#                         ac = IndividualkNN(dataset=DATASET, var=var, noisy_scale=sigma,
#                                    feature=FEATURE, num_query=NUM_QUERY, min_weight=min_weight,  clip=clip,
#                                    ind_budget=ind_budget, nb_labels=NUM_CLASS, dataset_path=DATASET_PATH)
#                         print(f"accuracy={ac} \n")
#                         each_ac_list.append(ac)
#                 each_ac_list = np.array(each_ac_list)
#                 optimal_ac = max(optimal_ac, np.mean(each_ac_list))
#     print(f'when eps is{eps} the best accuracy is {optimal_ac}')
#     test_ac_list.append(optimal_ac)
# record = {}
# record['eps_list'] = EPS_LIST
# record['noise_mul_list'] = NOISE_MUL_LIST
# record['acc_kNN'] = test_ac_list
# with open(kNN_file_name, 'wb') as f:
#     pickle.dump(record, f)
# print('indKNN acc', ac)

# Result of handcrafted dp
# such that the eps is chosen from [1.3**x*0.1 for x in range(10)]

EPS_LIST = [1.3 ** x * 0.1 for x in range(12)]
SIGMA_LIST = [5.5, 4.4, 3.5, 2.76, 2.2, 1.8, 1.53, 1.35, 1.21, 1.11, 1.02, 0.96]
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
# print('indKNN acc', ac)
