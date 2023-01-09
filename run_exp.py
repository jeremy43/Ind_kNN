from weight_knn import IndividualkNN
from hand_dp import hand_dp
import numpy as np

"""
Set the Parameters
"""
# dataset_path = '/home/yq/dataset'
# dataset_path = '/home/xuandong/mnt/dataset/'
DATASET_PATH = '/home/xuandongz/dataset'
NB_TEACHERS = [50]

# VARS = np.exp([0.1, 1, 3, 10, 100]) # VARS = np.exp([3]) # image
# VARS = [0.1] # sst-2 gaussian 0.08 0.09
IND_BUDGETS = [2000000]  # nondp
NUM_QUERY = 500
VARS = [0.2]
# NOISY_SCALES = [1e-20]  # nondp
NOISY_SCALES = [0]
# FEATURE = 'resnet50'
# DATASET = 'cifar10'
FEATURE = 'all-roberta-large-v1'
DATASET = 'dbpedia'

# for var in VARS:
#     for ind_budget in IND_BUDGETS:
#         for nb_teachers in NB_TEACHERS:
#             for noisy_scale in NOISY_SCALES:
#                 print(f"dataset={DATASET}, var={var}, noise_scale={noisy_scale}, ind_budget={ind_budget}, nb_teachers={nb_teachers}")
#
#                 ac = IndividualkNN(dataset=DATASET, var=var, noisy_scale=noisy_scale,
#                                    feature=FEATURE, num_query=NUM_QUERY, nb_teachers=nb_teachers,
#                                    ind_budget=ind_budget, nb_labels=14, dataset_path=DATASET_PATH)
#                 print(f"ind={ind_budget}, nb={nb_teachers}, noise={noisy_scale}, accuracy={ac} \n")

"""
Result of handcrafted dp
"""
#
hand_acc = hand_dp(feature=FEATURE, batch_size=256, mini_batch_size=256,
                   lr=1, optim="SGD", momentum=0.9, nesterov=False, noise_multiplier=4,
                   max_grad_norm=0.1, max_epsilon=None, epochs=10, logdir=None,
                   dataset=DATASET, dataset_path=DATASET_PATH, num_query=NUM_QUERY)

print('Hand DP acc', hand_acc)
# print('indKNN acc', ac)
