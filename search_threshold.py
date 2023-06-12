from threshold import IndividualkNN
import numpy as np
import random
import pickle
"""
Tune the best threshold for ind-kNN from a threshold list on a randomly picked validation set. 
"""
DATASET_PATH = 'dataset/'

NUM_CLASS = 10
NUM_QUERY = 200
# VARS is only used when kernel_method = 'RBF'
VARS = np.exp([  1.5])
#FEATURE = 'all-roberta-large-v1'
#DATASET = 'agnews'

FEATURE = 'vit'
#DATASET = 'fmnist'
DATASET = 'cifar10'
# The threshold is chosen from [0, 1]
threshold_list = [0.03*x for x in range( 0, 20)]
kernel_method = 'cosine'


seed = 0

for var in VARS:
    for repeat in range(1):
        print(f"dataset={DATASET}, kernel_method={kernel_method},  var={np.log(var)}")
        count_neighbor_list, ac = IndividualkNN(dataset=DATASET, var=var, kernel_method=kernel_method,
                   feature=FEATURE, num_query=NUM_QUERY, nb_labels=NUM_CLASS,seed=seed, threshold_list=threshold_list, dataset_path=DATASET_PATH)
        print(f"accuracy={ac} \n")



ac = np.array(ac)
print('argmax threshold is', threshold_list[np.argmax(ac)])
