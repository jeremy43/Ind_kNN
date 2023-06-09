from threshold import IndividualkNN
#from rescale_hash_knn import IndividualkNN
from hand_dp import hand_dp
import numpy as np
import random
import pickle
"""
Set the Parameters
"""
DATASET_PATH = 'dataset/'

NUM_CLASS = 10
NUM_QUERY = 200
VARS = np.exp([  1.5])
#FEATURE = 'all-roberta-large-v1'
#DATASET = 'agnews'
#FEATURE = 'resnet50'
#FEATURE = 'resnet50'
FEATURE = 'clr'
#DATASET = 'fmnist'
DATASET = 'cifar10'
threshold_list = [0.03*x for x in range( 0, 20)] #+[-0.04, -0.08, -0.12]
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
