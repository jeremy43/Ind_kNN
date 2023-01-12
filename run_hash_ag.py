from hash_knn import IndividualkNN
import numpy as np
import pickle
import random

NUM_QUERY = 500
VARS = np.exp([1.0])
NB_HASH_TABLES = [20]
PROJ_DIMS = [12]
FEATURE = 'all-roberta-large-v1'
DATASET = 'agnews'
DATASET_PATH = '/home/yq/dataset'

# DATASET = 'INaturalist'
EPS_LIST = [1.3 ** x * 0.1 for x in range(12)]
NOISE_MUL_LIST = [30.75, 24.19, 19.03, 14.96, 11.76, 9.24, 7.26, 5.71, 4.49, 3.54, 2.79, 2.2]
IND_BUDGETS = np.exp([-1.2, 0])
# IND_BUDGETS = [10]
NORM = 'centering+L2'
num_point = 12
idx = 0
# kernel_method = 'RBF'
kernel_method = 'cosine'
# kernel_method = 'student'
hash_method = 'basic+threshold'
hash_file_name = f'hash_{DATASET}_method_{hash_method}_Query_{NUM_QUERY}_record.pkl'
print(f'file_name is {hash_file_name}')
test_ac_list = []
for idx in range(7, 12):
    eps = EPS_LIST[idx]
    print('idx ', idx, 'current epsilon', eps)
    noise_mul = NOISE_MUL_LIST[idx]
    optimal_ac = 0
    for var in VARS:
        for ind_budget in IND_BUDGETS:
            sigma = ind_budget * noise_mul
            print('current budget is', ind_budget, 'sigma is', sigma)
            for min_weight in [0, 0.1, 0.3, 0.4, 0.5, 0.62]:
                # for nb_teachers in [50, 100]:
                # nb_teachers is useless in the threshold-based method
                for nb_tables in NB_HASH_TABLES:
                    for proj_dim in PROJ_DIMS:
                        for seed in [0, 1, 2, 3, 10, 100]:
                            each_ac_list = []
                            print(f"dataset={DATASET}, min_weight={min_weight},hash_method = {hash_method}, eps is ={idx},  nb_tables={nb_tables}, "
                                  f"var={np.log(var)}, proj_dim={proj_dim},  noise_scale={np.log(sigma)}, ind_budget={np.log(ind_budget)},  norm={NORM}")
                            ac = IndividualkNN(dataset=DATASET, var=var, noisy_scale=sigma, kernel_method=kernel_method, min_weight=min_weight,
                                               seed=seed, hash_method=hash_method, feature=FEATURE, num_query=NUM_QUERY,
                                               num_tables=nb_tables, proj_dim=proj_dim, ind_budget=ind_budget, nb_labels=14,
                                               norm=NORM, dataset_path=DATASET_PATH)
                            print(f"accuracy={ac}")
                            each_ac_list.append(ac)
                        optimal_ac = max(optimal_ac, sum(each_ac_list) / len(each_ac_list))
    print(f'when eps is {eps} the best accuracy is {optimal_ac}')
    test_ac_list.append(optimal_ac)

record = {}
record['eps_list'] = EPS_LIST
record['noise_mul_list'] = NOISE_MUL_LIST
record['acc_kNN'] = test_ac_list
with open(hash_file_name, 'wb') as f:
    pickle.dump(record, f)
