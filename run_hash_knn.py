from hash_knn import IndividualkNN
import numpy as np

IND_BUDGETS = [1000000]
NUM_QUERY = 200
VARS  = np.exp([  4.0])
NB_HASH_TABLES =  [8, 12]
PROJ_DIMS = [10, 12]
#FEATURE = 'resnet50'
CLIPS=[0.25]
DATASET = 'cifar10'
FEATURE = 'all-roberta-large-v1'
#DATASET = 'sst2'
DATASET = 'agnews'
DATASET_PATH = '/home/yq/dataset'
#DATASET = 'INaturalist'
EPS_LIST = [1.3**x*0.1 for x in range(12)]
NOISE_MUL_LIST = [30.75, 24.19, 19.03, 14.96, 11.76, 9.24, 7.26, 5.71, 4.49, 3.54, 2.79, 2.2]
IND_BUDGETS =np.exp([10])
NORM = 'L2'
num_point = 12
idx = 0
hash_method = 'basic+threshold'
test_ac_list = []
for idx  in range(8, 9):
    eps = EPS_LIST[idx]
    print('idx ', idx, 'current epsilon', eps)
    noise_mul  = NOISE_MUL_LIST[idx]
    optimal_ac = 0
    for var in VARS:
        #for sigma in NOISY_SCALES: 
        for ind_budget in IND_BUDGETS:
            #sigma = 0
            sigma = ind_budget * noise_mul
            sigma = 0
            print('current budget is', ind_budget, 'sigma is', sigma)
            #clip = 0.5
            #clip = ind_budget
            for nb_teachers in [ 50]:
            #for nb_teachers in [50, 100, 200, 1000]:
                clip = 0.25
                min_weight = np.exp(-8.5)
                #for max_dis in np.exp([2.5, 2.75] ):
                for nb_tables in NB_HASH_TABLES:
                    each_ac_list = []
                    for proj_dim in PROJ_DIMS:
                        print(f"dataset={DATASET}, hash_method = {hash_method}, eps is ={idx}, nb_teachers={nb_teachers}, nb_tables={nb_tables}, var={np.log(var)}, proj_dim={proj_dim},  noise_scale={np.log(sigma)}, ind_budget={np.log(ind_budget)}, clip threshold={clip} norm={NORM}")
                        ac = IndividualkNN(dataset=DATASET, var=var,
                 noisy_scale=sigma, clip=clip, min_weight = min_weight,  nb_teachers = nb_teachers, hash_method=hash_method,  feature=FEATURE, num_query=NUM_QUERY,  num_tables=nb_tables, proj_dim=proj_dim,
                 ind_budget=ind_budget, nb_labels=10, norm='L2', dataset_path=DATASET_PATH)
                        print(f"accuracy={ac}")
