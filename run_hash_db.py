from hash_knn import IndividualkNN
import numpy as np

NUM_QUERY = 300
VARS  = np.exp([1.0])
NB_HASH_TABLES =  [12, 16, 20]
PROJ_DIMS =  [12]
FEATURE = 'resnet50'
CLIPS=[0.25]
FEATURE = 'all-roberta-large-v1'
DATASET = 'dbpedia'
DATASET_PATH = '/home/yq/dataset'

#DATASET = 'INaturalist'
EPS_LIST = [1.3**x*0.1 for x in range(12)]
NOISE_MUL_LIST = [30.75, 24.19, 19.03, 14.96, 11.76, 9.24, 7.26, 5.71, 4.49, 3.54, 2.79, 2.2]
IND_BUDGETS =np.exp([-20, -18, -16, -14])
IND_BUDGETS =np.exp([-1])
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
            print('current budget is', ind_budget, 'sigma is', sigma)
            #clip = 0.5
            #clip = ind_budget
            for min_weight  in [ 0.6]:
            #for nb_teachers in [50, 100, 200, 1000]:
                clip = 0.25
                nb_teachers = 20
                #for max_dis in np.exp([2.5, 2.75] ):
                for nb_tables in NB_HASH_TABLES:
                    each_ac_list = []
                    for proj_dim in PROJ_DIMS:
                        print(f"dataset={DATASET}, min_weight={np.log(min_weight)},hash_method = {hash_method}, eps is ={idx}, nb_teachers={nb_teachers}, nb_tables={nb_tables}, var={np.log(var)}, proj_dim={proj_dim},  noise_scale={np.log(sigma)}, ind_budget={np.log(ind_budget)}, clip threshold={clip} norm={NORM}")
                        ac = IndividualkNN(dataset=DATASET, var=var,
                 noisy_scale=sigma, clip=clip, min_weight = min_weight,  nb_teachers = nb_teachers, hash_method=hash_method,  feature=FEATURE, num_query=NUM_QUERY,  num_tables=nb_tables, proj_dim=proj_dim,
                 ind_budget=ind_budget, nb_labels=14, norm='L2', dataset_path=DATASET_PATH)
                        print(f"accuracy={ac}")
