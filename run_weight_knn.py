from weight_knn import IndividualkNN
import numpy as np

NB_TEACHERS = [100, 200]
IND_BUDGETS = [1000]
NUM_QUERY = 100
VARS  = np.exp([1.])

NOISY_SCALES = [1e-20]
FEATURE = 'resnet50'
DATASET = 'INaturalist'
DATASET_PATH = '/home/yq/dataset'
#DATASET = 'cifar10'
for var in VARS:
    for ind_budget in IND_BUDGETS:
        for nb_teachers in NB_TEACHERS:
            for noisy_scale in NOISY_SCALES:
                print(f"dataset={DATASET}, var={var}, noise_scale={noisy_scale}, ind_budget={ind_budget}, nb_teachers={nb_teachers}")

                ac = IndividualkNN(dataset=DATASET, var=var,
                 noisy_scale=noisy_scale, feature=FEATURE, num_query=NUM_QUERY, nb_teachers=nb_teachers,
                 ind_budget=ind_budget, nb_labels=10000, norm='L2', dataset_path=DATASET_PATH)
                print(f"accuracy={ac}")
