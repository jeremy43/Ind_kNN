from weight_knn import IndividualkNN
import numpy as np

NB_TEACHERS = [400]
IND_BUDGETS = [1000]
NUM_QUERY = 100
VARS = np.exp([1., 2., 3., 4., 5., 6., 7., 8., 10.])

NOISY_SCALES = [1e-20]
# FEATURE = 'resnet50'
# DATASET = 'cifar10'
FEATURE = 'all-roberta-large-v1'
DATASET = 'sst2'
for var in VARS:
    for ind_budget in IND_BUDGETS:
        for nb_teachers in NB_TEACHERS:
            for noisy_scale in NOISY_SCALES:
                print(f"dataset={DATASET}, var={var}, noise_scale={noisy_scale}, ind_budget={ind_budget}, nb_teachers={nb_teachers}")

                ac = IndividualkNN(dataset=DATASET, var=var,
                                   noisy_scale=noisy_scale, feature=FEATURE, num_query=NUM_QUERY, nb_teachers=nb_teachers,
                                   ind_budget=ind_budget, nb_labels=10)
                print(f"accuracy={ac} \n")
