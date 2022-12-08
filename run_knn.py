from knn import IndividualkNN
import numpy as np

NB_TEACHERS = [150, 200]
IND_BUDGETS = [50]
NUM_QUERY = 100
NOISY_SCALE = 14.
FEATURE = 'resnet50'

for ind_budget in IND_BUDGETS:
    for nb_teachers in NB_TEACHERS:
        print(f"noise_scale={NOISY_SCALE}, ind_budget={ind_budget}, nb_teachers={nb_teachers}")
        ac = IndividualkNN(dataset="cifar10",
                           noisy_scale=NOISY_SCALE, feature=FEATURE, num_query=NUM_QUERY, nb_teachers=nb_teachers,
                           ind_budget=ind_budget)
        print(f"accuracy={ac}")
