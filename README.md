# IndkNN

This repository contains the implementation of "Private Prediction Strikes Back!” Private Kernelized Nearest Neighbors with
Individual Rényi Filter".


## Installation

The main dependencies are pytorch, datasets, autodp

You can install all requirements with:

`pip install -r rquirement.txt`




# Overview
The Ind-kNN algorithm is privacy-preserving prediction algorithm that allows individual private data controls
privacy loss at an individual level. Ind-kNN is easily updatable user dataset changes.
This table presents the  privacy cost of answering T = 2000 queries on CIFAR-10. To simulate the dataset change scenario, 
 we assume that a retraining (machine unlearning) request is made every answering 100 queries.
 We compare Ind-kNN with
 
1. Linear NoisySGD [Feldman and Zrnic, 2021] 
2. Private kNN [Zhu et al., 2020] 

The median accuracy of all approaches is aligned to 96.0%.

Method                | Linear NoisySGD| Linear NoisySGD (with retrain)| Private kNN| **Ind-KNN** | **Ind-kNN+hashing**
---------------------- | :------------------------: | :--------:| :---:   | :---: |:---:   
Privacy loss (epsilon)     |1.5    | 6.2     |  4.1 |  2.0| 3.2  

# How to run

  The algorithm

1. extracts features for both private labeled data and public unlabeled queries using a feature extractor.
2. assigns each private data point with a pre-determined privacy budget.
3. applies 'threshold'-based nearest neighbor to make predictions for each query and account privacy loss for each
selected private neighbor.
4. Drop private data whose privacy budget is empty.


## Tune hyper-parameter 

Ind-kNN has two hyper-parameters needed to tune: threshold and sigma_2

1. The threshold denotes the minimum kernel weight of the neighbor-selection strategy. To tune threshold
on CIFAR-10 dataset with vision transformer under RBF kernel, run 
`python search_threshold.py`
2. The sigma2 denotes the noise scale to perturb the sum of neighbors' predicition, which is tuned on the validation set.
We suggest tune sigma2 between [0.2, 0.8].

## Examples with vision transformer

Run Ind-kNN on CIFAR-10 on Vision transformer-based features under (epsilon=2.0, delta=1e-5)-DP, run:

'python ind_knn.py --eps=2.0 --delta=1e-5 --threshold=0.26 --kernel_method='cosine' --feature='vit' --dataset='cifar10' --sigma_2=0.5 --num_query=1000'

Run Ind-kNN on AG News on sentence transformer-based features under (epsilon=2.0, delta=1e-5)-DP., run:

`python ind_knn.py --eps=2.0 --delta=1e-5 --threshold=0.38 --kernel_method='cosine' --feature='all-roberta-large-v1' --dataset='agnews' --sigma_2=0.2 --num_query=500`