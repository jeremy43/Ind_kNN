import argparse
import numpy as np
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from weight_knn import PrepareData
from handcrafted_dp.models import StandardizeLayer
from handcrafted_dp.train_utils import get_device, train, test
from handcrafted_dp.dp_utils import ORDERS, get_privacy_spent, get_renyi_divergence
from handcrafted_dp.log import Logger


def hand_dp(feature=None, batch_size=2048, mini_batch_size=256,
            lr=1, optim="SGD", momentum=0.9, nesterov=False, noise_multiplier=1,
            max_grad_norm=0.1, max_epsilon=None, epochs=100, logdir=None,
            dataset='cifar10', dataset_path=None, num_query=1000, num_class=10):
    logger = Logger(logdir)

    device = get_device()

    # get pre-computed features
    x_train, y_train, x_test, y_test = PrepareData(dataset, feature, num_query, dataset_path)
    # train_data, test_data = get_data(dataset, augment=False)
    # y_train = np.asarray(train_data.targets)
    # y_test = np.asarray(test_data.targets)
    x_train = np.asarray(x_train, dtype='float32')
    x_test = np.asarray(x_test, dtype='float32')
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    print('y_train shape', y_train.shape)
    print('y_test shape', y_test.shape)
    print('x_train shape', x_train.shape)
    print('x_test shape', x_test.shape)
    trainset = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    testset = torch.utils.data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    bs = batch_size
    assert bs % mini_batch_size == 0
    n_acc_steps = bs // mini_batch_size
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=mini_batch_size, shuffle=True, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=mini_batch_size, shuffle=False, pin_memory=True)

    n_features = x_train.shape[-1]
    try:
        mean = np.load(f"{dataset}_{feature}_mean.npy")
        var = np.load(f"{dataset}_{feature}_var.npy")
    except FileNotFoundError:
        mean = np.zeros(n_features, dtype=np.float32)
        var = np.ones(n_features, dtype=np.float32)

    bn_stats = (torch.from_numpy(mean).to(device), torch.from_numpy(var).to(device))

    model = nn.Sequential(StandardizeLayer(bn_stats), nn.Linear(n_features, num_class)).to(device)

    if optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=momentum,
                                    nesterov=nesterov)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print('sample_rate', bs / y_train.shape[0], 'batch_size', bs, 'len of train_data', y_train.shape[0])
    privacy_engine = PrivacyEngine(
        model,
        sample_rate=bs / y_train.shape[0],
        alphas=ORDERS,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    privacy_engine.attach(optimizer)

    for epoch in range(0, epochs):
        print(f"\nEpoch: {epoch}")

        train_loss, train_acc = train(model, train_loader, optimizer, n_acc_steps=n_acc_steps)
        test_loss, test_acc = test(model, test_loader)

        if noise_multiplier > 0:
            rdp_sgd = get_renyi_divergence(
                privacy_engine.sample_rate, privacy_engine.noise_multiplier
            ) * privacy_engine.steps
            epsilon, _ = get_privacy_spent(rdp_sgd)
            print(f"ε = {epsilon:.3f}")

            if max_epsilon is not None and epsilon >= max_epsilon:
                return
        else:
            epsilon = None

        logger.log_epoch(epoch, train_loss, train_acc, test_loss, test_acc, epsilon)

    return test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--optim', type=str, default="SGD", choices=["SGD", "Adam"])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', action="store_true")
    parser.add_argument('--noise_multiplier', type=float, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--feature_path', type=str, default="./cifar10_resnet50")
    parser.add_argument('--max_epsilon', type=float, default=None)
    parser.add_argument('--logdir', default='log_dir')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--dataset_path', type=str, default='./dataset')
    args = parser.parse_args()
    hand_dp(**vars(args))
