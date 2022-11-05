from __future__ import absolute_import
import os
import sys
import errno
import network
from models import Resnext
from torchvision.models import resnet50, ResNet50_Weights
from sentence_transformers import SentenceTransformer
import pickle
import shutil
import json
import os.path as osp
import numpy as np
import torch
import scipy

sys.path.append('.')
from sklearn.decomposition import PCA, KernelPCA
import math

SHAPES = {
    "cifar10": (32, 32, 3),
    "cifar10_500K": (32, 32, 3),
    "fmnist": (28, 28, 1),
    "mnist": (28, 28, 1),
    "svhn": (298, 28, 3)
}


# def get_scatter_transform():
#     shape = SHAPES.get(config.dataset)
#     scattering = Scattering2D(J=2, shape=shape[:2])
#     K = 81 * shape[2]
#     (h, w) = shape[:2]
#     return scattering, K, (h // 4, w // 4)


def extract_feature(train_datapoint, test_datapoint, feature, dataset='cifar10'):
    """
    This help to compute feature for knn from pretrained network
    :param FLAGS:
    :param ckpt_path:
    :return:
    """
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,3'
    # check if a certain variable has been saved in the model
    print('dataset in extract feature', dataset)
    if feature == 'resnet50':
        # Update the feature extractor using the student model(filename) in the last iteration.
        # Replace the filename with the saved student model, the following in an example of the checkpoint

        weight = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weight)
        model.eval()

        if os.path.exists(f'{dataset}_{feature}_train.npy'):
            train_feature = np.load(f'{dataset}_{feature}_train.npy')
            test_feature = np.load(f'{dataset}_{feature}_test.npy')
            return train_feature, test_feature
        train_feature = network.predFeature(model, train_datapoint)
        print('feature shape', train_feature.shape)
        test_feature = network.predFeature(model, test_datapoint)
        np.save(f'{dataset}_{feature}_train.npy', train_feature)
        np.save(f'{dataset}_{feature}_test.npy', test_feature)
        return train_feature, test_feature

    elif feature == 'all-roberta-large-v1':
        model = SentenceTransformer('all-roberta-large-v1')
        if os.path.exists(f'{dataset}_{feature}_train.npy'):
            train_feature = np.load(f'{dataset}_{feature}_train.npy')
            test_feature = np.load(f'{dataset}_{feature}_test.npy')
            return train_feature, test_feature
        train_feature = model.encode(train_datapoint)
        print('feature shape', train_feature.shape)
        test_feature = model.encode(test_datapoint)
        np.save(f'{dataset}_{feature}_train.npy', train_feature)
        np.save(f'{dataset}_{feature}_test.npy', test_feature)
        return train_feature, test_feature

    elif feature == 'resnext29':
        if os.path.exists(dataset + '_resnext29_train.npy'):
            train_feature = np.load(dataset + '_resnext29_train.npy')
            test_feature = np.load(dataset + '_resnext29_test.npy')
            return train_feature, test_feature

        model = Resnext.resnext(
            cardinality=8,
            num_classes=100,
            depth=29,
            widen_factor=4,
            dropRate=0, )

        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        checkpoint = torch.load("/home/yq/ind_kNN/pytorch-classification/checkpoints/cifar10/resnext-8x64d/model_best.pth.tar")
        model.load_state_dict(checkpoint['state_dict'])
        train_feature = network.predFeature(model, train_datapoint)
        print('feature shape', train_feature.shape)
        test_feature = network.predFeature(model, test_datapoint)
        np.save(dataset + '_resnext29_train.npy', train_feature)
        np.save(dataset + '_resnext29_test.npy', test_feature)
        return train_feature, test_feature
    """
    if feature == 'scatter':
        scattering, K, (h, w) = get_scatter_transform()
        train_scatters = []
        each_length = int((9 + len(train_img)) / 10)
        for idx in range(10):
            print('load idx=', idx)
            train_scatter_path = os.path.join(config.scatter_path, config.dataset + str(idx)+'_train_scatter.pkl')
            test_scatter_path = os.path.join(config.scatter_path, config.dataset + str(idx)+'_test_scatter.pkl')
            p1 = idx * each_length
            p2 = min((idx + 1) * each_length, len(train_img))
            save_record = []
            if os.path.exists(train_scatter_path) == False:
                cache_train_img = [train_img[i] for i in range(p1,p2)]
                for (train_data,_) in cache_train_img:
                    train_feature = scattering(train_data)
                    train_feature = torch.flatten(train_feature)
                    save_record.append(train_feature)
                with open(train_scatter_path, 'wb') as f:
                    pickle.dump(save_record, f)
            else:
                with open(train_scatter_path, 'rb') as f:
                    if len(train_scatters)>0:
                        #train_scatters = torch.vstack((train_scatters, pickle.load(f)))
                        train_scatters = train_scatters + pickle.load(f)
                        # train_data = np.vstack((train_data, pickle.load(f)))
                    else:
                        train_scatters = pickle.load(f)
        return train_scatters, train_scatters

        if os.path.exists(test_scatter_path) == False:
            test_scatters = []
            for (test_data, target) in test_img:
                test_feature = scattering(test_data)
                test_scatters.append(test_feature)
            test_scatters = torch.cat(test_scatters, axis=0)
            with open(test_scatter_path, 'wb') as f:
                pickle.dump(test_scatters, f)
        else:
            with open(test_scatter_path, 'rb') as f:
                test_scatters = pickle.load(f)
        return train_scatters, test_scatters
    """


def hamming_precision(y_true, y_pred, torch=True, cate=True):
    acc_list = []
    if torch:
        from sklearn.metrics import accuracy_score
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
    for i in range(len(y_true)):

        set_true = set(np.where(y_true[i] == 1)[0])
        set_pred = set(np.where(y_pred[i] == 1)[0])
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred)) / \
                    float(len(set_true.union(set_pred)))
        acc_list.append(tmp_a)
    return np.mean(acc_list)


class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))


class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def set_bn_to_eval(m):
    # 1. no update for running mean and var
    # 2. scale and shift parameters are still trainable
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
