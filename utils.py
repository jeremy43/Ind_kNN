from __future__ import absolute_import
import os
import sys
import errno
import network
from models import Resnext
from torchvision.models import resnet50, ResNet50_Weights
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
import pickle
import shutil
import json
import os.path as osp
import numpy as np
from numpy import linalg as LA
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

def extract_label(dataset, name):
    if os.path.exists(f'{name}_label.npy'):
        label_list = np.load(f'{name}_label.npy')
        print('label_shape', label_list.shape)
    else:
        num_file = len(dataset)/50000
        label_list = []
        for idx, (imgs, label) in enumerate(dataset):
            if idx%10000==0:
                print('idx=', idx)
            label_list.append(label)
        label_list = np.array(label_list)
        np.save(f'{name}_label.npy', label_list)
        print('label_list shape', label_list.shape)
        """
        print('save label into', num_file)
        for i in range(int(num_file)):
            label_list = []
            loader = DataLoader(dataset[i*50000:(i+1)*50000], batch_size=200, shuffle=False, drop_last=False, num_workers=2
                            )
            for batch_idx, (imgs, label) in enumerate(loader):
                if batch_idx%100==0:
                    print(f'batch_idx={batch_idx}')
                label_list.append(label)
            label_list = np.concatenate(label_list, axis=0)
            print('label shape', label_list.shape)
            np.save(f'{name}_{i}_label.npy', label_list)
        label_list = []
        for i in range(num_file):
            cur_label_list = np.load('{name}_{i}_label.npy')
            label_list.append(cur_label_list)
        label_list = np.concatenate(cur_label_list, axis=0)
        print('save_label shape', label_list.shape)
        np.save(f'{name}_label.npy', label_list)
        """
    return label_list
def extract_feature(train_datapoint, test_datapoint, feature,   dataset='cifar10', norm=None):
    """
    This help to compute feature for knn from pretrained network
    :param FLAGS:
    :param ckpt_path:
    :return:
    """
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,3'
    # check if a certain variable has been saved in the model
    feature_path = 'features/'
    print('dataset in extract feature', dataset)
    if feature == 'resnet50':
        # Update the feature extractor using the student model(filename) in the last iteration.
        # Replace the filename with the saved student model, the following in an example of the checkpoint

        weight = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weight)
        model.eval()
        print('len of data', len(train_datapoint))
        if dataset =='dbpedia':
            train_path =f'/home/xuandongz/indkNN/dbpedia_all-roberta-large-v1_train.npy'
            test_path = f'/home/xuandongz/indkNN/dbpedia_all-roberta-large-v1_test.npy'
        else:
            train_path = f'features/{dataset}_{feature}_train.npy'
            test_path = f'features/{dataset}_{feature}_test.npy'
        if os.path.exists(train_path):
            print('file  exist')
            train_feature = np.load(train_path)
        else:
            print('file does not exist')
            train_feature = network.predFeature(model, train_datapoint)
            np.save(train_path, train_feature)
        if os.path.exists(test_path):
            test_feature = np.load(test_path)
        else:
            test_feature = network.predFeature(model, test_datapoint)
            np.save(test_path, test_feature)
        print('feature shape', train_feature.shape)
        print(' test feature shape', test_feature.shape)
        if norm == 'centering':
            print('use centering for preprocessing')
            train_mean = np.mean(train_feature, axis=0)
            train_var = np.var(train_feature, axis=0)
            test_mean = np.mean(test_feature, axis=0)
            test_var = np.var(test_feature, axis=0)
            train_feature_center = (train_feature -train_mean) / np.sqrt(train_var + 1e-5)
            test_feature_center  = (test_feature - test_mean) / np.sqrt(test_var + 1e-5)
            return train_feature_center, test_feature_center
        elif norm == 'centering+L2':
            print('use centering and L2')
            train_mean = np.mean(train_feature, axis=0)
            train_var = np.var(train_feature, axis=0)
            test_mean = np.mean(test_feature, axis=0)
            test_var = np.var(test_feature, axis=0)
            train_feature_center = train_feature -train_mean
            test_feature_center = test_feature - test_mean
            #train_feature_center = (train_feature -train_mean) / np.sqrt(train_var + 1e-5)
            #test_feature_center  = (test_feature - test_mean) / np.sqrt(test_var + 1e-5)
            train_l2_norm = LA.norm(train_feature_center, axis=1)
            test_l2_norm = LA.norm(test_feature_center, axis=1)
            train_feature_norm = train_feature_center / train_l2_norm[:, np.newaxis]
            test_feature_norm = test_feature_center / test_l2_norm[:, np.newaxis]
            print(f'test the first feature norm is {LA.norm(train_feature_norm[0,:])}')
            return train_feature_norm, test_feature_norm
        else:
            print(f'do not normalize feature')
            return train_feature, test_feature

    elif feature == 'all-roberta-large-v1':
        if dataset =='dbpedia':
            train_path =f'/home/xuandongz/indkNN/features/dbpedia_all-roberta-large-v1_train.npy'
            test_path = f'/home/xuandongz/indkNN/features/dbpedia_all-roberta-large-v1_test.npy'
        else:
            train_path = f'features/{dataset}_{feature}_train.npy'
            test_path = f'features/{dataset}_{feature}_test.npy'
        if os.path.exists(train_path):
            train_feature = np.load(train_path)
            test_feature = np.load(test_path)
            #return train_feature, test_feature
        else:
            model = SentenceTransformer('all-roberta-large-v1')
            train_feature = model.encode(train_datapoint)
            print('feature shape', train_feature.shape)
            test_feature = model.encode(test_datapoint)
            np.save(f'features/{dataset}_{feature}_train.npy', train_feature)
            np.save(f'features/{dataset}_{feature}_test.npy', test_feature)
        if norm == 'centering+L2':
            print('use centering and L2')
            print(f'test the first feature before normalization is {LA.norm(train_feature[0,:])}')
            train_mean = np.mean(train_feature, axis=0)
            print(f'train_mean is {train_mean}')
            train_var = np.var(train_feature, axis=0)
            test_mean = np.mean(test_feature, axis=0)
            test_var = np.var(test_feature, axis=0)
            train_feature_center = train_feature -train_mean
            test_feature_center = test_feature - test_mean
            #train_feature_center = (train_feature -train_mean) / np.sqrt(train_var + 1e-5)
            #test_feature_center  = (test_feature - test_mean) / np.sqrt(test_var + 1e-5)
            train_l2_norm = LA.norm(train_feature_center, axis=1)
            test_l2_norm = LA.norm(test_feature_center, axis=1)
            train_feature_norm = train_feature_center / train_l2_norm[:, np.newaxis]
            test_feature_norm = test_feature_center / test_l2_norm[:, np.newaxis]
            print(f'test the first feature norm is {LA.norm(train_feature_norm[0,:])}')
            return train_feature_norm, test_feature_norm
        else:
            print(f'do not normalize feature')
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
