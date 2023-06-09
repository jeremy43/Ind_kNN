import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
from datasets import load_dataset
import os
from transformers import ViTFeatureExtractor, ViTModel
import numpy as np
# load cifar10 (only small portion for demonstration purposes)

dataset = 'cifar10'
#train_ds, test_ds = load_dataset('cifar10', split=['train[:]', 'test[:]'])
train_ds, test_ds = load_dataset('mnist', split=['train[:]', 'test[:]'])

# split up training into training + validation
"""
splits = train_ds.train_test_split(test_size=0.1)
train_ds = splits['train']
val_ds = splits['test']
"""
from transformers import ViTImageProcessor
id2label = {id:label for id, label in enumerate(train_ds.features['label'].names)}
label2id = {label:id for id,label in id2label.items()}
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
model.eval()
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor)

image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]
print('size', size)
normalize = Normalize(mean=image_mean, std=image_std)

_val_transforms = Compose(
        [
            Resize(size),
            ToTensor(),
            normalize,
        ]
    )

def train_transforms(examples):
    #print('examples', examples)
    if dataset == 'mnist':
        examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['image']]
    else:
        examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['img']]
    return examples

def val_transforms(examples):
    #print('examples', examples)
    if dataset == 'mnist':
        examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
    else:
        examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]

    return examples


# Set the transforms
train_ds.set_transform(val_transforms)
#val_ds.set_transform(val_transforms)
test_ds.set_transform(val_transforms)

from torch.utils.data import DataLoader
import torch

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

train_dataloader = DataLoader(train_ds, collate_fn=collate_fn, batch_size=4)

from transformers import ViTForImageClassification
"""
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                  id2label=id2label,
                                                  label2id=label2id)
"""
layers_trans = list(model.children())
#model = nn.Sequential(*layers_trans[:-1])
#layers_trans = list(model.children())
#:print('model', layers_trans)
use_gpu = torch.cuda.is_available()

if use_gpu:
    # print("Currently using GPU {}".format(config.gpu_devices))
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(0)
    trainloader = DataLoader(train_ds, collate_fn = collate_fn, batch_size=200)
    testloader = DataLoader(test_ds, collate_fn=collate_fn, batch_size=200)
else:
    print("Currently using CPU (GPU is highly recommended)")

    pin_memory = True if use_gpu else False
    trainloader = DataLoader(train_ds, collate_fn = collate_fn, batch_size=200)
    testloader = DataLoader(test_ds, collate_fn, batch_size=200)

if use_gpu:
    model = nn.DataParallel(model).cuda()
model.eval()
os.makedirs("features/", exist_ok=True)
def extract_feature(loader):
    with torch.no_grad():
        pred_list, feature_list = [], []
        float_logit_list = []
        for batch_idx, batch in enumerate(loader):
            img_tuple, label = batch.items()
            img = img_tuple[1]
            #print('imgs,', img, 'label', label)
            if batch_idx == 0:
                print('image before pretrain', img.shape)
            if batch_idx % 50 == 0:
                print('batch {}/{}', batch_idx, len(loader))
            """
            from PIL import Image
            import requests
            url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
            img = Image.open(requests.get(url, stream=True).raw)
            """
            #print('input shape', img.shape)
            #inputs = feature_extractor(images=img, return_tensors='pt')
            output = model(img)
            features = output.last_hidden_state[:, 0, :]
            #print('feature', features.shape)
            # predA = predA.cpu()
            # print('features shape {} predA shape'.format(features.shape, predA.shape))
    
            feature_list.append(features.cpu())
            # print('predAs', predicted)
            # float_logit_list = (((torch.cat(float_logit_list, 0)).float()).numpy()).tolist()
            # float_logit_list = np.array(float_logit_list)

    feature_list = (((torch.cat(feature_list, 0)).float()).numpy()).tolist()
    feature_list = np.array(feature_list)
    return feature_list
features_train = extract_feature(trainloader)
features_test = extract_feature(testloader)
np.save("features/vit_cifar10_train.npy", features_train)
np.save("features/vit_cifar10_test.npy", features_test)
print('size of train', features_train.shape, 'size of test', features_test.shape)


