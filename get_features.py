import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

import argparse
import numpy as np
from math import ceil
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="Name of the dataset, for example Office_Home. Must be the same name of the dataset folder.", type=str)
parser.add_argument("--data_path", help="Path of the data, for example ../datasets", default='../datasets', type=str)
parser.add_argument("-d", "--domain", help="Dataset domain names.",
                    type=str, action='append')
args = parser.parse_args()

##load datasets
domain_names = [os.path.join(args.data_path, args.dataset, x) for x in args.domain]

train_transform = T.Compose([T.Resize(256), T.RandomCrop(224), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
train_datasets = [torchvision.datasets.ImageFolder(x, transform=train_transform) for x in domain_names]

test_transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
test_datasets = [torchvision.datasets.ImageFolder(x, transform=test_transform) for x in domain_names]

##model initialization
model = torchvision.models.resnet50(pretrained=True)
model.fc = nn.Identity()
model = model.cuda(0)

##get features
model.eval()
with torch.no_grad():
    for i, (name, train, test) in enumerate(zip(domain_names, train_datasets, test_datasets)):
            train_loader = torch.utils.data.DataLoader(train, batch_size=20, shuffle=False, num_workers=4)
            test_loader  = torch.utils.data.DataLoader(test,  batch_size=20, shuffle=False, num_workers=4)
            
            features, labels = [], []
            with tqdm(total=len(train_loader), ascii=True, desc='Train ' + os.path.basename(name)) as pbar:
                for x, y in train_loader:
                    x = x.cuda(0)
                    feat = model(x).cpu()
                    features.append(feat)
                    labels.append(y)
                    pbar.update(1)

            train_features = torch.cat(features, dim=0)
            train_labels = torch.cat(labels, dim=0)
            
            features, labels = [], []
            with tqdm(total=len(test_loader), ascii=True, desc='Test ' + os.path.basename(name)) as pbar:
                for x, y in test_loader:
                    x = x.cuda(0)
                    feat = model(x).cpu()
                    features.append(feat)
                    labels.append(y)
                    pbar.update(1)

            test_features = torch.cat(features, dim=0)
            test_labels = torch.cat(labels, dim=0)

            print(train_features.shape, train_labels.shape)
            print(test_features.shape, test_labels.shape)

            np.savez(os.path.join(args.data_path, '_'.join([args.dataset.lower(), args.domain[i].lower()])), 
                      train_x=train_features.numpy(),
                      train_y=train_labels.numpy(),
                      test_x=test_features.numpy(),
                      test_y=test_labels.numpy())
