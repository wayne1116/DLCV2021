import os
import numpy as np
import math
import glob
import csv
import random

import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

class DigitDataset(Dataset):
    def __init__(self, fnames, labels, transform=None):
        self.transform = transform
        self.fnames = sorted(fnames)
        self.num_samples = len(self.fnames)
        self.labels = [int(label[1]) for label in sorted(labels)]

    def __getitem__(self,idx):
        fname = self.fnames[idx]
        img = torchvision.io.read_image(fname)
        if img.size()[0] == 1:
            img = img.repeat(3,1,1)

        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return self.num_samples

def get_dataset(root, mode):
    fnames = glob.glob(os.path.join(root, mode, '*'))
    labels = []
    with open(os.path.join(root, mode+'.csv'), 'r') as csvfile:
        reader = csv.reader(csvfile)
        labels = [row for row in reader]
        labels.pop(0)
    
    compose = [
        transforms.ToPILImage(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    transform = transforms.Compose(compose)
    dataset = DigitDataset(fnames, labels, transform)
    return dataset

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            # input: bs x 3 x 28 x 28
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # input: bs x 64 x 14 x 14
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # input: bs x 128 x 7 x 7
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # input: bs x 256 x 3 x 3
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
    def forward(self, x):
        x = self.conv(x).squeeze()
        return x

class LabelPredictor(nn.Module):
    def __init__(self):
        super(LabelPredictor, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, h):
        c = self.layer(h)
        return c

class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1),
        )

    def forward(self, h):
        y = self.layer(h)
        return y

def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def set_lamb(epoch, iteration, one_epoch_iteration, total_epoch):
  training_progress = (epoch * one_epoch_iteration + iteration)/(total_epoch*one_epoch_iteration)
  return 2*(2/(1+math.exp(-10*training_progress))-1)

def train_epoch(source_dataloader, target_dataloader, epoch, total_epoch):
    # D loss: Domain Classifier的loss
    # F loss: Feature Extrator & Label Predictor的loss
    running_D_loss, running_F_loss = 0.0, 0.0
    total_hit, total_num = 0.0, 0.0
    total_target_hit, total_target_num = 0.0, 0.0
    count = 0
    for i, ((source_data, source_label), (target_data, target_label)) in enumerate(zip(source_dataloader, target_dataloader)):
        source_data = source_data.cuda()
        source_label = source_label.cuda()
        target_data = target_data.cuda()
        target_label = target_label.cuda()
        
        # Mixed the source data and target data, or it'll mislead the running params
        #   of batch_norm. (runnning mean/var of soucre and target data are different.)
        mixed_data = torch.cat([source_data, target_data], dim=0)
        domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).cuda()
        domain_label[:source_data.shape[0]] = 1

        # Step 1 : train domain classifier
        optimizer_D.zero_grad()
        feature = feature_extractor(mixed_data)
        domain_logits = domain_classifier(feature.detach())
        loss = domain_criterion(domain_logits, domain_label)
        running_D_loss += loss.item()
        loss.backward()
        optimizer_D.step()

        # Step 2 : train feature extractor and label classifier
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()
        class_logits = label_predictor(feature[:source_data.shape[0]])
        domain_logits = domain_classifier(feature)
        loss = class_criterion(class_logits, source_label) - set_lamb(epoch, count, len(source_dataloader), total_epoch) * domain_criterion(domain_logits, domain_label)
        running_F_loss += loss.item()
        loss.backward()
        optimizer_F.step()
        optimizer_C.step()

        total_hit += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()
        total_num += source_data.shape[0]
        print(i, end='\r')
        count += 1
        
        label_predictor.eval()
        target_class_logits = label_predictor(feature[source_data.shape[0]:])
        total_target_hit += torch.sum(torch.argmax(target_class_logits, dim=1) == target_label).item()
        total_target_num += target_data.shape[0]
        
        label_predictor.train()
    
    return running_D_loss / (i+1), running_F_loss / (i+1), total_hit / total_num, total_target_hit / total_target_num

if __name__=='__main__':
    same_seeds(1123)
    source_dataset = get_dataset(os.path.join('hw2_data', 'digits', 'svhn'), 'train')
    target_dataset = get_dataset(os.path.join('hw2_data', 'digits', 'mnistm'), 'train')
    source_dataloader = DataLoader(source_dataset, batch_size=64, shuffle=True)
    target_dataloader = DataLoader(target_dataset, batch_size=64, shuffle=True)
    
    feature_extractor = FeatureExtractor().cuda()
    label_predictor = LabelPredictor().cuda()
    domain_classifier = DomainClassifier().cuda()
    
    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()
    
    optimizer_F = torch.optim.Adam(feature_extractor.parameters(), lr=0.001)
    optimizer_C = torch.optim.Adam(label_predictor.parameters(), lr=0.001)
    optimizer_D = torch.optim.Adam(domain_classifier.parameters(), lr=0.001)
    
    n_epoch = 100
    best_acc = 0
    for epoch in range(n_epoch):
        train_D_loss, train_F_loss, train_acc, target_acc = train_epoch(source_dataloader, target_dataloader, epoch, n_epoch)
        if best_acc < target_acc:
            best_acc = target_acc
            print('save!')
            torch.save(feature_extractor.state_dict(), f'extractor_model_s2m.pth')
            torch.save(label_predictor.state_dict(), f'predictor_model_s2m.pth')
        
        print('epoch {:>3d}: train D loss: {:6.4f}, train F loss: {:6.4f}, acc {:6.4f} acc {:6.4f}'.format(epoch, train_D_loss, train_F_loss, train_acc, target_acc))