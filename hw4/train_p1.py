import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

import csv
import random
import numpy as np
from PIL import Image
import pandas as pd
import torch.nn.functional as F
filenameToPILImage = lambda x: Image.open(x)

def set_seed(myseed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)

class MiniDataset(Dataset):
    def __init__(self, data_dir, mode):
        self.names = []
        self.label = []
        data_df = pd.read_csv(os.path.join(data_dir, mode+'.csv')).set_index("id")
        for i in range(38400):
            self.names.append(os.path.join(data_dir, mode, data_df.loc[i, "filename"]))
            self.label.append(data_df.loc[i, "label"])
        
        
        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.Resize((84,84)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        paths = self.names[index]
        label = self.label[index]
        image = self.transform(paths)
        return image, label

    def __len__(self):
        return len(self.names)

class CategoriesSampler():
    def __init__(self, n_batch, n_ways, n_shot):
        self.n_batch = n_batch
        self.n_ways = n_ways
        self.n_shot = n_shot
        self.classes = 64
        self.each_number = 600
        
        self.m_ind = []
        for i in range(self.classes):
            ind = []
            for j in range(self.each_number):
                ind.append(i*self.each_number+j)
            ind = torch.from_numpy(np.array(ind))
            self.m_ind.append(ind)
    
    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(self.classes)[:self.n_ways]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(self.each_number)[:self.n_shot]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch

def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

# model architecture: follow the TA setting
class Convnet(nn.Module):
    def __init__(self, in_channels=3, hid_channels=64, out_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(in_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, out_channels),
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

# for parametric function
class Parametric(nn.Module):
    def __init__(self, input_dim=3200, out_dim=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
    def forward(self, x):
        return self.mlp(x)

def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -torch.sqrt(((a - b)**2).sum(dim=2))
    return logits

def cosineSim_metric(a, b):
    qn = torch.norm(a, p=2, dim=1).detach()
    qn = qn.unsqueeze(1)
    a = a.div(qn.expand_as(a))
    
    qn = torch.norm(b, p=2, dim=1).detach()
    qn = qn.unsqueeze(1)
    b = b.div(qn.expand_as(b))
    
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = ((a * b)).sum(dim=2)
    
    return logits

def parametric_metric(parametric, a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    
    logits = parametric(torch.cat((a,b), 2))
    return logits.squeeze(2)

def main():
    set_seed(1129)
    n_ways, n_shot, n_query = 5, 1, 15 
    
    trainset = MiniDataset(os.path.join('hw4_data', 'mini'), 'train')
    train_sampler = CategoriesSampler(100, n_ways, n_shot+n_query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=0, pin_memory=True)
    
    model = Convnet().cuda()
    # parametric = Parametric().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # optimizer_p = torch.optim.Adam(parametric.parameters(), lr=1e-3)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    n_epoch = 300
    best_acc = 0
    for epoch in range(n_epoch):
        # lr_scheduler.step()
        model.train()
        # parametric.train()
        total_loss = 0
        total_acc = 0
        
        for i, (data, labels) in enumerate(train_loader):
            data_shot = data[:n_ways*n_shot].cuda()
            data_query = data[n_ways*n_shot:].cuda()
            
            proto = model(data_shot)
            proto = proto.reshape(n_shot, n_ways, -1).mean(dim=0)
            
            label = torch.arange(n_ways).repeat(n_query)
            label = label.type(torch.cuda.LongTensor)
            
            logits = euclidean_metric(model(data_query), proto)
            # logits = cosineSim_metric(model(data_query), proto)
            # logits = parametric_metric(parametric, model(data_query), proto)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)
            total_loss += loss.item()
            total_acc += acc
            
            optimizer.zero_grad()
            # optimizer_p.zero_grad()
            loss.backward()
            optimizer.step()
            # optimizer_p.step()
        
        print('epoch {} loss={:.4f} acc={:.4f}'.format(epoch, total_loss/len(train_loader), total_acc/len(train_loader)))
        if best_acc < total_acc/len(train_loader):
            best_acc = total_acc/len(train_loader)
            torch.save(model.state_dict(), 'feature_extraction1.pth')
            # torch.save(parametric.state_dict(), 'parametric.pth')
            print('saved !')

if __name__=='__main__':
    main()