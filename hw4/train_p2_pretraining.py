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
import pandas as pd
import glob

from PIL import Image
from byol_pytorch import BYOL
from torchvision import models
filenameToPILImage = lambda x: Image.open(x)

class MiniImageNet(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.fname = glob.glob(os.path.join(data_root, '*'))
        
        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    def __len__(self):
        return len(self.fname)
    def __getitem__(self, index):
        fname = self.fname[index]
        img = self.transform(fname)
        
        return img

# set the random seed
SEED = 1129
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

# model backbone
resnet = models.resnet50(pretrained=False)
resnet.fc = nn.Linear(resnet.fc.in_features, 65)
# self-supervised learning method: BYOL
learner = BYOL(resnet, image_size=128, hidden_layer='avgpool').cuda()
optimizer = torch.optim.Adam(learner.parameters(), lr=3e-4)

# training setting & dataset
n_epoch = 100
batch_size = 128
dataset = MiniImageNet(os.path.join('hw4_data', 'mini', 'train'))
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
learner.train()

for epoch in range(n_epoch):
    total_loss = 0
    for i, img in enumerate(data_loader):
        img = img.cuda()
        
        loss = learner(img)
        print(loss, end='\r')
        optimizer.zero_grad()
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        learner.update_moving_average()
    
    # save the model
    print('epoch {} loss={:.4f}'.format(epoch, total_loss/len(data_loader)))
    torch.save(resnet.state_dict(), 'resnet_test.pth')
