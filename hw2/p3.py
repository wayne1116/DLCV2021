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
import argparse

class DigitDataset(Dataset):
    def __init__(self, root, fnames, transform=None):
        self.transform = transform
        self.root = root
        self.fnames = sorted(fnames)
        self.num_samples = len(self.fnames)

    def __getitem__(self,idx):
        fname = self.fnames[idx]
        img = torchvision.io.read_image(os.path.join(self.root, fname))
        
        if img.size()[0] == 1:
            img = img.repeat(3,1,1)
        
        if self.transform is not None:
            img = self.transform(img)
            if img.size()[0] == 1:
                img = torch.cat((img,img,img), 0)
        
        return img, self.fnames[idx]

    def __len__(self):
        return self.num_samples

def get_dataset(root, target_domain):
    fnames = []
    for i in os.listdir(root):
        fnames.append(i)
    # fnames = glob.glob(os.path.join(root, '*'))
    compose = None
    if target_domain == 'mnistm' or target_domain == 'usps':
        compose = [
            transforms.ToPILImage(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    else:
        compose = [
            transforms.Grayscale(),
            transforms.ToPILImage(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
        ]
    transform = transforms.Compose(compose)
    dataset = DigitDataset(root, fnames, transform)
    return dataset

class FeatureExtractor1(nn.Module):
    def __init__(self):
        super(FeatureExtractor1, self).__init__()
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

class LabelPredictor1(nn.Module):
    def __init__(self):
        super(LabelPredictor1, self).__init__()
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

class FeatureExtractor2(nn.Module):
    def __init__(self):
        super(FeatureExtractor2, self).__init__()
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
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
    def forward(self, x):
        x = self.conv(x).squeeze()
        return x

class LabelPredictor2(nn.Module):
    def __init__(self):
        super(LabelPredictor2, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(256, 256),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, h):
        c = self.layer(h)
        return c

def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def parse_config():
    parser = argparse.ArgumentParser(description='hw2-3-dann')
    parser.add_argument('--test_path', type=str, default='hw2_data/digits/mnistm/test', help='path to testing images in the target domain')
    parser.add_argument('--target_domain', type=str, default='mnistm', help='a string that indicates the name of the target domain')
    parser.add_argument('--output_csv', type=str, default='test_pred.csv', help='path to your output prediction file')
    args = vars(parser.parse_args())
    return args

def eval(args):
    same_seeds(1123)
    output_csv = args['output_csv']
    test_path = args['test_path']
    target_domain = args['target_domain']
    target_test = get_dataset(test_path, target_domain)
    dataloader = DataLoader(target_test, batch_size=1, shuffle=False)

    if target_domain == 'mnistm' or target_domain == 'usps':
        feature_extractor = FeatureExtractor1()
        label_predictor = LabelPredictor1()
        if target_domain == 'mnistm':
            feature_extractor.load_state_dict(torch.load('extractor_model_s2m.pth'))
            label_predictor.load_state_dict(torch.load('predictor_model_s2m.pth'))
        else:
            feature_extractor.load_state_dict(torch.load('extractor_model_m2u.pth'))
            label_predictor.load_state_dict(torch.load('predictor_model_m2u.pth'))
    else:
        feature_extractor = FeatureExtractor2()
        label_predictor = LabelPredictor2()
        feature_extractor.load_state_dict(torch.load('extractor_model_u2s.pth'))
        label_predictor.load_state_dict(torch.load('predictor_model_u2s.pth'))
        
    feature_extractor.cuda().eval()
    label_predictor.cuda().eval()

    print('='*20, 'Model is predicting...', '='*20)
    labels = []
    for i, (inputs, name) in enumerate(dataloader):
        with torch.no_grad():
            inputs = inputs.cuda()
            features = feature_extractor(inputs)
            logits = label_predictor(features)
            pred = logits.argmax(dim=-1).item()
            labels.append([name[0], int(pred)])

    print('='*20, 'save csv file...', '='*20)
    labels = sorted(labels, key=lambda s:s[0])    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_name', 'label'])
        for i in range(len(labels)):
            writer.writerow([labels[i][0], labels[i][1]])

    print('='*20, 'Finish !', '='*20)

if __name__=='__main__':
    args = parse_config()
    eval(args)

