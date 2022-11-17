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

from PIL import Image
filenameToPILImage = lambda x: Image.open(x)

# fix random seeds for reproducibility
def set_seed(SEED):
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(SEED)
    np.random.seed(SEED)

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# Model architecture
def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

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

# mini-Imagenet dataset
class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.Resize((84,84)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        image = self.transform(os.path.join(self.data_dir, path))
        return image

    def __len__(self):
        return len(self.data_df)

class GeneratorSampler(Sampler):
    def __init__(self, episode_file_path):
        episode_df = pd.read_csv(episode_file_path).set_index("episode_id")
        self.sampled_sequence = episode_df.values.flatten().tolist()

    def __iter__(self):
        return iter(self.sampled_sequence) 

    def __len__(self):
        return len(self.sampled_sequence)

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

def parametric_metric(parametric, a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    
    logits = parametric(torch.cat((a,b), 2))
    return logits.squeeze(2)

def predict(args, model, data_loader):
    model.eval()
    prediction_results = []
    with torch.no_grad():
        # each batch represent one episode (support data + query data)
        for i, data in enumerate(data_loader):

            # split data into support and query data
            support_input = data[:args.N_way * args.N_shot,:,:,:].cuda() 
            query_input   = data[args.N_way * args.N_shot:,:,:,:].cuda()

            # create the relative label (0 ~ N_way-1) for query data
            # label_encoder = {target[i * args.N_shot] : i for i in range(args.N_way)}
            # query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[args.N_way * args.N_shot:]])

            # extract the feature of support and query data
            proto = model(support_input)
            query = model(query_input)

            # calculate the prototype for each class according to its support data
            proto = proto.reshape(args.N_shot, args.N_way, -1).mean(dim=0)
            logits = euclidean_metric(query, proto)
            # logits = cosineSim_metric(query, proto)
            # logits = parametric_metric(parametric, query, proto)

            # classify the query data depending on the its distense with each prototype
            pred = torch.argmax(logits, dim=1)
            prediction_results.append(pred)

    return prediction_results

def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--N_way', default=5, type=int, help='N_way (default: 5)')
    parser.add_argument('--N_shot', default=1, type=int, help='N_shot (default: 1)')
    parser.add_argument('--N_query', default=15, type=int, help='N_query (default: 15)')
    parser.add_argument('--test_csv', type=str, help="Testing images csv file")
    parser.add_argument('--test_data_dir', type=str, help="Testing images directory")
    parser.add_argument('--testcase_csv', type=str, help="Test case csv")
    parser.add_argument('--output_csv', type=str, help="Output filename")

    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()
    set_seed(1129)

    test_dataset = MiniDataset(args.test_csv, args.test_data_dir)
    test_loader = DataLoader(
        test_dataset, batch_size=args.N_way * (args.N_query + args.N_shot),
        num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
        sampler=GeneratorSampler(args.testcase_csv))

    # load your model
    model = Convnet().cuda()
    # parametric = Parametric().cuda()
    model.load_state_dict(torch.load('feature_extraction.pth'))
    # parametric.load_state_dict(torch.load('parametric.pth'))
    prediction_results = predict(args, model, test_loader)

    # output your prediction to csv
    with open(args.output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        titles = ['episode_id']
        for i in range(args.N_way*args.N_query):
            titles.append('query'+str(i))
        writer.writerow(titles)

        for i in range(len(prediction_results)):
            value = [i]
            for j in range(len(prediction_results[i])):
                value.append(prediction_results[i][j].item())
            writer.writerow(value)
