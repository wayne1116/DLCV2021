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
label_mapping = {'Couch': 0, 'Helmet': 1, 'Refrigerator': 2, 'Alarm_Clock': 3, 'Bike': 4, 'Bottle': 5, 'Calculator': 6, 'Chair': 7, 'Mouse': 8, 'Monitor': 9, 'Table': 10, 'Pen': 11, 'Pencil': 12, 'Flowers': 13, 'Shelf': 14, 'Laptop': 15, 'Speaker': 16, 'Sneakers': 17, 'Printer': 18, 'Calendar': 19, 'Bed': 20, 'Knives': 21, 'Backpack': 22, 'Paper_Clip': 23, 'Candles': 24, 'Soda': 25, 'Clipboards': 26, 'Fork': 27, 'Exit_Sign': 28, 'Lamp_Shade': 29, 'Trash_Can': 30, 'Computer': 31, 'Scissors': 32, 'Webcam': 33, 'Sink': 34, 'Postit_Notes': 35, 'Glasses': 36, 'File_Cabinet': 37, 'Radio': 38, 'Bucket': 39, 'Drill': 40, 'Desk_Lamp': 41, 'Toys': 42, 'Keyboard': 43, 'Notebook': 44, 'Ruler': 45, 'ToothBrush': 46, 'Mop': 47, 'Flipflops': 48, 'Oven': 49, 'TV': 50, 'Eraser': 51, 'Telephone': 52, 'Kettle': 53, 'Curtains': 54, 'Mug': 55, 'Fan': 56, 'Push_Pin': 57, 'Batteries': 58, 'Pan': 59, 'Marker': 60, 'Spoon': 61, 'Screwdriver': 62, 'Hammer': 63, 'Folder': 64}
class_mapping = {0: 'Couch', 1: 'Helmet', 2: 'Refrigerator', 3: 'Alarm_Clock', 4: 'Bike', 5: 'Bottle', 6: 'Calculator', 7: 'Chair', 8: 'Mouse', 9: 'Monitor', 10: 'Table', 11: 'Pen', 12: 'Pencil', 13: 'Flowers', 14: 'Shelf', 15: 'Laptop', 16: 'Speaker', 17: 'Sneakers', 18: 'Printer', 19: 'Calendar', 20: 'Bed', 21: 'Knives', 22: 'Backpack', 23: 'Paper_Clip', 24: 'Candles', 25: 'Soda', 26: 'Clipboards', 27: 'Fork', 28: 'Exit_Sign', 29: 'Lamp_Shade', 30: 'Trash_Can', 31: 'Computer', 32: 'Scissors', 33: 'Webcam', 34: 'Sink', 35: 'Postit_Notes', 36: 'Glasses', 37: 'File_Cabinet', 38: 'Radio', 39: 'Bucket', 40: 'Drill', 41: 'Desk_Lamp', 42: 'Toys', 43: 'Keyboard', 44: 'Notebook', 45: 'Ruler', 46: 'ToothBrush', 47: 'Mop', 48: 'Flipflops', 49: 'Oven', 50: 'TV', 51: 'Eraser', 52: 'Telephone', 53: 'Kettle', 54: 'Curtains', 55: 'Mug', 56: 'Fan', 57: 'Push_Pin', 58: 'Batteries', 59: 'Pan', 60: 'Marker', 61: 'Spoon', 62: 'Screwdriver', 63: 'Hammer', 64: 'Folder'}

class OfficeImage(Dataset):
    def __init__(self, data_root, csv_file):
        self.data_root = data_root
        self.names = []
        data_df = pd.read_csv(csv_file).set_index("id")
        self.id = data_df.index

        for i in range(len(data_df)):
            name = data_df.loc[self.id[i], "filename"]
            self.names.append(name)
        
        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, index):
        fname = self.names[index]
        img = self.transform(os.path.join(self.data_root, fname))
        
        return img

def parse_args():
    parser = argparse.ArgumentParser(description="self supervised learning")
    parser.add_argument('--test_csv', type=str, help="Testing images csv file")
    parser.add_argument('--test_data_dir', type=str, help="Testing images directory")
    parser.add_argument('--output_csv', type=str, help="Output filename")

    return parser.parse_args()

def set_seed(SEED):
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(SEED)
    np.random.seed(SEED)

if __name__=='__main__':
    args = parse_args()
    set_seed(1129)

    dataset = OfficeImage(args.test_data_dir, args.test_csv)
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 65)
    model.load_state_dict(torch.load('resnetC.pth'))

    model.cuda()
    model.eval()
    
    result_id = []
    result_filename = []
    result_label = []

    with torch.no_grad():
        for i in range(len(dataset.names)):
            img = dataset.transform(os.path.join(dataset.data_root, dataset.names[i]))
            img = img.unsqueeze(0).cuda()

            logits = model(img)
            pred = torch.argmax(logits, dim=1).item()
            
            result_id.append(dataset.id[i])
            result_filename.append(dataset.names[i])
            result_label.append(class_mapping[pred])
    
    with open(args.output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        titles = ['id', 'filename', 'label']
        writer.writerow(titles)

        for i in range(len(result_id)):
            writer.writerow([str(result_id[i]), result_filename[i], result_label[i]])
