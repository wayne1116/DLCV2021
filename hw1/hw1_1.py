import os
import math
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.models as models
# from torchsummary import summary
from PIL import Image
import argparse
import csv

class VGG(nn.Module):
    def __init__(self, num_class=50):
        super(VGG, self).__init__()
        self.model = models.resnet101(pretrained=True)
#         for param in self.model.parameters():
#             param.requires_grad = False
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, num_class),
        )
        
    def forward(self, x):
        x = self.model(x)
        return x

class Image_dataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.transform = transform
        self.data_path = data_path
        self.file_name = [file_path for file_path in os.listdir(self.data_path)]
    
    def __len__(self):
        return len(self.file_name)
    
    def __getitem__(self, index):
        image = Image.open(os.path.join(self.data_path, self.file_name[index])).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, self.file_name[index]

def eval_one_epoch(model, data_loader, device="cpu"):
    model.eval()
    preds_name = []
    preds_label = []
    for i, (inputs, names) in enumerate(data_loader):
        with torch.no_grad():
            inputs = inputs.to(device)
            logits = model(inputs)
            pred = logits.argmax(dim=-1)
            preds_name.append(names[0])
            preds_label.append(int(pred.item()))

    return preds_name, preds_label

def main(args):
    input_directory = args['img_dir']
    vdata_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    validSet = Image_dataset(input_directory, vdata_transform)
    val_loader = DataLoader(dataset=validSet, batch_size=1, shuffle=False, num_workers=1)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGG().to(device)
    model.load_state_dict(torch.load('best_acc.pth'))

    print('='*20, 'Model predicting...', '='*20)
    preds_name, preds_label = eval_one_epoch(model, val_loader, device)
    
    print('='*20, 'Saving the csv file...', '='*20)
    output_csv = args['save_csv']
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_id', 'label'])
        for i in range(len(preds_name)):
            writer.writerow([preds_name[i], preds_label[i]])
    
    print('='*20, 'Finish!', '='*20)

def parse_config():
    parser = argparse.ArgumentParser(description='hw1-1-classification')
    parser.add_argument('--img_dir', type=str, default='hw1_data/p1_data/validation/', help='input directory')
    parser.add_argument('--save_csv', type=str, default='output_hw1-2/', help='output the csv file')
    args = vars(parser.parse_args())
    return args

if __name__=='__main__':
    args = parse_config()
    main(args)
