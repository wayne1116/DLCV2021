import os
import csv
import torch
import argparse
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pytorch_pretrained_vit import ViT
from torchvision import transforms

def parse_config():
    parser = argparse.ArgumentParser(description='hw3-1-vit')
    parser.add_argument('--folder', type=str, default='hw3_data/p1_data/val/', help='path to the folder containing test images')
    parser.add_argument('--save_csv', type=str, default='output/pred.csv', help='path of the ouput csv file')
    args = vars(parser.parse_args())
    
    return args

def set_seed(myseed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)

class PetDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.transform = transform
        self.data_path = data_path
        self.filename = [file_path for file_path in os.listdir(self.data_path)]
    
    def __len__(self):
        return len(self.filename)
    
    def __getitem__(self, index):
        image = Image.open(os.path.join(self.data_path, self.filename[index])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        return image, self.filename[index]

def eval_one_epoch(model, data_loader):
    model.eval()
    preds_name = []
    preds_label = []
    for i, (inputs, names) in enumerate(data_loader):
        with torch.no_grad():
            inputs = inputs.cuda()
            logits = model(inputs)
            pred = logits.argmax(dim=-1)
            preds_name.append(names[0])
            preds_label.append(int(pred.item())) 
    
    return preds_name, preds_label
    
def main(args):
    set_seed(1129)
    input_folder = args['folder']
    # build the vision transformer model
    model = ViT('B_16', num_classes=37)
    model.load_state_dict(torch.load('best_vit.pth'))
    model = model.cuda()

    # the dataloader for validation set
    data_transform = transforms.Compose([
        transforms.Resize(model.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
    ])
    validSet = PetDataset(input_folder, data_transform)
    val_loader = DataLoader(dataset=validSet, batch_size=1, shuffle=False, num_workers=1)
    
    # model prediction on validation set
    print('='*20, 'Model prediction...', '='*20)
    preds_name, preds_label = eval_one_epoch(model, val_loader)

    # output the csv file
    output_csv = args['save_csv']
    print('='*20, 'Output the csv file...', '='*20)
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'label'])
        for i in range(len(preds_name)):
            writer.writerow([preds_name[i], str(preds_label[i])])
    print('='*20, 'Finish!', '='*20)

if __name__=='__main__':
    args = parse_config()
    main(args)