import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import argparse

class Image_dataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.sat_files = sorted([file_path for file_path in os.listdir(self.data_path)])
        self.transform = transform
        self.classes = 7
        # color mapping
        # self.mapping = {65535:0, 16776960:1, 16711935:2, 65280:3, 255:4, 16777215:5, 0:6}
        self.colormap2label = np.zeros(256**3)
        self.colormap2label[65535] = 0       # Urban
        self.colormap2label[16776960] = 1    # Agriculture
        self.colormap2label[16711935] = 2    # Rangeland 
        self.colormap2label[65280] = 3       # Forest land
        self.colormap2label[255] = 4         # Water
        self.colormap2label[16777215] = 5    # Barren land
        self.colormap2label[0] = 6           # Unknown 
        
    def __len__(self):
        return len(self.sat_files)
    
    def __getitem__(self, index):
        sat_image = Image.open(os.path.join(self.data_path, self.sat_files[index])).convert('RGB')
        if self.transform is not None:
            sat_image = self.transform(sat_image)
        
        return sat_image, self.sat_files[index]

class VGG16_FCN16s(nn.Module):
    def __init__(self, num_class=7):
        super(VGG16_FCN16s, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.features = vgg16.features
        self.pool4_index = 23
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv2d(128, num_class, 1, 1, 1)
        )
        self.conv_pool4 = nn.Conv2d(512, num_class, kernel_size=1)
        self.upsampling2 = nn.ConvTranspose2d(num_class, num_class, kernel_size=4, stride=2, bias=False)
        self.upsampling16 = nn.ConvTranspose2d(num_class, num_class, kernel_size=32, stride=16, bias=False)
        
    def forward(self, x):
        pool4 = x
        f = x
        for i in range(len(self.features)):
            f = self.features[i](f)
            if i == self.pool4_index:
                pool4 = f
            
        f = self.classifier(f)
        f = self.upsampling2(f)
        
        pool4 = self.conv_pool4(pool4)
        f = f[:, :, 1:1+pool4.size()[2], 1:1+pool4.size()[3]]
        f = f + pool4 
        
        f = self.upsampling16(f)
        cx = int((f.shape[3] - x.shape[3])/2)
        cy = int((f.shape[2] - x.shape[2])/2)
        f = f[:,:,cy:cy+x.shape[2], cx:cx+x.shape[3]]
        
        return f

def eval_one_epoch(model, data_loader, device="cpu"):
    model.eval()
    preds_name = []
    preds_mask = []
    # RGB mapping to label
    label2colormap = np.zeros((7,3))
    label2colormap[0] = np.array([0,255,255])
    label2colormap[1] = np.array([255,255,0])
    label2colormap[2] = np.array([255,0,255])
    label2colormap[3] = np.array([0,255,0])
    label2colormap[4] = np.array([0,0,255])
    label2colormap[5] = np.array([255,255,255])
    label2colormap[6] = np.array([0,0,0])
    
    for i, (inputs, names) in enumerate(data_loader):
        with torch.no_grad():
            inputs = inputs.to(device)
            logits = model(inputs)
            # (b, c, w, h) -> (b, w, h, c)
            logits = logits.permute(0,2,3,1)
            # (b, w, h, c) -> (b, w, h)
            result = logits.argmax(dim=-1)
            val_mask = result.cpu().numpy().squeeze(0)
            mask_image = label2colormap[val_mask]
            preds_mask.append(mask_image)
            preds_name.append(names[0])
    
    return preds_mask, preds_name

def main(args):
    img_dir = args['img_dir']
    vdata_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    validSet = Image_dataset(img_dir, vdata_transform)
    val_loader = DataLoader(dataset=validSet, batch_size=1, shuffle=False, num_workers=0)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGG16_FCN16s().to(device)
    model.load_state_dict(torch.load('best_seg.pth'))
    model.device = device

    output_dir = args['save_dir']
    preds_mask, preds_name = eval_one_epoch(model, val_loader, device)
    for i in range(len(preds_name)):
        image = Image.fromarray(preds_mask[i].astype(np.uint8))
        image.save(os.path.join(output_dir, preds_name[i].split('.')[0]+'.png'))

def parse_config():
    parser = argparse.ArgumentParser(description='hw1-2-segmentation')
    parser.add_argument('--img_dir', type=str, default='hw1_data/p2_data/testing/', help='input directory')
    parser.add_argument('--save_dir', type=str, default='output_hw2-2/', help='output directory')
    args = vars(parser.parse_args())
    return args

if __name__=='__main__':
    args = parse_config()
    main(args)
