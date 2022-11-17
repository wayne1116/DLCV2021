import os
import numpy as np
import math
import glob
import csv
import random
import argparse

import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        
        self.img_size = 32
        self.channels = 3
        self.n_classes = 10
        
        self.conv_blocks = nn.Sequential(
            *discriminator_block(self.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
#         self.apply(weights_init_normal)
        ds_size = self.img_size // 2 ** 4
        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, self.n_classes), nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.n_classes = 10
        self.latent_dim = 100
        self.img_size = 32
        self.channels = 3
        
        self.label_emb = nn.Embedding(self.n_classes, self.latent_dim)

        self.init_size = self.img_size // 4 
        self.l1 = nn.Sequential(nn.Linear(self.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, self.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )
#         self.apply(weights_init_normal)

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

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
    parser = argparse.ArgumentParser(description='hw2-2-acgan')
    parser.add_argument('--save_dir', type=str, default='output_hw2-2/', help='output directory')
    args = vars(parser.parse_args())
    return args

def main(args):
    # fix random seed
    same_seeds(1123)
    save_directory = args['save_dir']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    G = Generator()
    G.load_state_dict(torch.load('G_2.pth'))
    G.eval()
    G.cuda()

    n_output = 1000
    z_dim = 100
    z_sample = Variable(torch.randn(n_output, z_dim)).cuda()
    label_sample = np.array([num for num in range(10) for _ in range(100)])
    label_sample = Variable(torch.cuda.LongTensor(label_sample))

    print('='*20, 'Model is generating...', '='*20)
    imgs = G(z_sample, label_sample)
    imgs_sample = (imgs.data + 1) / 2.0
    print('='*20, 'Saving the 1000 images...', '='*20)
    for i in range(n_output):
        im = transforms.ToPILImage()(imgs_sample[i]).resize((28,28)).convert('RGB')
        im.save(os.path.join(save_directory, '{:d}_{:03d}.png'.format(int(i/100), (i%100)+1)))
    print('='*20, 'Finish!', '='*20)

if __name__=='__main__':
    args = parse_config()
    main(args)
