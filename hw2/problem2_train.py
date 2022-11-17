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
    def __init__(self, fnames, labels, transform):
        self.transform = transform
        self.fnames = sorted(fnames)
        self.num_samples = len(self.fnames)
        self.labels = [int(label[1]) for label in sorted(labels)]

    def __getitem__(self,idx):
        fname = self.fnames[idx]
        img = torchvision.io.read_image(fname)
        img = self.transform(img)
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return self.num_samples

def get_dataset(root):
    fnames = glob.glob(os.path.join(root, 'train', '*'))
    labels = []
    with open(os.path.join(root, 'train.csv'), 'r') as csvfile:
        reader = csv.reader(csvfile)
        labels = [row for row in reader]
        labels.pop(0)
    
    compose = [
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    transform = transforms.Compose(compose)
    dataset = DigitDataset(fnames, labels, transform)
    return dataset

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

def main():
    same_seeds(1123)
    z_dim = 100
    lr_G, lr_D = 0.0002, 0.0002
    G = Generator().cuda()
    D = Discriminator().cuda()
    opt_G = torch.optim.Adam(G.parameters(), lr=lr_G, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr_D, betas=(0.5, 0.999))
    
    # training hyper-parameter
    n_epoch = 100
    batch_size = 256
    adversarial_loss = torch.nn.BCELoss()
    auxiliary_loss = torch.nn.CrossEntropyLoss()
    
    dataset = get_dataset(os.path.join('hw2_data', 'digits', 'mnistm'))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    z_sample = Variable(torch.randn(100, z_dim)).cuda()
    label_sample = np.array([num for _ in range(10) for num in range(10)])
    label_sample = Variable(torch.cuda.LongTensor(label_sample))
    
    G.train()
    D.train()
    for epoch in range(n_epoch):
        total_loss_G = []
        total_loss_D = []
        for i, (imgs, labels) in enumerate(data_loader):
            imgs = imgs.cuda()
            bs = imgs.size(0)
            labels = labels.cuda()
            # ============ training generator ============
            valid = torch.ones((bs,1)).cuda()
            fake = torch.zeros((bs,1)).cuda()
            
            z = Variable(torch.randn(bs, z_dim)).cuda()
            gen_labels = Variable(torch.LongTensor(torch.randint(10, (bs,)))).cuda()
            gen_imgs = G(z, gen_labels)
            validity, pred_label = D(gen_imgs)
#             print(validity.size(), valid.size(), pred_label.size(), gen_labels.size())
            g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels))
            total_loss_G.append(g_loss.item())
            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()
            
            # ============ training discriminator ============
            real_pred, real_aux = D(imgs)
#             print(type(real_pred), type(valid), type(real_aux), type(labels))
            d_real_loss = 0.5 * (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) 
            fake_pred, fake_aux = D(gen_imgs.detach())
            d_fake_loss = 0.5 * (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels))
            d_loss = 0.5 * (d_real_loss + d_fake_loss) 
            total_loss_D.append(d_loss.item())
            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()
            
            # ============ evaulation ============
            pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
            gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
            d_acc = np.mean(np.argmax(pred, axis=1) == gt)
            
        
        # G.eval()
        # imgs = G(z_sample, label_sample)
        # f_imgs_sample = (imgs.data + 1) / 2.0
        # grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
        # plt.figure(figsize=(10,10))
        # plt.imshow(grid_img.permute(1, 2, 0))
        # plt.show()
        
        G.train()
        avg_loss_G = sum(total_loss_G) / len(total_loss_G)
        avg_loss_D = sum(total_loss_D) / len(total_loss_D)
        print(f"[ Train | {epoch + 1:03d}/{n_epoch:03d} ] avg_loss_G = {avg_loss_G:.5f}, avg_loss_D = {avg_loss_D:.5f}")

    torch.save(G.state_dict(), 'G_2.pth')
    torch.save(D.state_dict(), 'D_2.pth')

if __name__=='__main__':
    main()
    
