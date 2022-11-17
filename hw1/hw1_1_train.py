import os
import math
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.models as models
from PIL import Image

class Image_dataset(Dataset):
    def __init__(self, data_path, training, transform=None):
        self.transform = transform
        self.data_path = os.path.join(data_path, training)
        self.file_name = [file_path for file_path in os.listdir(self.data_path)]
    
    def __len__(self):
        return len(self.file_name)
    
    def __getitem__(self, index):
        image = Image.open(os.path.join(self.data_path, self.file_name[index])).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, int(self.file_name[index].split("_")[0]) 

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

def train_one_epoch(model, data_loader, criterion, optimizer, device="cpu"):
    train_acc = []
    train_loss = []
    model.train()
    for i, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        
        optimizer.zero_grad()
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        
        train_acc.append(acc.item())
        train_loss.append(loss.item())
    
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_acc) / len(train_acc)
    
    return train_loss, train_acc

def eval_one_epoch(model, data_loader, criterion, device="cpu"):
    model.eval()
    valid_acc = 0
    total = 0
    for i, (inputs, labels) in enumerate(data_loader):
        with torch.no_grad():
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
                
        loss = criterion(logits, labels)
        acc = (logits.argmax(dim=-1) == labels).float().sum()
        total += len(inputs)
        valid_acc += acc
    
    val_acc = valid_acc / total
    return val_acc

def set_seed(seed):
    myseed = seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)

def main():
    set_seed(9028)
    data_transform = transforms.Compose([
#         transforms.Resize((40,40)),
#         transforms.CenterCrop(32),
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    vdata_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    trainSet = Image_dataset(os.path.join('hw1_data', 'p1_data'), 'train_50', data_transform)
    validSet = Image_dataset(os.path.join('hw1_data', 'p1_data'), 'val_50', vdata_transform)
    data_loader = DataLoader(dataset=trainSet, batch_size=128, shuffle=True, num_workers=1)
    val_loader = DataLoader(dataset=validSet, batch_size=32, shuffle=False, num_workers=1)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGG().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    n_epochs = 15
    final_acc = 0

    for epoch in range(n_epochs):
        train_loss, train_acc = train_one_epoch(model, data_loader, criterion, optimizer, device)
        val_acc = eval_one_epoch(model, val_loader, criterion, device)
        
        if val_acc > final_acc:
            final_acc = val_acc
            print('Saving model (epoch = {:2d}, acc = {:.4f})'.format(epoch+1, val_acc))
            torch.save(model.state_dict(), 'best_acc.pth')
        
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] acc = {val_acc:.5f}")


if __name__=='__main__':
    main()

