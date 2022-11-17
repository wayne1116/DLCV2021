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

class Image_dataset(Dataset):
    def __init__(self, data_path, training, transform=None):
        self.data_path = os.path.join(data_path, training)
        files = sorted([file_path for file_path in os.listdir(self.data_path)])
        self.mask_files = [files[i] for i in range(0, len(files), 2)]
        self.sat_files = [files[i] for i in range(1, len(files), 2)]
        self.transform = transform
        self.classes = 7
        # color mapping
        self.mapping = {65535:0, 16776960:1, 16711935:2, 65280:3, 255:4, 16777215:5, 0:6, 16711935:2}
        self.colormap2label = np.zeros(256**3)
        self.colormap2label[65535] = 0
        self.colormap2label[16776960] = 1
        self.colormap2label[16711935] = 2
        self.colormap2label[65280] = 3
        self.colormap2label[255] = 4
        self.colormap2label[16777215] = 5
        self.colormap2label[0] = 6
        self.colormap2label[16711935] = 2
        
    def __len__(self):
        return len(self.mask_files)
    
    def __getitem__(self, index):
        sat_image = Image.open(os.path.join(self.data_path, self.sat_files[index])).convert('RGB')
        mask = Image.open(os.path.join(self.data_path, self.mask_files[index])).convert('RGB')
        mask = np.array(mask)

        # type of cv2 image
        mask = mask[:,:,::-1]
        # mask = cv2.imread(os.path.join(self.data_path, self.mask_files[index]))
        mask = mask[:,:,::-1] # reverse
#         mask_image = np.full((sat_image.size[0], sat_image.size[1]), 0, dtype=np.int)
        if self.transform is not None:
            sat_image = self.transform(sat_image)
        
        mask = mask.astype(np.int32)
        idx = mask[:,:,0]*256*256+mask[:,:,1]*256+mask[:,:,2]
        mask_image = self.colormap2label[idx]
        mask_image = np.array(mask_image, dtype=np.int)
        
        return sat_image, mask_image 

class VGG16_FCN32s(nn.Module):
    def __init__(self, num_class=7):
        super(VGG16_FCN32s, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.features = vgg16.features
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv2d(128, num_class, 1, 1, 1)
        )
        self.upsampling32 = nn.ConvTranspose2d(num_class, num_class, kernel_size=64, stride=32, bias=False)
        
    def forward(self, x):
        f = self.features(x)
        f = self.classifier(f)
        f = self.upsampling32(f)
        cx = int((f.shape[3] - x.shape[3])/2)
        cy = int((f.shape[2] - x.shape[2])/2)
        f = f[:,:,cy:cy+x.shape[2], cx:cx+x.shape[3]]
        
        return f

def read_masks(file_list):
    n_masks = len(file_list)
    masks = np.empty((n_masks, 512, 512))

    for i, file in enumerate(file_list):
        mask = file
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        masks[i, mask == 3] = 0  # (Cyan: 011) Urban land 
        masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land 
        masks[i, mask == 5] = 2  # (Purple: 101) Rangeland 
        masks[i, mask == 2] = 3  # (Green: 010) Forest land 
        masks[i, mask == 1] = 4  # (Blue: 001) Water 
        masks[i, mask == 7] = 5  # (White: 111) Barren land 
        masks[i, mask == 0] = 6  # (Black: 000) Unknown 

    return masks

def mean_iou_score(pred, labels):
    mean_iou = 0
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6
        print('class #%d : %1.5f'%(i, iou))
    print('\nmean_iou: %f\n' % mean_iou)

    return mean_iou

def train_one_epoch(model, data_loader, criterion, optimizer, device="cpu"):
    train_loss = []
    model.train()
    for i, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        logits = F.log_softmax(logits, dim=1)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    
    train_loss = sum(train_loss) / len(train_loss)
    return train_loss

def eval_one_epoch(model, data_loader, device="cpu"):
    model.eval()
    ground_truth = []
    predicts = []
    label2colormap = np.zeros((7,3))
    label2colormap[0] = np.array([0,255,255])
    label2colormap[1] = np.array([255,255,0])
    label2colormap[2] = np.array([255,0,255])
    label2colormap[3] = np.array([0,255,0])
    label2colormap[4] = np.array([0,0,255])
    label2colormap[5] = np.array([255,255,255])
    label2colormap[6] = np.array([0,0,0])
    
    for i, (inputs, labels) in enumerate(data_loader):
        with torch.no_grad():
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            logits = logits.permute(0,2,3,1)
            result = logits.argmax(dim=-1)
            val_mask = result.cpu().numpy().squeeze(0)
            mask_image = label2colormap[val_mask]
            labels = labels.cpu().numpy().squeeze(0)
            labels = label2colormap[labels]
            predicts.append(mask_image)
            ground_truth.append(labels)
    
    preds = read_masks(predicts)
    labels = read_masks(ground_truth)
    return mean_iou_score(preds, labels)

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
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    vdata_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    trainSet = Image_dataset(os.path.join('hw1_data', 'p2_data'), 'train', data_transform)
    validSet = Image_dataset(os.path.join('hw1_data', 'p2_data'), 'validation', vdata_transform)
    data_loader = DataLoader(dataset=trainSet, batch_size=24, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset=validSet, batch_size=1, shuffle=False, num_workers=0)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGG16_FCN32s().to(device)
    model.device = device

    criterion = nn.NLLLoss2d()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    
    n_epochs = 20
    final_miou = 0
    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, data_loader, criterion, optimizer, device)
        eval_miou = eval_one_epoch(model, val_loader, device)
        if final_miou < eval_miou:
            final_miou = eval_miou
            print('Saving model (epoch = {:2d}, miou = {:.4f})'.format(epoch+1, eval_miou))
            torch.save(model.state_dict(), 'best_seg.pth')
            
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}")

if __name__=='__main__':
    main()