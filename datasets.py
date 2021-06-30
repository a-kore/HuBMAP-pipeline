from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torchvision.transforms as transforms
import torch

class HuBMAPDatset(Dataset):
    def __init__(self, path='./data256', train=True, split=0.8, transform=None):
        self.path = path
        self.images = os.listdir(path + '/train')
        self.images = [os.path.join(path + '/train', self.images[i]) for i in range(len(self.images))]
        self.masks = os.listdir(path + '/masks')
        self.masks = [os.path.join(path + '/masks', self.masks[i]) for i in range(len(self.masks))]
        if train == True:
            self.images = self.images[0:int(len(self.images) * split)]
            self.masks = self.masks[0:int(len(self.masks) * split)]
        else:
            self.images = self.images[int(len(self.images) * (1-split)):]
            self.masks = self.masks[int(len(self.masks) * (1-split)):]
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.ToTensor()
        self.mask_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        mask = Image.open(self.masks[idx])
        if self.transform:
            img = self.transform(img)
            mask = self.mask_transform(mask)
            if torch.sum(mask) > 0:
                mask = mask/mask.max()
        return img, mask
