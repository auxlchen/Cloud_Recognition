import pandas as pd
from torch.utils import data
import torch as t
import torch.nn as nn
import os
from PIL import Image
from torchvision.transforms import ToTensor,ToPILImage,Normalize
from torch.autograd import variable as v

imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}


class Trainloader(data.Dataset):
    def __init__(self,train_imgs, train_labels, training=False):
        self.train_imgs = train_imgs
        self.train_labels = train_labels
        self.totensor = ToTensor()
        self.normalize = Normalize(**imagenet_stats)
    def __getitem__(self,idx):
        img = Image.open(self.train_imgs[idx])
        img = img.convert('RGB')
        img = img.resize((256,256), Image.BILINEAR)
        data = self.totensor(img)
        data = self.normalize(data)
        label = int(self.train_labels[idx])-1
        return data,label

    def __len__(self):
        return len(self.train_imgs)