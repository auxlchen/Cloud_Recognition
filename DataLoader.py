import pandas as pd
from torch.utils import data
import torch as t
import torch.nn as nn
import os
from PIL import Image
from torchvision.transforms import ToTensor,ToPILImage
from torch.autograd import variable as v


class Trainloader(data.Dataset):
    def __init__(self,trainfolder):
        self.csv = pd.read_csv('Train_label.csv')
        self.trainfolder = trainfolder
        self.totensor = ToTensor()
    def __getitem__(self,idx):
        imgpath = os.path.join(self.trainfolder,csv.FileName[idx])
        data = v(self.totensor(Image.open(imgpath)))
        label = v(t.tensor(int(csv.Code[idx])))
        return data,label