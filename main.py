import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from listfile import listfile
from DataLoader import Trainloader
from senet.se_resnet import se_resnet50

train_imgs, train_labels, val_imgs, val_labels = listfile('./dataset')

TrainImgLoader = torch.utils.data.DataLoader(
         Trainloader(train_imgs, train_labels),
         batch_size=32, shuffle= True, num_workers= 8, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
         Trainloader(val_imgs, val_labels),
         batch_size=16, shuffle= False, num_workers= 8, drop_last=False)

model = se_resnet50(num_classes=29, pretrained=False)
model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.1)
loss_func = F.cross_entropy

def train(img, label):
    model.train()
    img = Variable(torch.FloatTensor(img))
    label = Variable(torch.LongTensor(label))
    img = img.cuda()
    label = label.cuda()

    optimizer.zero_grad()
    predict = model(img)
    predict = F.softmax(predict, dim=1)
    #print(predict)
    #print(label)
    loss = loss_func(predict, label)

    loss.backward()
    optimizer.step()

    return loss

def test(img, label):
    model.eval()
    img = Variable(torch.FloatTensor(img))
    label = Variable(torch.LongTensor(label))
    img = img.cuda()
    label = label.cuda()

    with torch.no_grad():
        logits = model(img)
        pred = logits.argmax(dim=1)
        correct = torch.eq(pred, label).sum().float().item()

    return correct/img.shape[0]





if __name__ == '__main__':
    epochs = 10
    for epoch in range(epochs):
        for idx, (img,label) in enumerate(TrainImgLoader):
            loss = train(img, label)
            print('Training----------epoch:{}/{}, iter:{}, loss:{}------------'.format(epoch,epochs,idx,loss ))

        for idx, (img,label) in enumerate(TestImgLoader):
            acc = test(img,label)
            print('Validating--------epoch:{}/{}, iter:{}, acc:{}%------------'.format(epoch,epochs,idx,acc*100 ))