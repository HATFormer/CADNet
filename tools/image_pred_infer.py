import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
import cv2
from tqdm import tqdm
import os
import torch.nn.init as init
from torch.utils.data import dataloader,dataset


def toArray(x):
    out = x[0].permute((1,2,0)).cpu().detach().numpy()
    out = np.uint8((out-np.min(out))/(np.max(out)-np.min(out))*255)
    return out
def visualize(x,y,out):
    fontsize = 18
    f, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].imshow(toArray(x)[...,::-1])
    ax[0].set_title('Input Image', fontsize=fontsize)

    ax[1].imshow(toArray(y)[...,::-1])
    ax[1].set_title('Reference Image', fontsize=fontsize)

    ax[2].imshow(toArray(out)[...,::-1])
    ax[2].set_title('Output Image', fontsize=fontsize)

    plt.show()



class predMatrixLinear(nn.Module):
    def __init__(self, num_layer):
        super(predMatrix, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layer):
            if i == 0:
                layer = nn.Linear(25,100,bias=True)#conv_block_nested(2, 16, 16)
            else:
                layer = nn.Linear(100,100,bias=True)#conv_block_nested(16, 16, 16)
            self.layers.append(layer)
        self.xcor_conv = nn.Linear(100,25,bias=True)#nn.Conv2d(100, 25, kernel_size=3, padding=1, bias=True)
        #self.ycor_conv = nn.Conv2d(16, 5, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        out = x.view(-1)
        for layer in self.layers:
            out = layer(out)
        #xcor = F.softmax(self.xcor_conv(out), dim=1)
        #ycor = F.softmax(self.xcor_conv(out), dim=1)
        #return xcor, ycor
        return self.xcor_conv(out).view(5,5)


class predImage(nn.Module):
    def __init__(self, num_layer,input_size):
        super(predImage, self).__init__()
        self.size = input_size
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(
                        nn.Conv2d(6,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2)))
        self.layers.append(nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2)))
        self.layers.append(nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2)))
        for i in range(num_layer):
            if i == 0:
                layer = nn.Conv2d(64, 64, kernel_size=input_size//8, bias=True)
            else:
                layer = nn.Conv2d(64, 64, kernel_size=1, bias=True)
            self.layers.append(layer)
        self.head_conv = nn.Conv2d(64, input_size*input_size*3, kernel_size=1, bias=True)
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, mode='fan_out',
                                     nonlinearity='relu')
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return self.head_conv(out).view(1,3,self.size,self.size)

def prepareMatrix(input_size):
    x = torch.eye(input_size).view((1,1,input_size,input_size))#torch.randn([1,1,50, 50])
    y = torch.from_numpy(x.numpy()[..., ::-1].copy())
    #x.requires_grad_(True)
    y.requires_grad_(True)

    data = torch.cat([x,y],dim=1)
    data.requires_grad_(True)
    return x,y.cuda(),data.cuda()

def toTensor(x):
    x = (x-np.min(x))/(np.max(x)-np.min(x))
    x = x.transpose([2,0,1])
    return torch.from_numpy(x).unsqueeze(0).float()

def prepareImage(input_size):
    x = toTensor(cv2.imread(r"D:\Datasets\DefectImage\Face\exp3\oksCuted_512\1-1.bmp"))
    y = toTensor(cv2.imread(r"D:\Datasets\DefectImage\Face\exp3\SynLCD_OKandNG\bg1_onlyGeo\mixed\A\3.png"))

    data = torch.cat([x,y],dim=1)

    return x,y,data.cuda()

if __name__ == '__main__':
    # 准备数据
    path = r'D:\Datasets\DefectImage\Face\exp3\tformsTest'
    input_size = 512
    x,y,data = prepareImage(input_size)
    model = predImage(1,input_size)
    model.load_state_dict(torch.load(os.path.join(path,'weights','weight.pth')))
    model.cuda()
    model.eval()

    out = model(data)

    visualize(x,y,out)

