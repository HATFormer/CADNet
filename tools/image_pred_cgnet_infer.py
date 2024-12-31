import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
import cv2
from tqdm import tqdm
import os
import torch.nn.init as init
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from mmseg.models.backbones.cgnet import CGNet

class mydataset(Dataset):
    def __init__(self, pathRef, pathGt, input_size):
        super().__init__()
        names_ref = os.listdir(pathRef)
        names_gt = os.listdir(pathGt)
        self.ref = [os.path.join(pathRef, x) for x in names_ref]
        self.gt = [os.path.join(pathGt, x) for x in names_gt]
        self.tforms = transforms.Compose([transforms.Resize((input_size, input_size),
                                                            interpolation=transforms.InterpolationMode.BICUBIC),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
        ok = Image.open(r"./data/SynLCD_with_geoTforms/oksCuted_512/1-1.bmp")
        self.ok = self.tforms(ok)

    def __getitem__(self, idx):
        ref = Image.open(self.ref[idx])
        gt = Image.open(self.gt[idx])

        ref = self.tforms(ref)
        gt = self.tforms(gt)

        return self.ok, ref, gt  # ,torch.cat([self.ok,ref],dim=0)

    def __len__(self):
        return len(self.ref)


def toArray(x):
    out = x[0].permute((1, 2, 0)).cpu().detach().numpy()
    out = np.uint8((out - np.min(out)) / (np.max(out) - np.min(out)) * 255)
    return out


def visualize(x, y, out, LOSS):
    fontsize = 18
    f, ax = plt.subplots(2, 2, figsize=(8, 8))

    ax[0, 0].imshow(toArray(x))
    ax[0, 0].set_title('Input Image', fontsize=fontsize)

    ax[0, 1].imshow(toArray(y))
    ax[0, 1].set_title('Reference Image', fontsize=fontsize)

    ax[1, 0].imshow(toArray(out))
    ax[1, 0].set_title('Output Image', fontsize=fontsize)

    # ax[1, 1].imshow(np.abs(toArray(y)[...,::-1]-toArray(out)[...,::-1]))
    # ax[1, 1].set_title('Difference Image', fontsize=fontsize)
    ax[1, 1].plot(LOSS)
    ax[1, 1].set_title('MSE loss', fontsize=fontsize)
    ax[1, 1].set_xlabel('Step', fontsize=fontsize)

    plt.show()


# class conv_block_nested(nn.Module):
#     def __init__(self, in_ch, mid_ch, out_ch):
#         super(conv_block_nested, self).__init__()
#         self.activation = nn.ReLU(inplace=True)
#         self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
#         self.bn1 = nn.BatchNorm2d(mid_ch)
#         self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
#         self.bn2 = nn.BatchNorm2d(out_ch)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         identity = x
#         x = self.bn1(x)
#         x = self.activation(x)
#
#         x = self.conv2(x)
#         x = self.bn2(x)
#         output = self.activation(x + identity)
#         return output

class predMatrixLinear(nn.Module):
    def __init__(self, num_layer):
        super(predMatrix, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layer):
            if i == 0:
                layer = nn.Linear(25, 100, bias=True)  # conv_block_nested(2, 16, 16)
            else:
                layer = nn.Linear(100, 100, bias=True)  # conv_block_nested(16, 16, 16)
            self.layers.append(layer)
        self.xcor_conv = nn.Linear(100, 25, bias=True)  # nn.Conv2d(100, 25, kernel_size=3, padding=1, bias=True)
        # self.ycor_conv = nn.Conv2d(16, 5, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        out = x.view(-1)
        for layer in self.layers:
            out = layer(out)
        # xcor = F.softmax(self.xcor_conv(out), dim=1)
        # ycor = F.softmax(self.xcor_conv(out), dim=1)
        # return xcor, ycor
        return self.xcor_conv(out).view(5, 5)

class reshape(nn.Module):
    def __init__(self,c,h,w):
        super(reshape, self).__init__()
        self.c, self.h, self.w = c,h,w
    def forward(self,x):
        return x.view(x.shape[0],self.c,self.h,self.w)
class predImage(nn.Module):
    def __init__(self, num_layer, input_size):
        super(predImage, self).__init__()
        self.size = input_size
        out_channels = [38, 134, 256]
        for i in range(num_layer):
            if i == 0:
                layer = nn.Conv2d(64, input_size, kernel_size=input_size // 8, bias=True)
            else:
                layer = nn.Conv2d(64, 64, kernel_size=1, bias=True)
            self.layers.append(layer)

        self.relation_conv = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, bias=True),
            nn.BatchNorm2d(512, momentum=1, affine=True),
            nn.PReLU(512),
            nn.Conv2d(512, 64, kernel_size=1, bias=True),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.PReLU(64),
            nn.Conv2d(64, 64*64, kernel_size=64, bias=True,groups=32),
            # nn.PReLU(),
            # nn.Conv2d(input_size, input_size * input_size * 3, kernel_size=1, bias=True),
            reshape(1,64,64)
        )
        self.head_conv1 = nn.Conv2d(256, 3, kernel_size=3, padding=1, bias=True)
        self.head_conv = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='nearest', align_corners=None),
            # nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=True,groups=3),
            nn.ConvTranspose2d(3, 3, 2, stride=2,groups=3),
            nn.BatchNorm2d(3, momentum=1, affine=True),
            nn.ReLU(),
            # nn.Upsample(scale_factor=2, mode='nearest', align_corners=None),
            # nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=True,groups=3),
            nn.ConvTranspose2d(3, 3, 2, stride=2,groups=3),
            nn.BatchNorm2d(3, momentum=1, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 3, 2, stride=2,groups=3),
        )
        ## resize逐级上采样
        # torch.nn.Upsample(scale_factor=2, mode='bi', align_corners=None),
        #                 nn.Conv2d(16, 3, kernel_size=3, padding=1, bias=True),
        # nn.BatchNorm2d(64*3, momentum=1, affine=True),
        # nn.ReLU(),
        # nn.Conv2d(64, input_size*input_size*3, kernel_size=1, bias=True)
        self._initialize_weights()

        self.model = CGNet()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, mode='fan_out',
                                     nonlinearity='relu')
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, ok,ref):
        H,W = ok.shape[1:3]
        x = self.model(ok, ref)
        # out[0] = [F.interpolate(out[0][i],size=ok.shape[2:], mode='bilinear')
        #        for i in range(len(out[0]))]
        # out[1] = [F.interpolate(out[1][i],size=ok.shape[2:], mode='bilinear')
        #        for i in range(len(out[1]))]
        out = self.relation_conv(torch.cat([x[0][-1],x[1][-1]], dim=1))
        out = torch.matmul(x[0][-1],out)
        out = self.head_conv1(out)
        #out = self.head_conv2(out)
        out = F.interpolate(out, size=ok.shape[2:], mode='bilinear')
        # out = [F.interpolate(out[i],size=ok.shape[2:], mode='bilinear')
        #        for i in range(len(out))]
        # out = torch.cat(out, dim=1)

        return out#.view(out.shape[0], 3, self.size, self.size)


def prepareMatrix(input_size):
    x = torch.eye(input_size).view((1, 1, input_size, input_size))  # torch.randn([1,1,50, 50])
    y = torch.from_numpy(x.numpy()[..., ::-1].copy())
    # x.requires_grad_(True)
    y.requires_grad_(True)

    data = torch.cat([x, y], dim=1)
    data.requires_grad_(True)
    return x, y.cuda(), data.cuda()


def toTensor(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    x = x.transpose([2, 0, 1])
    return torch.from_numpy(x).unsqueeze(0).float()


def prepareImage(input_size):
    x = toTensor(cv2.imread(r"D:\Datasets\DefectImage\Face\exp3\oksCuted_512\1-1.bmp"))
    y = toTensor(cv2.imread(r"D:\Datasets\DefectImage\Face\exp3\SynLCD_OKandNG\bg1_onlyGeo\mixed\A\3.png"))
    gt = toTensor(cv2.imread(r"D:\Datasets\DefectImage\Face\exp3\SynLCD_OKandNG\bg1_onlyGeo\mixed\B\3.png"))
    # x.requires_grad_(True)
    y.requires_grad_(True)
    data = torch.cat([x, y], dim=1)
    data.requires_grad_(True)
    return x, y, gt.cuda(), data.cuda()


if __name__ == '__main__':
    # 准备数据
    path = r'./work_dirs/tformsTest'
    input_size = 512
    # x,y,gt,data = prepareImage(input_size)
    model = predImage(0,input_size)
    model.load_state_dict(torch.load(os.path.join(path, 'weights', 'weight_3.pth')),strict=False)
    model.cuda()
    model.eval()


    criteria = nn.L1Loss().cuda()
    dset = mydataset(r'./data/SynLCD_OKandNG/bg2/mixed/A',
                     r'./data/SynLCD_OKandNG/bg2/mixed/B', input_size)

    train_data = DataLoader(dset,
                            shuffle=True,
                            batch_size=4,
                            num_workers=4)
    LOSS = []
    with tqdm(train_data) as t:
        for ok, ref, gt in t:
            # Zero the gradient
            # data = data.cuda()
            gt = gt.cuda()
            out = model(ok.cuda(), ref.cuda())
            loss = criteria(out, gt)
            loss.backward()
            LOSS.append(loss.item())
            t.set_description('mean loss:{:.2f}'.format(np.mean(LOSS)))
            cv2.imwrite(os.path.join(path, '%d.png' % len(LOSS)), toArray(out)[..., ::-1])
            cv2.imwrite(os.path.join(path, '%d_gt.png' % len(LOSS)), toArray(gt)[..., ::-1])

    visualize(ok, ref, out, LOSS)

    print(loss.item())
