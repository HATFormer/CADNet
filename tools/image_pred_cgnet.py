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


class ChannelAttention(nn.Module):
    def __init__(self, input_size, ratio = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((1,input_size,input_size))
        self.max_pool = nn.AdaptiveMaxPool3d((1,input_size,input_size))
        self.fc1 = nn.Conv2d(1,ratio,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(ratio, 1,1,bias=False)
        self.sigmod = nn.Sigmoid()
    def forward(self,x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)

class reshape(nn.Module):
    def __init__(self, c, h, w):
        super(reshape, self).__init__()
        self.c, self.h, self.w = c, h, w

    def forward(self, x):
        return x.view(x.shape[0], self.c, self.h, self.w)
class ChannelShuffle(nn.Module):
    def __init__(self,group):
        super(ChannelShuffle,self).__init__()
        self.group = group
    def forward(self,x):
        b,c,h,w = x.shape
        if c % self.group != 0:
            raise ValueError('in_channels must be divisible by groups')
        x = x.view(b,self.group, c//self.group,h,w)
        x = x.permute(0,2,1,3,4)

        return x.reshape(b,c,h,w)

class predImage(nn.Module):
    def __init__(self, num_upsample, input_size):
        super(predImage, self).__init__()
        self.size = input_size
        out_channels = [11, 35, 64]
        # self.ok_feature_conv = ChannelAttention(input_size=input_size)
        # self.ref_feature_conv = ChannelAttention(input_size=input_size)
        self.feature_conv = nn.Sequential(
            ChannelShuffle(2), # 每相邻两个分别来自OK和Ref
            nn.Conv2d(out_channels[-1]*2,out_channels[-1],kernel_size=3,
                      padding=1,groups=out_channels[-1]),
            nn.BatchNorm2d(out_channels[-1], momentum=1, affine=True),
            nn.ReLU(),
            nn.Conv2d(out_channels[-1], 2, kernel_size=1, groups=2),
            nn.BatchNorm2d(2, momentum=1, affine=True),
            nn.ReLU(),
        )
        self.relation_conv = nn.Sequential(
            nn.Conv2d(2,  input_size, kernel_size=input_size, bias=True),
            nn.PReLU(),
            nn.Conv2d(input_size, input_size*input_size*3, kernel_size=1, bias=True),
            reshape(3, input_size, input_size)
        )
        # Upsampling layers
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [
                nn.Conv2d(out_channels[-1], out_channels[-1]*4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.PixelShuffle(upscale_factor=2),
            ]
        self.upsampling = nn.Sequential(*upsample_layers)

        # Final output block
        self.head_conv = nn.Sequential(
            nn.Conv2d(out_channels[-1], out_channels[-1], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels[-1], 3, kernel_size=3, stride=1, padding=1),
        )
        self.model = CGNet(num_channels=(8, 16, 32),
                            num_blocks=(3, 5),)
        self._initialize_weights()



    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, mode='fan_out',
                                     nonlinearity='relu')
                try:
                    m.bias.data.fill_(0)
                except:
                    continue
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, ok, ref):
        H, W = ok.shape[1:3]
        x = self.model(ok, ref)
        # x[0] = torch.cat([F.interpolate(x[0][i], size=ok.shape[2:], mode='bilinear')
        #         for i in range(len(x[0]))],dim=1)
        # x[1] = torch.cat([F.interpolate(x[1][i], size=ok.shape[2:], mode='bilinear')
        #         for i in range(len(x[1]))],dim=1)
        # ok_feature = self.ok_feature_conv(x[0][-1])
        # ref_feature = self.ref_feature_conv(x[1][-1])
        ok_ref_feature = self.feature_conv(torch.cat([x[0][-1],x[1][-1]],dim=1))

        # T = self.relation_conv(torch.cat([ok_feature,ref_feature],dim=1))
        T = self.relation_conv(ok_ref_feature)
        # out = torch.matmul(x[0][-1], T)

        #out = self.upsampling(out)
        # out = self.head_conv(out)
        # out = self.head_conv2(out)
        out = F.interpolate(T, size=ok.shape[2:], mode='bilinear')
        # out = [F.interpolate(out[i],size=ok.shape[2:], mode='bilinear')
        #        for i in range(len(out))]
        # out = torch.cat(out, dim=1)
        return out


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
    model = predImage(3, input_size//8)
    model.train()
    model.cuda()
    criteria = nn.L1Loss().cuda()  ##nn.MSELoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    dset = mydataset(r'./data/SynLCD_OKandNG/bg1/mixed/A',
                     r'./data/SynLCD_OKandNG/bg1/mixed/B', input_size)

    train_data = DataLoader(dset,
                            shuffle=True,
                            batch_size=4,
                            num_workers=4)
    for epoch in range(201):
        LOSS = []
        with tqdm(train_data) as t:
            for ok, ref, gt in t:
                # Zero the gradient
                # data = data.cuda()
                gt = gt.cuda()
                optimizer.zero_grad()
                out = model(ok.cuda(), ref.cuda())
                loss = criteria(out, gt)
                loss.backward()
                optimizer.step()
                LOSS.append(loss.item())
                t.set_description('Epoch:{} loss:{:.2f}'.format(epoch, np.mean(LOSS)))
        if epoch % 5 == 0:
            cv2.imwrite(os.path.join(path, 'epoch_%d.png' % epoch), toArray(out)[..., ::-1])

    weight_path = os.path.join(path, 'weights')
    os.makedirs(weight_path, exist_ok=True)
    weight_id = len(os.listdir(weight_path)) + 1
    torch.save(model.state_dict(), os.path.join(weight_path, 'weight_%d.pth' % weight_id))
    visualize(ok, ref, out, LOSS)

    print(loss.item())
