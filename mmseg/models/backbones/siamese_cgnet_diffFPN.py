# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from ..builder import BACKBONES
import torch
from .cgnet import ContextGuidedBlock,InputInjection

class conv_block_nested(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x + identity)
        return output
class up(nn.Module):
    def __init__(self, in_ch, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

    def forward(self, x):

        x = self.up(x)
        return x
def cgblock(in_ch,out_ch):
    norm_cfg = {'type': 'SyncBN', 'eps': 0.001, 'requires_grad': True}
    act_cfg = {'type': 'PReLU', 'num_parameters': 32}
    return ContextGuidedBlock(in_ch, out_ch,
                       dilation=2,
                       reduction=8,
                       downsample=False,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg,skip_connect=False)
@BACKBONES.register_module()
class SiameseCGNetDiffFPN(nn.Module):
    # SNUNet-CD with ECAM
    def __init__(self, in_ch=3, bilinear=True):
        super(SiameseCGNetDiffFPN, self).__init__()
        torch.nn.Module.dump_patches = True

        n1 = 32     # the initial number of channels of feature map
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv0_0 = cgblock(in_ch, filters[0])
        self.conv1_0 = cgblock(filters[0], filters[1])
        self.Up1_0 = up(filters[1],bilinear)
        self.conv2_0 = cgblock(filters[1], filters[2])
        self.Up2_0 = up(filters[2],bilinear)
        self.conv3_0 = cgblock(filters[2], filters[3])
        self.Up3_0 = up(filters[3],bilinear)
        self.conv4_0 = cgblock(filters[3], filters[4])
        self.Up4_0 = up(filters[4],bilinear)

        self.conv0_1 = cgblock(filters[0] * 2 + filters[1], filters[0])
        self.conv1_1 = cgblock(filters[1] * 2 + filters[2], filters[1])
        self.Up1_1 = up(filters[1],bilinear)
        self.conv2_1 = cgblock(filters[2] * 2 + filters[3], filters[2])
        self.Up2_1 = up(filters[2],bilinear)
        self.conv3_1 = cgblock(filters[3] * 2 + filters[4], filters[3])
        self.Up3_1 = up(filters[3],bilinear)

        self.conv0_2 = cgblock(filters[0] * 3 + filters[1], filters[0])
        self.conv1_2 = cgblock(filters[1] * 3 + filters[2], filters[1])
        self.Up1_2 = up(filters[1],bilinear)
        self.conv2_2 = cgblock(filters[2] * 3 + filters[3], filters[2])
        self.Up2_2 = up(filters[2],bilinear)

        self.conv0_3 = cgblock(filters[0] * 4 + filters[1], filters[0])
        self.conv1_3 = cgblock(filters[1] * 4 + filters[2], filters[1])
        self.Up1_3 = up(filters[1],bilinear)

        self.conv0_4 = cgblock(filters[0] * 5 + filters[1], filters[0])

        # self.ca = ChannelAttention(filters[0] * 4, ratio=16)
        # self.ca1 = ChannelAttention(filters[0], ratio=16 // 4)
        #
        # self.conv_final = nn.Conv2d(filters[0] * 4, out_ch, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        '''xA'''
        bsize = x.shape[0]//2
        #x = torch.cat([xA,xB],dim=0)
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        x0_0A, x0_0B = x0_0[:bsize,::], x0_0[bsize:,::]
        x1_0A, x1_0B = x1_0[:bsize,::], x1_0[bsize:,::]
        x2_0A, x2_0B = x2_0[:bsize,::], x2_0[bsize:,::]
        x3_0A, x3_0B = x3_0[:bsize,::], x3_0[bsize:,::]
        x4_0A, x4_0B = x4_0[:bsize,::], x4_0[bsize:,::]

        x0_1 = self.conv0_1(torch.cat([x0_0A, x0_0B, self.Up1_0(x1_0B)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0A, x1_0B, self.Up2_0(x2_0B)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0A, x0_0B, x0_1, self.Up1_1(x1_1)], 1))


        x2_1 = self.conv2_1(torch.cat([x2_0A, x2_0B, self.Up3_0(x3_0B)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0A, x1_0B, x1_1, self.Up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0A, x0_0B, x0_1, x0_2, self.Up1_2(x1_2)], 1))

        x3_1 = self.conv3_1(torch.cat([x3_0A, x3_0B, self.Up4_0(x4_0B)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0A, x2_0B, x2_1, self.Up3_1(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0A, x1_0B, x1_1, x1_2, self.Up2_2(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0A, x0_0B, x0_1, x0_2, x0_3, self.Up1_3(x1_3)], 1))

        return (x0_1, x0_2, x0_3, x0_4),(x3_0A,x3_0B)

    def forward_bak(self, xA, xB):
        '''xA'''
        x0_0A = self.conv0_0(xA)
        x1_0A = self.conv1_0(self.pool(x0_0A))
        x2_0A = self.conv2_0(self.pool(x1_0A))
        x3_0A = self.conv3_0(self.pool(x2_0A))
        #x4_0A = self.conv4_0(self.pool(x3_0A))
        '''xB'''
        x0_0B = self.conv0_0(xB)
        x1_0B = self.conv1_0(self.pool(x0_0B))
        x2_0B = self.conv2_0(self.pool(x1_0B))
        x3_0B = self.conv3_0(self.pool(x2_0B))
        x4_0B = self.conv4_0(self.pool(x3_0B))

        x0_1 = self.conv0_1(torch.cat([x0_0A, x0_0B, self.Up1_0(x1_0B)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0A, x1_0B, self.Up2_0(x2_0B)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0A, x0_0B, x0_1, self.Up1_1(x1_1)], 1))


        x2_1 = self.conv2_1(torch.cat([x2_0A, x2_0B, self.Up3_0(x3_0B)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0A, x1_0B, x1_1, self.Up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0A, x0_0B, x0_1, x0_2, self.Up1_2(x1_2)], 1))

        x3_1 = self.conv3_1(torch.cat([x3_0A, x3_0B, self.Up4_0(x4_0B)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0A, x2_0B, x2_1, self.Up3_1(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0A, x1_0B, x1_1, x1_2, self.Up2_2(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0A, x0_0B, x0_1, x0_2, x0_3, self.Up1_3(x1_3)], 1))

        return (x0_1, x0_2, x0_3, x0_4),(x3_0A,x3_0B)
