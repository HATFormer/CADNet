# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class distAtt(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, mid_ch=32):
        super(distAtt, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(1)
        self.relu = h_swish()

    def forward(self,x):

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)

        y = self.conv3(y)
        y = self.bn3(y)

        return y

class l2normalization(nn.Module):
    def __init__(self, scale):
        super(l2normalization, self).__init__()
        self.scale = scale
        self.norm = nn.Softmax2d()

    def forward(self, x, dim=1):
        '''out = scale * x / sqrt(\sum x_i^2)'''
        # f = x.data.cpu().numpy()
        # scal = self.scale * x * x.pow(2).sum(dim).clamp(min=1e-12).rsqrt().expand_as(x)
        # sca = scal.data.cpu().numpy()
        y = x.pow(2).sum(dim).unsqueeze(1)
        return self.scale * x * y.clamp(min=1e-12).rsqrt().expand_as(x)



@HEADS.register_module()
class SiameseSegformerHead(BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        # assert num_inputs == len(self.in_index)
        self.norm = l2normalization(scale=1)
        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        self.cam = distAtt()
    def _transform_inputs(self, x):
        return self.norm(x)


    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        outs = []
        for bidx, branch in enumerate(inputs):
            tmp = []
            for fidx in range(len(inputs[bidx])):
                x = inputs[bidx][fidx]
                conv = self.convs[fidx]
                tmp.append(
                    resize(
                        input=conv(x),
                        size=inputs[bidx][0].shape[2:],
                        mode=self.interpolate_mode,
                        align_corners=self.align_corners))
            outs.append(tmp)
        mid = [self.fusion_conv(torch.cat(x, dim=1)) for x in outs]
        mid = [self._transform_inputs(x) for x in mid]

        n, c, h, w = mid[0].shape
        distance = self.various_distance(mid).view(n, 1, h, w)
        distance = self.cam(distance)
        out = self.cls_seg(torch.cat(mid, dim=1)*distance)

        return out, mid, 'contra'

# For debug
def save_img(name,img):
    import numpy as np
    import cv2
    img = img.cpu().detach().numpy()
    img = (img - img.min())/(img.max()-img.min()+1e-10)
    img = np.uint8(img*255)
    cv2.imwrite(name,cv2.applyColorMap(cv2.resize(img,(512,512)),cv2.COLORMAP_JET))
# save_img('siamesenet_dist2.png',distance[0][0])
# save_img('siamesenet_out2.png',seg_output.softmax(dim=1).argmax(dim=1)[0][0])