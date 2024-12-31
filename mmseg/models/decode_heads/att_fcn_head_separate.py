# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .decode_head import BaseDecodeHead
from ..utils.threedAttention import CoordAtt, ChannelAttention
import torch.nn.functional as F


class cam_head(nn.Module):
    def __init__(self, mid_ch=None, out_ch=None, cam=True, thatt=False, cross=False):
        super(cam_head, self).__init__()
        assert mid_ch is not None

        self.ch = mid_ch // 4
        self.cam = cam
        self.thatt = thatt
        self.cross = cross
        if self.cam:
            if not self.cross:
                self.hw_ca = CoordAtt(mid_ch, mid_ch, h=512, w=512, groups=16)
                self.d_ca = ChannelAttention(mid_ch, ratio=16)
                self.hw_ca_1 = CoordAtt(mid_ch // 4, mid_ch // 4, h=512, w=512, groups=16 // 4)
                self.d_ca1 = ChannelAttention(mid_ch // 4, ratio=16 // 4)

                self.ca = ChannelAttention(mid_ch, ratio=16)
                self.ca1 = ChannelAttention(mid_ch // 4, ratio=16 // 4)
            else:
                ''' abpt_v3 '''
                # self.hw_ca = CoordAtt(mid_ch+1, mid_ch+1, h=512, w=512, groups=16 // 4)
                # self.d_ca = ChannelAttention(mid_ch+1, ratio=16)
                ''' abpt_v4 '''
                #self.hw_ca = CoordAtt(mid_ch+1, mid_ch+1, h=512, w=512, groups=16 // 4)
                self.d_ca = ChannelAttention(mid_ch, ratio=16)
                ''' line '''
                # self.d_ca = ChannelAttention(mid_ch, ratio=16)
                # self.hw_ca = CoordAtt(mid_ch//4, mid_ch//4, h=512, w=512, groups=16 // 4)

        self.convs = nn.Identity()
        self.convs1 = nn.Identity()
        self.convs2 = nn.Identity()
        if self.cross:
            #self.conv_final = nn.Conv2d(mid_ch+1, 1, kernel_size=1)
            self.conv_tot = nn.Conv2d(mid_ch, 2, kernel_size=1)
            self.conv_change = nn.Conv2d(mid_ch, 2, kernel_size=1)
            #self.conv_tot
            self.Relu = nn.ReLU(inplace=False)
            #self.conv_nonchange = nn.Conv2d(mid_ch, 2, kernel_size=1)
        else:
            self.conv_final = nn.Conv2d(mid_ch, out_ch, kernel_size=1)

    def normMask1(self,x):
        return (x - x.min()) / (x.max() - x.min() + 1e-10)

    def sigmoid_mask(self,x):
        return F.sigmoid(x)

    def forward(self, out, distance):
        # out = self.ca(out)
        x0_1, x0_2, x0_3, x0_4 = out[:, :self.ch, :, :], out[:, self.ch:2 * self.ch, :, :], \
                                 out[:, 2 * self.ch:3 * self.ch, :, :], out[:, 3 * self.ch:, :, :]

        intra = torch.sum(torch.stack((x0_1, x0_2, x0_3, x0_4)), dim=0)

        if self.cam:
            if self.thatt:
                hw_ca_1 = self.hw_ca_1(intra+distance)
                out = self.d_ca(out + distance) * (out + hw_ca_1.repeat(1, 4, 1, 1))

            elif self.cross:
                distance = self.normMask1(distance.clone().detach())
                #distance = distance.clone().detach()
                out_tot = self.d_ca(out) * out
                out_change = distance * out_tot
                out_nonchange = (1-distance) * out_tot


            else:
                # original channel attention
                ca1 = self.ca1(intra)
                out = self.ca(out) * (out + ca1.repeat(1, 4, 1, 1))
        out = self.convs(out_tot)

        return self.conv_tot(out_tot), self.conv_change(out_change), self.Relu(out_nonchange), self.Relu(out_tot)
        #return self.conv_tot(out_tot), self.conv_change(out_nonchange), self.Relu(out_change), self.Relu(out_tot)


@HEADS.register_module()
class AttFCNHead_separate(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 imglevel_cls_output=False,
                 contra=True,
                 cam=True,
                 thatt=False,
                 cross=False,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        self.contra = contra
        self.cam = cam
        self.thatt=thatt
        super(AttFCNHead_separate, self).__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        self.imglevel_cls_output = imglevel_cls_output
        if imglevel_cls_output:
            self.cls_fconvs = nn.Sequential(
                nn.MaxPool2d((2, 2), stride=2),
                self._ConvModule(self.channels, self.channels // 4, ksize=5, pad=2),
                nn.MaxPool2d((2, 2), stride=2),
                self._ConvModule(self.channels // 4, self.channels // 16, ksize=5, pad=2),
            )
            self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
            self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.cls_img = nn.Linear(in_features=(self.channels // 16 + self.num_classes) * 2,
                                     out_features=1)
        self.att_head = cam_head(mid_ch=self.channels, out_ch=self.num_classes,
                                 cam=cam, thatt=thatt,cross=cross)

    def _ConvModule(self, in_ch, out_ch, ksize, pad):
        return ConvModule(
            in_ch, out_ch,
            kernel_size=ksize,
            padding=pad,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
    def normMask(self, x, strenth=0.5):
        b,c,h,w = x.size()
        max_value = x.reshape(b,-1).max(1)[0]
        max_value = max_value.reshape(b,1,1,1)
        x = x/(max_value*strenth)
        x = torch.clamp(x, 0, 1)
        return x

    def various_distance(self, feature_pairs):
        fea1, fea2 = feature_pairs
        n, c, h, w = fea1.shape
        fea1_rz = torch.transpose(fea1.view(n, c, h * w), 2, 1)
        fea2_rz = torch.transpose(fea2.view(n, c, h * w), 2, 1)
        return F.pairwise_distance(fea1_rz, fea2_rz, p=2)

    def get_distance(self,inputs):
        mid = [
            F.interpolate(
                input=x,
                size=inputs[0].shape[2:],
                mode='bilinear',
                align_corners=False) for x in inputs]
        mid = torch.cat(mid, dim=1)
        n, c, h, w = mid.shape
        n = n // 2
        mid = [mid[:n, ::], mid[n:, ::]]
        return self.various_distance(mid).view(n, 1, h, w)

    def forward(self, inputs):
        """Forward function."""
        #inputs, mid = inputs
        #x = self._transform_inputs(inputs)
        #mid = self._transform_inputs(mid)
        x, mid, distance = inputs

        output_tmp = self.convs(x)

        if self.concat_input:
            output_tmp = self.conv_cat(torch.cat([x, output_tmp], dim=1))

        seg_output = self.att_head(output_tmp, distance)

        if self.imglevel_cls_output:
            feature = self.cls_fconvs(output_tmp)  # 8*64*64
            avg_fea_bench = self.global_avg_pool(feature)  # 8x1
            max_fea_bench = self.global_max_pool(feature)  # 8x1
            avg_seg_bench = self.global_avg_pool(seg_output)  # 3x1
            max_seg_bench = self.global_max_pool(seg_output)  # 3x1
            cls_out_fea = torch.cat([avg_fea_bench, max_fea_bench,
                                     avg_seg_bench, max_seg_bench], dim=1).reshape(max_seg_bench.size(0), -1)
            cls_output = self.cls_img(cls_out_fea)

            return seg_output, cls_output, 'cls'

        elif self.contra:

            return seg_output, distance, 'contra'

        else:
            return seg_output, distance, 'celoss'
