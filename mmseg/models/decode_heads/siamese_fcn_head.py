# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class SiameseFCNHead(BaseDecodeHead):
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
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super(SiameseFCNHead, self).__init__(**kwargs)
        if num_convs == 0:
            #assert self.in_channels == self.channels
            pass

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels//2,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels//2,
                    self.channels//2,
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
            self.cls_img = nn.Linear(in_features=(self.channels//16+self.num_classes)*2,
                                     out_features=1)

    def _ConvModule(self, in_ch, out_ch, ksize, pad):
        return ConvModule(
            in_ch, out_ch,
            kernel_size=ksize,
            padding=pad,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def single_layer_similar_heatmap_visual(self, output_t0,output_t1):

        n, c, h, w = output_t0.data.shape
        out_t0_rz = torch.transpose(output_t0.view(c, h * w), 1, 0)
        out_t1_rz = torch.transpose(output_t1.view(c, h * w), 1, 0)
        distance = torch.Functional.pairwise_distance(out_t0_rz, out_t1_rz, p=2)
        return distance

    def forward(self, inputs):
        """Forward function."""
        x = [self._transform_inputs(_) for _ in inputs]
        output_tmp = [self.convs(_) for _ in x]
        x1, x2 = output_tmp
        diff = torch.abs(x1 - x2)
        dist_sq = torch.pow(diff + 1e-6, 2).sum(dim=1)
        dist = torch.sqrt(dist_sq)

        if self.concat_input:
            output_tmp = self.conv_cat(torch.cat([x, output_tmp], dim=1))
        y = torch.cat(output_tmp,dim=1)*dist.unsqueeze(1)
        seg_output = self.cls_seg(y)
        # TODO
        # to realize classification through feature comparison
        if self.imglevel_cls_output:
            feature = self.cls_fconvs(output_tmp)  # 8*64*64
            avg_fea_bench = self.global_avg_pool(feature)  # 8x1
            max_fea_bench = self.global_max_pool(feature)  # 8x1
            avg_seg_bench = self.global_avg_pool(seg_output)  # 3x1
            max_seg_bench = self.global_max_pool(seg_output)  # 3x1
            cls_out_fea = torch.cat([avg_fea_bench, max_fea_bench,
                                    avg_seg_bench, max_seg_bench], dim=1).reshape(max_seg_bench.size(0),-1)
            cls_output = self.cls_img(cls_out_fea)

            return seg_output, cls_output
        else:
            return seg_output, output_tmp
