# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize
from ..utils.vis_tsne import visualize_tsne
from ..utils.threedAttention import CoordAtt, ChannelAttention


class cad_head(nn.Module):
    def __init__(self, mid_ch=256,  cross=False, HW=512):
        super(cad_head, self).__init__()

        assert mid_ch is not None
        self.ch = mid_ch // 4
        self.cross = cross
        self.ca = ChannelAttention(mid_ch, ratio=16)
        if not self.cross:
            self.hw_a = CoordAtt(mid_ch//4, mid_ch // 4, h=HW, w=HW, groups=16 // 4)


    def normMask(self, x):
        return (x - x.min()) / (x.max() - x.min() + 1e-10)

    def forward(self, out, distance):

        distance = distance.clone().detach().sigmoid()
        x0_1, x0_2, x0_3, x0_4 = out
        out = torch.cat(out, dim=1)

        if not self.cross:
            intra = torch.sum(torch.stack((x0_1, x0_2, x0_3, x0_4)), dim=0)
            hw_a = self.hw_a(intra + distance)
            out = self.ca(out + distance) * (out + hw_a.repeat(1, 4, 1, 1))

        else:
            distance = self.normMask(distance)
            out = distance * self.ca(out) * out

        return out


@HEADS.register_module()
class SegformerHeadContra(BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', change_att=False,
                 cross=False, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

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
        if change_att:
            self.cad_head = cad_head(mid_ch=self.channels * num_inputs, cross=cross, HW=128)
        self.change_att = change_att
        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs, distance = inputs
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        if self.change_att:
            outs = self.cad_head(outs, distance)

            out = self.fusion_conv(outs)
        else:
            out = self.fusion_conv(torch.cat(outs,dim=1))

        #vis_feature = out.clone().detach()

        out = self.cls_seg(out)

        if out.shape[1] == 1:
            return out, distance, 'all_contra'
        else:
            return out, distance, 'contra'
