# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule
from mmcv.utils.parrots_wrapper import _BatchNorm

from ..builder import BACKBONES
from .cgnet import ContextGuidedBlock,InputInjection


@BACKBONES.register_module()
class Siamese_CGNet(BaseModule):
    """CGNet backbone.

    This backbone is the implementation of `A Light-weight Context Guided
    Network for Semantic Segmentation <https://arxiv.org/abs/1811.08201>`_.

    Args:
        in_channels (int): Number of input image channels. Normally 3.
        num_channels (tuple[int]): Numbers of feature channels at each stages.
            Default: (32, 64, 128).
        num_blocks (tuple[int]): Numbers of CG blocks at stage 1 and stage 2.
            Default: (3, 21).
        dilations (tuple[int]): Dilation rate for surrounding context
            extractors at stage 1 and stage 2. Default: (2, 4).
        reductions (tuple[int]): Reductions for global context extractors at
            stage 1 and stage 2. Default: (8, 16).
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='PReLU').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels=3,
                 num_channels=(32, 64, 128),
                 num_blocks=(3, 21),
                 dilations=(2, 4),
                 reductions=(8, 16),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='PReLU'),
                 norm_eval=False,
                 with_cp=False,
                 pretrained=None,
                 init_cfg=None):

        super(Siamese_CGNet, self).__init__(init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer=['Conv2d', 'Linear']),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm']),
                    dict(type='Constant', val=0, layer='PReLU')
                ]
        else:
            raise TypeError('pretrained must be a str or None')

        self.in_channels = in_channels
        self.num_channels = num_channels
        assert isinstance(self.num_channels, tuple) and len(
            self.num_channels) == 3
        self.num_blocks = num_blocks
        assert isinstance(self.num_blocks, tuple) and len(self.num_blocks) == 2
        self.dilations = dilations
        assert isinstance(self.dilations, tuple) and len(self.dilations) == 2
        self.reductions = reductions
        assert isinstance(self.reductions, tuple) and len(self.reductions) == 2
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        if 'type' in self.act_cfg and self.act_cfg['type'] == 'PReLU':
            self.act_cfg['num_parameters'] = num_channels[0]
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        cur_channels = in_channels
        self.stem = nn.ModuleList()
        for i in range(3):
            self.stem.append(
                ConvModule(
                    cur_channels,
                    num_channels[0],
                    3,
                    2 if i == 0 else 1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            cur_channels = num_channels[0]

        self.inject_2x = InputInjection(1)  # down-sample for Input, factor=2
        self.inject_4x = InputInjection(2)  # down-sample for Input, factor=4

        cur_channels += in_channels
        self.norm_prelu_0 = nn.Sequential(
            build_norm_layer(norm_cfg, cur_channels)[1],
            nn.PReLU(cur_channels))

        # stage 1
        self.level1 = nn.ModuleList()
        for i in range(num_blocks[0]):
            self.level1.append(
                ContextGuidedBlock(
                    cur_channels if i == 0 else num_channels[1],
                    num_channels[1],
                    dilations[0],
                    reductions[0],
                    downsample=(i == 0),
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    with_cp=with_cp))  # CG block

        cur_channels = 2 * num_channels[1] + in_channels
        self.norm_prelu_1 = nn.Sequential(
            build_norm_layer(norm_cfg, cur_channels)[1],
            nn.PReLU(cur_channels))

        # stage 2
        self.level2 = nn.ModuleList()
        for i in range(num_blocks[1]):
            self.level2.append(
                ContextGuidedBlock(
                    cur_channels if i == 0 else num_channels[2],
                    num_channels[2],
                    dilations[1],
                    reductions[1],
                    downsample=(i == 0),
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    with_cp=with_cp))  # CG block

        cur_channels = 2 * num_channels[2]
        self.norm_prelu_2 = nn.Sequential(
            build_norm_layer(norm_cfg, cur_channels)[1],
            nn.PReLU(cur_channels))
    def subforward(self,x):
        output = []

        # stage 0
        inp_2x = self.inject_2x(x)
        inp_4x = self.inject_4x(x)
        for layer in self.stem:
            x = layer(x)
        x = self.norm_prelu_0(torch.cat([x, inp_2x], 1))
        output.append(x)

        # stage 1
        for i, layer in enumerate(self.level1):
            x = layer(x)
            if i == 0:
                down1 = x
        x = self.norm_prelu_1(torch.cat([x, down1, inp_4x], 1))
        output.append(x)

        # stage 2
        for i, layer in enumerate(self.level2):
            x = layer(x)
            if i == 0:
                down2 = x
        x = self.norm_prelu_2(torch.cat([down2, x], 1))
        output.append(x)
        return output

    def forward(self, x, ref_x):
        bc = x.shape[0]
        x = torch.cat([x, ref_x], 0)
        tmp = self.subforward(x)
        outs = [[], []]
        for x in tmp:
            outs[0].append(x[:bc, :, :, :])
            outs[1].append(x[bc:, :, :, :])
        return outs



    def train(self, mode=True):
        """Convert the model into training mode will keeping the normalization
        layer freezed."""
        super(Siamese_CGNet, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
