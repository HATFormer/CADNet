# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from mmseg.ops import resize
from ..builder import NECKS
import torch


@NECKS.register_module()
class diffFPN(BaseModule):
    """Feature Pyramid Network.

    This neck is the implementation of `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        extra_convs_on_inputs (bool, deprecated): Whether to apply extra convs
            on the original feature from the backbone. If True,
            it is equivalent to `add_extra_convs='on_input'`. If False, it is
            equivalent to set `add_extra_convs='on_output'`. Default to True.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 # upsample_cfg=dict(mode='nearest'),
                 upsample_cfg=dict(mode='bilinear',
                                   align_corners=False),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(diffFPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # For compatibility with previous release
                # TODO: deprecate `extra_convs_on_inputs`
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        dpv3_channel_list = [[256 * 4, 256 * 5, 256 * 6, 256 * 7],
                             [512 * 4, 512 * 5, 512 * 6],
                             [1024 * 4, 1024 * 5],
                             [2048 * 3],
                             [2048]]
        channel_list = []
        for row_incs in range(len(in_channels), 0, -1):
            tmp = []
            for i in range(row_incs):
                tmp.append(in_channels[len(in_channels) - row_incs] * (2 + i)
                           + out_channels[len(in_channels) - row_incs + 1])
            channel_list.append(tmp)
        channel_list.append([in_channels[-1]])

        for layer_i, layer_i_channels in enumerate(channel_list):
            self.tmp_convs = nn.ModuleList()
            for conv_j_channel in layer_i_channels:
                fpn_conv = ConvModule(
                    conv_j_channel,
                    out_channels[layer_i],
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.tmp_convs.append(fpn_conv)
            self.fpn_convs.append(self.tmp_convs)

    @auto_fp16()
    def dpv3_forward(self, inputs):

        assert len(inputs[0]) == len(self.in_channels)
        inputs[1].append(self.fpn_convs[4][0](F.max_pool2d(inputs[1][-1], 1, stride=2)))
        x0_1 = self.fpn_convs[0][0](torch.cat([inputs[0][0], inputs[1][0],
                                               resize(inputs[1][1], **self.upsample_cfg)], 1))
        x1_1 = self.fpn_convs[1][0](torch.cat([inputs[0][1], inputs[1][1], inputs[1][2]], 1))
        x0_2 = self.fpn_convs[0][1](torch.cat([inputs[0][0], inputs[1][0], x0_1,
                                               resize(x1_1, **self.upsample_cfg)], 1))

        x2_1 = self.fpn_convs[2][0](torch.cat([inputs[0][2], inputs[1][2], inputs[1][3]], 1))
        x1_2 = self.fpn_convs[1][1](torch.cat([inputs[0][1], inputs[1][1], x1_1, x2_1], 1))
        x0_3 = self.fpn_convs[0][2](torch.cat([inputs[0][0], inputs[1][0], x0_1, x0_2,
                                               resize(x1_2, **self.upsample_cfg)], 1))

        x3_1 = self.fpn_convs[3][0](torch.cat([inputs[0][3], inputs[1][3],
                                               resize(inputs[1][4], **self.upsample_cfg)], 1))
        x2_2 = self.fpn_convs[2][1](torch.cat([inputs[0][2], inputs[1][2], x2_1, x3_1], 1))
        x1_3 = self.fpn_convs[1][2](torch.cat([inputs[0][1], inputs[1][1], x1_1, x1_2, x2_2], 1))
        x0_4 = self.fpn_convs[0][3](torch.cat([inputs[0][0], inputs[1][0], x0_1, x0_2, x0_3,
                                               resize(x1_3, **self.upsample_cfg)], 1))

        return tuple([x0_4, x1_3, x2_2, x3_1])

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs[0]) == len(self.in_channels)
        inputs[1].append(self.fpn_convs[-1][0](F.max_pool2d(inputs[1][-1], 1, stride=2)))
        outs = [inputs[1]]
        for row_num in range(len(self.fpn_convs) - 1, 0, -1):
            col_outs = []
            col_id = len(self.fpn_convs) - 1 - row_num
            for row_id in range(row_num):
                col_outs.append(self.fpn_convs[row_id][col_id](
                    torch.cat([inputs[0][row_id]] +
                              [x[row_id] for x in outs] +
                              [resize(outs[-1][row_id+1], size=outs[-1][row_id].shape[2:],
                              **self.upsample_cfg)],1)
                ))
            outs.append(col_outs)
        return tuple([outs[i][-1] for i in range(len(self.fpn_convs)-1,0,-1)])
