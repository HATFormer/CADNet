# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from ..backbones.cgnet import GlobalContextExtractor
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from ..utils.CoordAttention import CoordAtt

def normMask(mask, strenth = 0.5):
    """
    :return: to attention more region

    """
    batch_size, c_m, c_h, c_w = mask.size()
    max_value = mask.reshape(batch_size, -1).max(1)[0]
    max_value = max_value.reshape(batch_size, 1, 1, 1)
    mask = mask/(max_value*strenth)
    mask = torch.clamp(mask, 0, 1) # 将输入input张量每个元素的夹紧到区间 [min,max]

    return mask

def build_norm_layer(ch):
    layer = nn.BatchNorm2d(ch, eps=0.01)
    for param in layer.parameters():
        param.requires_grad = True
    return layer

class ContextGuidedBlock(nn.Module):
    """Context Guided Block for CGNet.

    This class consists of four components: local feature extractor,
    surrounding feature extractor, joint feature extractor and global
    context extractor.

    Args:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output feature channels.
        dilation (int): Dilation rate for surrounding context extractor.
            Default: 2.
        reduction (int): Reduction for global context extractor. Default: 16.
        skip_connect (bool): Add input to output or not. Default: True.
        downsample (bool): Downsample the input to 1/2 or not. Default: False.
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='PReLU').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation=2,
                 reduction=16,
                 skip_connect=True,
                 downsample=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='PReLU'),
                 with_cp=False):
        super(ContextGuidedBlock, self).__init__()
        self.with_cp = with_cp
        self.downsample = downsample

        # channels = out_channels if downsample else out_channels // 2
        channels = out_channels // 2
        if 'type' in act_cfg and act_cfg['type'] == 'PReLU':
            act_cfg['num_parameters'] = channels
        kernel_size = 3 if downsample else 1
        stride = 2 if downsample else 1
        padding = (kernel_size - 1) // 2
        # self.channel_shuffle = ChannelShuffle(2 if in_channels==in_channels//2*2 else in_channels)
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=kernel_size, stride=stride, padding=padding),
            build_norm_layer(channels),
            nn.PReLU(num_parameters=channels)
        )

        self.f_loc = nn.Conv2d(channels, channels, kernel_size=3,
                               padding=1, groups=channels, bias=False)

        self.f_sur = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation,
                               dilation=dilation, groups=channels, bias=False)

        self.bn = build_norm_layer(2 * channels)
        self.activate = nn.PReLU(2 * channels)

        # original bottleneck in CGNet: A light weight context guided network for segmantic segmentation
        # is removed for saving computation amount
        # if downsample:
        #     self.bottleneck = build_conv_layer(
        #         conv_cfg,
        #         2 * channels,
        #         out_channels,
        #         kernel_size=1,
        #         bias=False)

        self.skip_connect = skip_connect and not downsample
        self.f_glo = GlobalContextExtractor(out_channels, reduction, with_cp)
        # self.f_glo = CoordAtt(out_channels,out_channels,groups=reduction)

    def forward(self, x):

        def _inner_forward(x):
            # x = self.channel_shuffle(x)
            out = self.conv1x1(x)
            loc = self.f_loc(out)
            sur = self.f_sur(out)

            joi_feat = torch.cat([loc, sur], 1)  # the joint feature
            joi_feat = self.bn(joi_feat)
            joi_feat = self.activate(joi_feat)
            if self.downsample:
                pass
                # joi_feat = self.bottleneck(joi_feat)  # channel = out_channels
            # f_glo is employed to refine the joint feature
            out = self.f_glo(joi_feat)

            if self.skip_connect:
                return x + out
            else:
                return out

        return _inner_forward(x)

def cgblock(in_ch, out_ch, dilation=2, reduction=8, skip_connect=False):
    return nn.Sequential(
        ContextGuidedBlock(in_ch, out_ch,
                           dilation=dilation,
                           reduction=reduction,
                           downsample=False,
                           skip_connect=skip_connect))

class cam_head(nn.Module):
    def __init__(self, mid_ch=None, filters=None, out_ch=None, n=4):
        super(cam_head, self).__init__()
        assert filters is not None or mid_ch is not None
        ch = filters[0] if filters is not None else mid_ch
        self.ca = CoordAtt(ch * n, ch * n, h=256, w=256, groups=16)
        self.conv_final = nn.Conv2d(ch * n, out_ch, kernel_size=1)

    def forward(self, x0_1, x0_2, x0_3=None):
        if x0_3 is not None:
            out = torch.cat([x0_1, x0_2, x0_3], 1)
        else:
            out = torch.cat([x0_1, x0_2], 1)
        out = self.ca(out)
        return self.conv_final(out)

class up(nn.Module):
    def __init__(self, in_ch=3, bilinear=False):
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

class diffFPN(nn.Module):
    def __init__(self, cur_channels=None, mid_ch=None,
                 dilations=None, reductions=None,
                 bilinear=True):
        super(diffFPN, self).__init__()
        # lateral convs for unifing channels
        self.lateral_convs = nn.ModuleList()
        for i in range(3):
            self.lateral_convs.append(
                cgblock(cur_channels[i] * 2, mid_ch * 2 ** i, dilations[i], reductions[i])
            )
        # top_down_convs
        self.top_down_convs = nn.ModuleList()
        for i in range(2, 0, -1):
            self.top_down_convs.append(
                cgblock(mid_ch * 2 ** i, mid_ch * 2 ** (i - 1), dilation=dilations[i], reduction=reductions[i])
            )

        # diff convs
        self.diff_convs = nn.ModuleList()

        for i in range(2):
            self.diff_convs.append(
                cgblock(mid_ch * (3 * 2 ** i), mid_ch * (2 * 2 ** i), dilations[i], reductions[i])
            )
        self.diff_convs.append(
            cgblock(mid_ch * 6, mid_ch*2,
                    dilation=dilations[0], reduction=reductions[0])
        )
        self.up2x = up(32, bilinear)

    def forward(self, output):

        tmp = [self.lateral_convs[i](torch.cat([output[0][i], output[1][i]], dim=1))
               for i in range(3)]

        # top_down_path
        for i in range(2, 0, -1):
            tmp[i - 1] += self.up2x(self.top_down_convs[2 - i](tmp[i]))

        # x0_1
        tmp = [self.diff_convs[i](torch.cat([tmp[i], self.up2x(tmp[i + 1])], dim=1)) for i in [0, 1]]
        x0_1 = tmp[0]
        # x0_2
        x0_2 = self.diff_convs[2](torch.cat([tmp[0], self.up2x(tmp[1])], dim=1))

        return x0_1, x0_2

@HEADS.register_module()
class diffFPNHead(BaseDecodeHead):
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
        super(diffFPNHead, self).__init__(**kwargs)
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
        mid_ch=64
        out_ch=3
        bilinear=True
        dilations = (1, 2, 4, 8)
        reductions = (4, 8, 16, 32)

        self.head = cam_head(mid_ch=mid_ch, out_ch=out_ch)

        self.FPN = diffFPN(cur_channels=[35, 131, 256], mid_ch=mid_ch,
                           dilations=dilations, reductions=reductions, bilinear=bilinear)

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
        output_tmp = inputs
        x0_1, x0_2 = self.FPN(output_tmp)
        # diff = torch.abs(x1 - x2)
        # dist_sq = torch.pow(diff + 1e-6, 2).sum(dim=1)
        # dist = torch.sqrt(dist_sq)
        # y = torch.cat(output_tmp, dim=1) * dist.unsqueeze(1)

        seg_output = self.head(x0_1, x0_2)

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
            return seg_output
