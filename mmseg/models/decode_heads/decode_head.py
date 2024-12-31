# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from torch.nn import functional as F
import torch
import torch.nn as nn
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from mmseg.core import build_pixel_sampler
from mmseg.ops import resize
from ..builder import build_loss
from ..losses import accuracy

class BaseDecodeHead(BaseModule, metaclass=ABCMeta):
    """Base class for BaseDecodeHead.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict): Config of decode loss.
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 siamese=False,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ignore_index=255,
                 sampler=None,
                 align_corners=False,
                 init_cfg=dict(
                     type='Normal', std=0.01, override=dict(name='conv_seg'))):
        super(BaseDecodeHead, self).__init__(init_cfg)
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.loss_decode = build_loss(loss_decode)
        self.loss_cls = nn.BCEWithLogitsLoss(reduction='mean')

        self.ignore_index = ignore_index
        self.align_corners = align_corners
        self.margin1 = 0.3
        self.margin2 = 2.2
        self.eps = 1e-10
        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None
        # TODO
        if siamese:
            self.conv_seg = nn.Conv2d(2*channels, num_classes, kernel_size=1)
        else:
            self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            #import pdb;pdb.set_trace()
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)

        if type(seg_logits) != torch.Tensor:
            if seg_logits[-1] == 'cls':
                losses = self.losses_withCLS(seg_logits[0], seg_logits[1], gt_semantic_seg)
            elif seg_logits[-1] == 'contra':
                losses = self.contra_losses(seg_logits[0], seg_logits[1], gt_semantic_seg)
            elif seg_logits[-1] == 'all_contra':
                losses = self.all_contra_losses(seg_logits[0], seg_logits[1], gt_semantic_seg)
            elif seg_logits[-1] == 'all_contra_twoch':
                losses = self.all_contra_twoch_losses(seg_logits[0], seg_logits[1], gt_semantic_seg)
            elif seg_logits[-1] == 'celoss':
                losses = self.losses(seg_logits[0], gt_semantic_seg)
            elif seg_logits[-1] == 'contra_and_ce':
                losses = self.contra_and_ce_losses(seg_logits[0], seg_logits[1][0], seg_logits[1][1], gt_semantic_seg)
            else:
                losses = self.losses(seg_logits, gt_semantic_seg)
        else:
            losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def various_distance(self, feature_pairs):
        fea1, fea2 = feature_pairs
        n, c, h, w = fea1.shape
        fea1_rz = torch.transpose(fea1.view(n, c, h * w), 2, 1)
        fea2_rz = torch.transpose(fea2.view(n, c, h * w), 2, 1)
        return F.pairwise_distance(fea1_rz,fea2_rz,p=2)

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs)

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    @force_fp32(apply_to=('seg_logit',))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        # if self.ignore_index not in seg_label.unique():
        #     seg_label -= 1
        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        # print(loss)
        return loss

    @force_fp32(apply_to=('seg_logit', 'dist'))
    def sub_contra_multicls(self, seg_label, dist, BCL=True, weight=50, m1=None, m2=None):
        dist = resize(
            input=dist,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if not m1 and not m2:
            m1 = self.margin1
            m2 = self.margin2
        contra_label = seg_label.clone()
        contra_label[contra_label == 255] = 0

        mdist_pos = torch.clamp(dist - m1, min=0.0)
        mdist_neg = torch.clamp(m2 - dist, min=0.0)

        multi_cls = True

        contra_label[contra_label != 0] = 1
        loss_neg = contra_label * mdist_neg

        loss_pos = (contra_label == 0) * (mdist_pos)

        return (loss_pos.sum() + loss_neg.sum()) / (seg_label.shape[2] * weight), multi_cls

    @force_fp32(apply_to=('seg_logit', 'dist'))
    def sub_contra(self, seg_label, dist, weight=50, m1=None, m2=None, class_weights=None):
        dist = resize(
            input=dist,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if not m1 and not m2:
            m1 = self.margin1
            m2 = self.margin2
        contra_label = seg_label.clone().detach()
        #contra_label[contra_label == 255] = 0

        mdist_pos = torch.clamp(dist - m1, min=0.0)
        mdist_neg = torch.clamp(m2 - dist, min=0.0)

        multi_cls = False

        if class_weights is not None:
            multi_cls = True

            # cls_1 = torch.sum(contra_label == 1)
            # cls_2 = torch.sum(contra_label == 2)
            # w1 = (cls_1 + cls_2) / cls_1
            # w2 = (cls_1 + cls_2) / cls_2
            loss_neg = 0
            for idx in range(1,len(class_weights)+1):
                loss_neg_idx = (contra_label == idx) * mdist_neg
                loss_neg += class_weights[idx-1] * loss_neg_idx.sum()
        else:
            labeled_points = (contra_label != 0) == (contra_label != 255)
            loss_neg = labeled_points * mdist_neg

        loss_pos = (contra_label == 0) * (mdist_pos)

        return (loss_pos.sum() + loss_neg.sum()) / (seg_label.shape[2]*weight), multi_cls

    def sub_segloss(self, seg_logit, seg_label, seg_weight):

        seg_label = seg_label.squeeze(1)

        return self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)

    @force_fp32(apply_to=('seg_logit', 'dist'))
    def contra_losses(self, seg_logit, dist, seg_label, seg_weight=None):
        """Compute segmentation loss."""

        loss = dict()
        # multi_cls = True
        ## contrastive loss
        loss_contra_dist, multi_cls = self.sub_contra(seg_label, dist, weight=512, class_weights=[1/0.924, 1/0.076])
        if not multi_cls:
            loss_contra_seglogit, _ = self.sub_contra(seg_label, seg_logit, weight=1)

        if multi_cls:
            ## cross entroy loss
            if not multi_cls:
                seg_label[seg_label!=0] = 1

            loss_weights = [1.0, 0.5]
            if type(seg_logit)==torch.Tensor:
                seg_logit = [seg_logit]

            for i in range(len(seg_logit)):
                seg_logit[i] = resize(
                    input=seg_logit[i],
                    size=seg_label.shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners)

            if len(seg_logit) <= 2:
                for idx, _ in enumerate(seg_logit):
                    loss['loss_seg_{%d}'%idx] = loss_weights[idx] * self.sub_segloss(_, seg_label, seg_weight)

                    loss['acc_seg_{%d}'%idx] = accuracy(_, seg_label.squeeze(1))
            else:
                for idx, _ in enumerate(seg_logit[:2]):
                    loss['loss_seg_{%d}' % idx] = loss_weights[idx] * self.sub_segloss(_, seg_label, seg_weight)
                    loss['acc_seg_{%d}' % idx] = accuracy(_, seg_label.squeeze(1))

                tot_sum = seg_logit[3].clone().detach().sum(dim=1)
                nonchange_sum = seg_logit[2].sum(dim=1)
                loss['loss_bas'] = nonchange_sum/(tot_sum + 1e-8)
                loss['loss_bas'][nonchange_sum >= tot_sum] = 0  ## or 1
                loss['loss_bas'] = loss['loss_bas'].mean()*512

            loss['loss_contra'] = 0.001*loss_contra_dist

        else:
            loss['loss_contra'] = loss_contra_dist + loss_contra_seglogit
        # if multi_cls:
        #     loss['class1'] = w1*loss_neg_1.sum()
        #     loss['class2'] = w2*loss_neg_2.sum()

        return loss

    @force_fp32(apply_to=('seg_logit', 'dist'))
    def contra_and_ce_losses(self, seg_logit, dist_logit, dist, seg_label, seg_weight=None):
        """Compute segmentation loss."""

        loss = dict()
        # multi_cls = True
        ## contrastive loss
        loss_contra_dist, _ = self.sub_contra(seg_label, dist, weight=256)
        loss_contra_dist_logit, _ = self.sub_contra(seg_label, seg_logit, weight=8)

        loss_weights = [1.0, 0.5]
        if type(seg_logit)==torch.Tensor:
            seg_logit = [seg_logit]

        for i in range(len(seg_logit)):
            seg_logit[i] = resize(
                input=seg_logit[i],
                size=seg_label.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        if len(seg_logit) <= 2:
            for idx, _ in enumerate(seg_logit):
                loss['loss_seg_{%d}'%idx] = loss_weights[idx] * self.sub_segloss(_, seg_label, seg_weight)

                loss['acc_seg_{%d}'%idx] = accuracy(_, seg_label.squeeze(1))


        loss['loss_contra_dist'] = loss_contra_dist
        loss['loss_contra_dist_logit'] = loss_contra_dist_logit


        return loss

    @force_fp32(apply_to=('seg_logit', 'dist'))
    def all_contra_losses(self, seg_logit, dist, seg_label, seg_weight=None):
        """Compute segmentation loss."""

        loss = dict()
        ## contrastive loss
        loss_contra_dist, multi_cls = self.sub_contra(seg_label, dist, weight=512*8)

        if not multi_cls:
            loss_contra_seglogit, _ = self.sub_contra(seg_label, seg_logit, weight=8)

        loss['loss_contra_dist'] = loss_contra_dist
        loss['loss_contra_seglogit'] = loss_contra_seglogit

        return loss

    @force_fp32(apply_to=('seg_logit', 'dist'))
    def all_contra_twoch_losses(self, seg_logit, dist, seg_label, seg_weight=None):
        """Compute segmentation loss."""

        loss = dict()
        ## contrastive loss
        loss_contra_dist, multi_cls = self.sub_contra(seg_label, dist, weight=5)

        loss_contra_seglogit, _ = self.sub_contra(seg_label, seg_logit, weight=512, m1=0.1, m2=1.0)

        loss['loss_contra_dist'] = loss_contra_dist
        loss['loss_contra_seglogit'] = loss_contra_seglogit

        return loss

    @force_fp32(apply_to=('seg_logit', 'dist'))
    def contra_losses_back(self, seg_logit, dist, seg_label, seg_weight=None):
        """Compute segmentation loss."""

        loss = dict()

        seg_logit = resize(
                input=seg_logit,
                size=seg_label.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        contra_label = seg_label.clone()
        contra_label[contra_label == 255] = 0

        mdist_pos = torch.clamp(dist - self.margin1, min=0.0)
        mdist_neg = torch.clamp(self.margin2 - dist, min=0.0)
        multi_cls = False
        if 1 in seg_label.unique() and 2 in seg_label.unique():
            multi_cls = True
            cls_1 = torch.sum(seg_label==1)
            cls_2 = torch.sum(seg_label==2)
            w1 = (cls_1+cls_2)/cls_1
            w2 = (cls_1+cls_2)/cls_2

            loss_neg_1 = (seg_label == 1) * mdist_neg
            loss_neg_2 = (seg_label == 2) * mdist_neg

            loss_neg = w1*loss_neg_1.sum() + w2*loss_neg_2.sum()
            #loss_neg = loss_neg_1.sum() + loss_neg_2.sum()
        else:
            contra_label[contra_label != 0] = 1

            loss_neg = contra_label * mdist_neg

        loss_pos = (contra_label==0) * (mdist_pos)

        num_pos = torch.sum(seg_label==0)
        num_neg = torch.sum(contra_label!=0)
        w_pos = (num_pos+num_neg)/num_pos
        w_neg = (num_pos+num_neg)/num_neg

        loss_contra = (loss_pos.sum() + loss_neg.sum())/(seg_label.shape[2])

        seg_label = seg_label.squeeze(1)
        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        loss['loss_contra'] = loss_contra
        if multi_cls:
            loss['class1'] = w1*loss_neg_1.sum()
            loss['class2'] = w2*loss_neg_2.sum()
            loss['pos'] = loss_pos.sum()

        return loss

    @force_fp32(apply_to=('seg_logit',))
    def losses_withCLS(self, seg_logit, cls_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        if self.ignore_index not in seg_label.unique():
            seg_label -= 1
        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        cls_label = torch.zeros_like(cls_logit)
        cls_label[seg_label.sum(dim=[1, 2]) > 0] = 1.

        # loss['loss_cls'] = self.loss_cls(cls_logit, cls_label)
        return loss
