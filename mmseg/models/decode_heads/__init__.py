# Copyright (c) OpenMMLab. All rights reserved.
from .ann_head import ANNHead
from .apc_head import APCHead
from .aspp_head import ASPPHead
from .cc_head import CCHead
from .da_head import DAHead
from .dm_head import DMHead
from .dnl_head import DNLHead
from .dpt_head import DPTHead
from .ema_head import EMAHead
from .enc_head import EncHead
from .fcn_head import FCNHead
from .fpn_head import FPNHead
from .gc_head import GCHead
from .isa_head import ISAHead
from .lraspp_head import LRASPPHead
from .nl_head import NLHead
from .ocr_head import OCRHead
from .point_head import PointHead
from .psa_head import PSAHead
from .psp_head import PSPHead
from .segformer_head import SegformerHead
from .sep_aspp_head import DepthwiseSeparableASPPHead
from .sep_fcn_head import DepthwiseSeparableFCNHead
from .setr_mla_head import SETRMLAHead
from .setr_up_head import SETRUPHead
from .uper_head import UPerHead
from .siamese_fcn_head import SiameseFCNHead
from .diffPFN_head import diffFPNHead
from .att_fcn_head import AttFCNHead
from .att_fcn_head_hwatt import AttFCNHead_HWAtt
from .att_fcn_head_separate import AttFCNHead_separate
from .siamese_segformer_head import SiameseSegformerHead
from .att_fcn_head_onech import AttFCNHead_onech
from .segformer_head_contra import SegformerHeadContra
#from .mask2former_head import Mask2FormerHead
from .segformer_head_contra_and_ce import SegformerHeadContra_and_CE

__all__ = [
    'FCNHead', 'PSPHead', 'ASPPHead', 'PSAHead', 'NLHead', 'GCHead', 'CCHead',
    'UPerHead', 'DepthwiseSeparableASPPHead', 'ANNHead', 'DAHead', 'OCRHead',
    'EncHead', 'DepthwiseSeparableFCNHead', 'FPNHead', 'EMAHead', 'DNLHead',
    'PointHead', 'APCHead', 'DMHead', 'LRASPPHead', 'SETRUPHead','AttFCNHead',
    'SETRMLAHead', 'DPTHead', 'SETRMLAHead', 'SegformerHead', 'ISAHead','SiameseFCNHead',
    'diffFPNHead','SiameseSegformerHead','AttFCNHead_HWAtt', 'AttFCNHead_separate',
    'AttFCNHead_onech','SegformerHeadContra','SegformerHeadContra_and_CE'
]
