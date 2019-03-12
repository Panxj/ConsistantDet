import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from ..utils import ConvModule
from ..registry import NECKS


@NECKS.register_module
class SFA_FPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 normalize=None,
                 activation=None,
                 with_sfa_loss=False):
        super(SFA_FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.with_bias = normalize is None
        self.with_sfa = True
        self.with_sfa_loss = with_sfa_loss
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

        self.sfa_l_convTs = nn.ModuleList()
        self.sfa_tp_convs = nn.ModuleList()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            sfa_l_convT = nn.ConvTranspose2d(
                in_channels[i],
                in_channels[i],
                3,
                stride=2,
                padding=1,
                output_padding=1
            )
            sfa_td_conv = ConvModule(
                in_channels[i+1],
                in_channels[i],
                1,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False) if i+1 < self.backbone_end_level else None
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)

            self.sfa_l_convTs.append(sfa_l_convT)
            if sfa_td_conv is not None:
                self.sfa_tp_convs.append(sfa_td_conv)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)


            # lvl_id = i - self.start_level
            # setattr(self, 'lateral_conv{}'.format(lvl_id), l_conv)
            # setattr(self, 'fpn_conv{}'.format(lvl_id), fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                in_channels = (self.in_channels[self.backbone_end_level - 1]
                               if i == 0 else out_channels)
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    normalize=normalize,
                    bias=self.with_bias,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

        self.sfa_conv_top = ConvModule(
            in_channels[0],
            in_channels[0]//4,
            1,
            bias=False,
            normalize=dict(type='BN', frozen=False))

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build super feature
        sfa_laterals = [self.sfa_l_convTs[i](inputs[i + self.start_level])
               for i in range(len(inputs))]

        # build top-down path
        used_backbone_levels = len(sfa_laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            sfa_laterals[i-1] += F.interpolate(
                self.sfa_tp_convs[i-1](sfa_laterals[i]), scale_factor=2, mode='nearest')

        # build laterals
        laterals = [
            lateral_conv(sfa_laterals[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                orig = inputs[self.backbone_end_level - 1]
                outs.append(self.fpn_convs[used_backbone_levels](orig))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    # BUG: we should add relu before each extra conv
                    outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)

    def loss(self, up_x, large_x, stage=1):
        losses = dict()
        if stage == 1:
            up_x = self.sfa_conv_top(up_x)
        assert isinstance(large_x,list)
        for l_x in large_x:
            l_feat_h, l_feat_w = l_x.size(2), l_x.size(3)
            losses['sfa_loss'] = F.mse_loss(up_x[:,:,:l_feat_h,:l_feat_w], l_x, reduction='mean')
        return losses