import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from ..utils import ConvModule
from ..registry import NECKS
from mmdet.core import bbox_scale

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
                 with_sfa_loss=False,
                 with_ss_loss=False,
                 with_orig =False,
                 with_sfa = False,
                 orig_loss_weight=1.0,
                 only_sfa_result=False,
                 only_orig_result=False,
                 segm_out_flag=0,
                 with_rpn_clip=False,
                 loss_weight=1.0,
                 aug_channels=True,
                 up_shuffle=False,
                 l_td_cat=True,
                 with_residual=False):
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
        self.with_orig = with_orig
        self.with_sfa = with_sfa
        assert self.with_orig or self.with_sfa
        self.only_sfa_result = only_sfa_result
        self.only_orig_result = only_orig_result
        self.segm_out_flag = segm_out_flag
        self.with_rpn_clip = with_rpn_clip
        self.loss_weight = loss_weight
        self.orig_loss_weight = orig_loss_weight
        self.aug_channels = aug_channels
        self.up_shufle=up_shuffle
        self.l_td_cat=l_td_cat
        self.with_residual = with_residual
        self.with_ss_loss = with_ss_loss
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

        self.sfa_l_ups = nn.ModuleList()
        if self.l_td_cat:
            self.sfa_l_1x1_mixs = nn.ModuleList()
        self.sfa_l_dim_reds = nn.ModuleList()
        self.sfa_tp_convs = nn.ModuleList()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            if self.with_sfa:
                if self.up_shufle:
                    if self.with_residual:
                        sfa_l_reduce_dim = nn.Sequential(
                            nn.Conv2d(in_channels[i], out_channels, 1),
                            nn.ReLU(inplace=True))
                        sfa_l_up = nn.Sequential(
                            nn.Conv2d(out_channels, out_channels*4, 3, padding=1),
                            nn.ReLU(inplace=True),
                            nn.PixelShuffle(upscale_factor=2)
                        )
                        sfa_td_conv = ConvModule(
                            out_channels,
                            out_channels,
                            1,
                            normalize=normalize,
                            bias=self.with_bias,
                            activation=self.activation,
                            inplace=False) if i + 1 < self.backbone_end_level else None
                        self.sfa_l_dim_reds.append(sfa_l_reduce_dim)
                    else:
                        if self.l_td_cat:
                            sfa_l_up = nn.Sequential(
                                nn.PixelShuffle(upscale_factor=2),
                                nn.Conv2d(in_channels[i] //4, in_channels[i], 3, padding=1)
                            ) if i == self.backbone_end_level -1 else nn.Sequential(
                                nn.PixelShuffle(upscale_factor=2),
                                nn.Conv2d(in_channels[i] //4, in_channels[i]//2, 3, padding=1)
                            )
                            sfa_td_conv = nn.Sequential(
                                nn.PixelShuffle(upscale_factor=2),
                                nn.Conv2d(in_channels[i + 1] // 4, in_channels[i] // 2, 1)
                            ) if i + 1 < self.backbone_end_level else None
                            sfa_l_1x1_mix = ConvModule(
                                in_channels[i],
                                in_channels[i],
                                1,
                                normalize=normalize,
                                bias=self.with_bias,
                                activation=self.activation,
                                inplace=False) if i + 1 < self.backbone_end_level else None
                        else:
                            sfa_l_up = nn.Sequential(
                                nn.PixelShuffle(upscale_factor=2),
                                nn.Conv2d(in_channels[i] // 4, in_channels[i], 3, padding=1)
                            )
                            sfa_td_conv = nn.Sequential(
                                nn.PixelShuffle(upscale_factor=2),
                                nn.Conv2d(in_channels[i + 1] // 4, in_channels[i] // 2, 1))
                        # sfa_td_conv = ConvModule(
                        #     out_channels,
                        #     out_channels,
                        #     1,
                        #     normalize=normalize,
                        #     bias=self.with_bias,
                        #     activation=self.activation,
                        #     inplace=False) if i + 1 < self.backbone_end_level else None
                else:
                    if self.aug_channels:
                        if self.l_td_cat:
                            sfa_l_up = nn.ConvTranspose2d(
                                in_channels[i],
                                in_channels[i],
                                3,
                                stride=2,
                                padding=1,
                                output_padding=1
                            ) if i == self.backbone_end_level -1 else nn.ConvTranspose2d(
                                in_channels[i],
                                in_channels[i]//2,
                                3,
                                stride=2,
                                padding=1,
                                output_padding=1
                            )
                            sfa_td_conv = nn.Sequential(
                                nn.PixelShuffle(upscale_factor=2),
                                nn.Conv2d(in_channels[i+1]//4, in_channels[i]//2, 1)
                            ) if i+1 < self.backbone_end_level else None
                            sfa_l_1x1_mix = ConvModule(
                                in_channels[i],
                                in_channels[i],
                                1,
                                normalize=normalize,
                                bias=self.with_bias,
                                activation=self.activation,
                                inplace=False) if i+1 < self.backbone_end_level else None
                        # sfa_td_conv = ConvModule(
                        #     in_channels[i+1],
                        #     in_channels[i],
                        #     1,
                        #     normalize=normalize,
                        #     bias=self.with_bias,
                        #     activation=self.activation,
                        #     inplace=False) if i+1 < self.backbone_end_level else None
                    else:
                        sfa_l_up = nn.ConvTranspose2d(
                            in_channels[i],
                            out_channels,
                            3,
                            stride=2,
                            padding=1,
                            output_padding=1
                        )
                        sfa_td_conv = ConvModule(
                            out_channels,
                            out_channels,
                            1,
                            normalize=normalize,
                            bias=self.with_bias,
                            activation=self.activation,
                            inplace=False) if i + 1 < self.backbone_end_level else None
                self.sfa_l_ups.append(sfa_l_up)
                self.sfa_l_1x1_mixs.append(sfa_l_1x1_mix)
                if sfa_td_conv is not None:
                    self.sfa_tp_convs.append(sfa_td_conv)
            if self.with_orig or self.aug_channels:
                l_conv = ConvModule(
                    in_channels[i],
                    out_channels,
                    1,
                    normalize=normalize,
                    bias=self.with_bias,
                    activation=self.activation,
                    inplace=False)
                self.lateral_convs.append(l_conv)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)


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
        if self.with_sfa:
            if self.aug_channels:
                self.sfa_conv_top = ConvModule(
                    in_channels[0],
                    in_channels[0]//4,
                    1,
                    bias=self.with_bias,
                    activation=self.activation,
                    )
            else:
                self.sfa_conv_top = ConvModule(
                    out_channels,
                    out_channels//4,
                    1,
                    bias=self.with_bias,
                    activation=self.activation,
                )

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        outs_list = []
        if self.with_sfa:
            if self.up_shufle and self.with_residual:
                sfa_lat_dim_reds = [self.sfa_l_dim_reds[i](inputs[i + self.start_level])
                                  for i in range(len(inputs))]
                sfa_laterals_1 = [self.sfa_l_ups[i](sfa_lat_dim_reds[i + self.start_level]) +
                                     F.interpolate(sfa_lat_dim_reds[i + self.start_level], scale_factor=2,
                                    mode='nearest') for i in range(len(inputs))]
            else:
                # build super feature
                sfa_laterals_1 = [self.sfa_l_ups[i](inputs[i + self.start_level])
                       for i in range(len(inputs))]
                # sfa_laterals_1 = [F.interpolate(inputs[i + self.start_level], scale_factor=2, mode='nearest')
                #                   for i in range(len(inputs))]

            # build top-down path
            used_backbone_levels = len(sfa_laterals_1)
            for i in range(used_backbone_levels - 1, 0, -1):
                #1. default td interpolate
                # sfa_laterals_1[i-1] += F.interpolate(
                #     self.sfa_tp_convs[i-1](sfa_laterals_1[i]), scale_factor=2, mode='nearest')

                #2. td shuffle
                # sfa_laterals_1[i - 1] += self.sfa_tp_convs[i - 1](sfa_laterals_1[i])

                sfa_laterals_1[i - 1] = self.sfa_l_1x1_mixs[i-1](torch.cat((sfa_laterals_1[i - 1],
                                                   self.sfa_tp_convs[i - 1](sfa_laterals_1[i])),
                                                  dim=1))

            if self.aug_channels:
                # build laterals for sfa
                sfa_laterals_2 = [
                    lateral_conv(sfa_laterals_1[i + self.start_level])
                    for i, lateral_conv in enumerate(self.lateral_convs)
                ]

                # build top-down path for sfa
                used_backbone_levels = len(sfa_laterals_2)
                for i in range(used_backbone_levels - 1, 0, -1):
                    sfa_laterals_2[i - 1] += F.interpolate(
                        sfa_laterals_2[i], scale_factor=2, mode='nearest')

                # build outputs for sfa
                # part 1: from original levels
                sfa_outs = [
                    self.fpn_convs[i](sfa_laterals_2[i]) for i in range(used_backbone_levels)
                ]
            else:
                sfa_outs = [
                    self.fpn_convs[i](sfa_laterals_1[i]) for i in range(used_backbone_levels)
                ]
            outs_list = [sfa_outs]
        if self.with_orig:
            # build laterals
            orig_laterals = [
                lateral_conv(inputs[i + self.start_level])
                for i, lateral_conv in enumerate(self.lateral_convs)
            ]

            # build top-down path
            used_backbone_levels = len(orig_laterals)
            for i in range(used_backbone_levels - 1, 0, -1):
                orig_laterals[i - 1] += F.interpolate(
                    orig_laterals[i], scale_factor=2, mode='nearest')

            # build outputs
            # part 1: from original levels
            orig_outs = [
                self.fpn_convs[i](orig_laterals[i]) for i in range(used_backbone_levels)
            ]
            outs_list =[orig_outs]
        if self.with_orig and self.with_sfa:
            outs_list= [sfa_outs, orig_outs]
        final_outs = []
        for outs in outs_list:
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
            final_outs.append(outs)
        return tuple(final_outs), sfa_laterals_1[0]

    def loss(self, up_x, large_x, stage=1, proposal=None,stride=4, pro_flag=1):
        losses = dict()
        if stage == 1:
            up_x = self.sfa_conv_top(up_x)
        assert isinstance(large_x,list)
        for l_x in large_x:
            l_feat_h, l_feat_w = l_x.size(2), l_x.size(3)
            if proposal is None:
                losses['sfa_loss'] = F.mse_loss(up_x[:,:,:l_feat_h,:l_feat_w], l_x, reduction='mean')
            else:
                if pro_flag == 0:
                    loss = 0
                    avg_size = 0
                    num_imgs = len(proposal)
                    for i in range(num_imgs):
                        proposals = proposal[i]
                        avg_size += proposals.size(0)
                        proposals = (proposals / stride).round()
                        proposals = bbox_scale(proposals, scale_factor=1.5, img_shape=[l_feat_h, l_feat_w])
                        proposals = proposals.int()
                        for j in range(proposals.size(0)):
                            loss += F.mse_loss(up_x[i][:, proposals[j, 1]:(proposals[j, 3] + 1),
                                               proposals[j, 0]:(proposals[j, 2] + 1)],
                                               l_x[i][:, proposals[j, 1]:(proposals[j, 3] + 1),
                                               proposals[j, 0]:(proposals[j, 2] + 1)],
                                               reduction='mean')
                    loss /= avg_size
                    loss *= self.loss_weight
                    losses['sfa_loss'] = loss
                elif pro_flag == 1:
                    loss = 0
                    avg_size = proposal.size(0)
                    proposal[:,1:] = (proposal[:,1:]/stride).round()
                    proposal = proposal.int()
                    proposal[:, 1] = proposal[:, 1].clamp(0, l_feat_w - 1)
                    proposal[:, 3] = proposal[:, 3].clamp(0, l_feat_w - 1)
                    proposal[:, 2] = proposal[:, 2].clamp(0, l_feat_h - 1)
                    proposal[:, 4] = proposal[:, 4].clamp(0, l_feat_h - 1)
                    num_imgs = l_x.size(0)
                    for i in range(num_imgs):
                        inds = torch.nonzero(proposal[:,0] == i).squeeze()
                        proposals = proposal[inds,:][:,1:]
                        for j in range(proposals.size(0)):
                            loss += F.mse_loss(up_x[i][:, proposals[j,1]:(proposals[j,3]+1),
                                               proposals[j,0]:(proposals[j,2]+1)],
                                               l_x[i][:, proposals[j,1]:(proposals[j,3]+1),
                                               proposals[j,0]:(proposals[j,2]+1)],
                                               reduction='mean')
                    loss /= avg_size
                    loss *= self.loss_weight
                    losses['sfa_loss'] = loss
                else:
                    raise Exception('Not implementation.')
        return losses