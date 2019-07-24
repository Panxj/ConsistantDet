import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from .DCNv2.dcn_v2 import DCNv2
from .DCNv2.dcn_v2 import DCN
from .DCNv2.dcn_v2 import dcn_v2_conv

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _nms(heat, kernel=1):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _topk(scores, K=20):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)

    topk_clses = (topk_inds / (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

def _decode(
    tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr, 
    K=100, kernel=1, ae_threshold=1, num_dets=1000, no_border=False
):
    batch, cat, height, width = tl_heat.size()

    tl_heat = torch.sigmoid(tl_heat)
    br_heat = torch.sigmoid(br_heat)

    # perform nms on heatmaps
    tl_heat = _nms(tl_heat, kernel=kernel)
    br_heat = _nms(br_heat, kernel=kernel)

    tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk(tl_heat, K=K)
    br_scores, br_inds, br_clses, br_ys, br_xs = _topk(br_heat, K=K)

    tl_ys = tl_ys.view(batch, K, 1).expand(batch, K, K)
    tl_xs = tl_xs.view(batch, K, 1).expand(batch, K, K)
    br_ys = br_ys.view(batch, 1, K).expand(batch, K, K)
    br_xs = br_xs.view(batch, 1, K).expand(batch, K, K)

    if no_border:
        tl_ys_binds = (tl_ys == 0)
        tl_xs_binds = (tl_xs == 0)
        br_ys_binds = (br_ys == height - 1)
        br_xs_binds = (br_xs == width  - 1)

    if tl_regr is not None and br_regr is not None:
        tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
        tl_regr = tl_regr.view(batch, K, 1, 2)
        br_regr = _tranpose_and_gather_feat(br_regr, br_inds)
        br_regr = br_regr.view(batch, 1, K, 2)

        tl_xs = tl_xs + tl_regr[..., 0]
        tl_ys = tl_ys + tl_regr[..., 1]
        br_xs = br_xs + br_regr[..., 0]
        br_ys = br_ys + br_regr[..., 1]

    # all possible boxes based on top k corners (ignoring class)
    bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)

    tl_tag = _tranpose_and_gather_feat(tl_tag, tl_inds)
    tl_tag = tl_tag.view(batch, K, 1)
    br_tag = _tranpose_and_gather_feat(br_tag, br_inds)
    br_tag = br_tag.view(batch, 1, K)
    dists  = torch.abs(tl_tag - br_tag)

    tl_scores = tl_scores.view(batch, K, 1).expand(batch, K, K)
    br_scores = br_scores.view(batch, 1, K).expand(batch, K, K)
    scores    = (tl_scores + br_scores) / 2

    # reject boxes based on classes
    tl_clses = tl_clses.view(batch, K, 1).expand(batch, K, K)
    br_clses = br_clses.view(batch, 1, K).expand(batch, K, K)
    cls_inds = (tl_clses != br_clses)

    # reject boxes based on distances
    dist_inds = (dists > ae_threshold)

    # reject boxes based on widths and heights
    width_inds  = (br_xs < tl_xs)
    height_inds = (br_ys < tl_ys)

    if no_border:
        scores[tl_ys_binds] = -1
        scores[tl_xs_binds] = -1
        scores[br_ys_binds] = -1
        scores[br_xs_binds] = -1

    scores[cls_inds]    = -1
    scores[dist_inds]   = -1
    scores[width_inds]  = -1
    scores[height_inds] = -1

    scores = scores.view(batch, -1)
    scores, inds = torch.topk(scores, num_dets)
    scores = scores.unsqueeze(2)

    bboxes = bboxes.view(batch, -1, 4)
    bboxes = _gather_feat(bboxes, inds)

    clses  = tl_clses.contiguous().view(batch, -1, 1)
    clses  = _gather_feat(clses, inds).float()

    tl_scores = tl_scores.contiguous().view(batch, -1, 1)
    tl_scores = _gather_feat(tl_scores, inds).float()
    br_scores = br_scores.contiguous().view(batch, -1, 1)
    br_scores = _gather_feat(br_scores, inds).float()

    detections = torch.cat([bboxes, scores, tl_scores, br_scores, clses], dim=2)
    return detections

class upsample(nn.Module):
    def __init__(self, scale_factor):
        super(upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor)

class merge(nn.Module):
    def forward(self, x, y):
        return x + y

class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn   = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn   = self.bn(conv)
        relu = self.relu(bn)
        return relu

class residual(nn.Module):
    def __init__(self, inp_dim, out_dim, k=3, stride=1):
        super(residual, self).__init__()
        p = (k - 1) // 2

        self.conv1 = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(p, p), stride=(stride, stride), bias=False)
        self.bn1   = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (k, k), padding=(p, p), bias=False)
        self.bn2   = nn.BatchNorm2d(out_dim)
        
        self.skip  = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2   = self.bn2(conv2)

        skip  = self.skip(x)
        return self.relu(bn2 + skip)

class corner_pool(nn.Module):
    def __init__(self, dim, pool1, pool2, mode=0):
        super(corner_pool, self).__init__()
        self._init_layers(dim, pool1, pool2, mode=mode)

    def _init_layers(self, dim, pool1, pool2, mode=0):
        self.mode = mode
        if mode==0:
            self.filter1 = corner_filter(128,128,0,4)
            self.filter2 = corner_filter(128,128,1,4)
        elif mode==1:
            self.filter1 = corner_filter(128, 128, 2, 4)
            self.filter2 = corner_filter(128, 128, 3, 4)
        else:
            raise Exception('Wrong mode number for class corner_filter.')
        self.p1_conv1 = convolution(3, dim, 128)
        self.p2_conv1 = convolution(3, dim, 128)

        self.p_conv1 = nn.Conv2d(128, dim, (3, 3), padding=(1, 1), bias=False)
        self.p_bn1   = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, (1, 1), bias=False)
        self.bn1   = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = convolution(3, dim, dim)

        self.pool1 = pool1()
        self.pool2 = pool2()

    def forward(self, x):
        # pool 1
        # image_name='000000226417'
        # draw_heatmaps(x.data.cpu().numpy(), '{}_conv_backbone_{}.jpg'.format(image_name, self.mode), ratio=50.)
        # input = torch.randn(1,1,5,5).cuda()
        # dcn = DCN(1,1,kernel_size=(3,3), stride=1,padding=1).cuda()
        # dcv_conv_mask_o = dcn.conv_offset_mask(input)
        # dcn_x = dcn(input)
        p1_conv1 = self.p1_conv1(x)
        # f1_conv1 = F.relu(self.filter1(p1_conv1))
        # sparse_p1 = _nms(p1_conv1, kernel=5)
        # pool1    = self.pool1(sparse_p1)
        pool1    = self.pool1(p1_conv1)
        # draw_heatmaps(p1_conv1.data.cpu().numpy(), '{}_conv_top_{}.jpg'.format(image_name, self.mode), ratio=100.)
        # # draw_heatmaps(f1_conv1.data.cpu().numpy(), '000000000785_afterf_top.jpg', ratio=100.)
        # draw_heatmaps(pool1.data.cpu().numpy(), '{}_afterp_top_{}.jpg'.format(image_name, self.mode), ratio=100.)

        # pool 2
        p2_conv1 = self.p2_conv1(x)
        # f2_conv1 = F.relu(self.filter2(p2_conv1))
        # sparse_p2 = _nms(p2_conv1, kernel=5)
        # pool2    = self.pool2(sparse_p2)
        pool2    = self.pool2(p2_conv1)
        # draw_heatmaps(p2_conv1.data.cpu().numpy(), '{}_conv_left_{}.jpg'.format(image_name, self.mode), ratio=100.)
        # # draw_heatmaps(f2_conv1.data.cpu().numpy(), '000000000785_afterf_left.jpg', ratio=100.)
        # draw_heatmaps(pool2.data.cpu().numpy(), '{}_afterp_left_{}.jpg'.format(image_name, self.mode), ratio=100.)

        # pool 1 + pool 2
        p_conv1 = self.p_conv1(pool1 + pool2)
        # sparse_p3 = _nms(p_conv1, kernel=3)
        p_bn1   = self.p_bn1(p_conv1)

        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)

        conv2 = self.conv2(relu1)
        # draw_heatmaps(conv2.data.cpu().numpy(), '{}_conv_lt_{}.jpg'.format(image_name, self.mode), ratio=100.)
        return conv2


class DcnPool(nn.Module):
    def __init__(self, dim_in, dim_out, mode0, mode1, kernel=(3,3), padding=1, group=1):
        # **********************************
        # mode: 1 for top-left corner
        #       2 for bottom-right corner
        # ==================================
        super(DcnPool, self).__init__()
        self._init_layers(dim_in, dim_out, mode0, mode1, kernel=kernel, padding=padding, group=group)

    def _init_layers(self, dim_in, dim_out, mode0, mode1, kernel, padding, group):
        self.corner_conv = convolution(3, dim_in, dim_out)
        self.p1_conv1 = convolution(3, dim_in, dim_out)
        self.p2_conv1 = convolution(3, dim_in, dim_out)

        self.p_conv1 = nn.Conv2d(dim_out, dim_in, (3, 3), padding=(1, 1), bias=False)
        self.p_bn1 = nn.BatchNorm2d(dim_in)

        self.conv1 = nn.Conv2d(dim_in, dim_in, (1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(dim_in)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = convolution(3, dim_in, dim_in)

        self.pool1 = DCNPooling(dim_out, dim_out, kernel, 1, padding, mode0)
        self.pool2 = DCNPooling(dim_out, dim_out, kernel, 1, padding, mode1)
        self.dcn_bn= nn.BatchNorm2d(dim_out)


    def forward(self, x):
        p1_conv1 = self.p1_conv1(x)
        pool1, offset_1 = self.pool1(p1_conv1)

        p2_conv1 = self.p2_conv1(x)
        pool2, offset_2 = self.pool2(p2_conv1)

        p_conv1 = self.p_conv1(pool1 + pool2)
        p_bn1 = self.p_bn1(p_conv1)

        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)

        conv2 = self.conv2(relu1)
        return conv2, torch.cat((offset_1,offset_2), dim=1)

    # generate feature maps shifted based on offset_mask of DCN
    # def gen_off(self, feat):
    #     dcn_offset = self.dcn.conv_offset_mask(feat)
    #     x, y, mask = torch.chunk(dcn_offset, 3, dim=1)
    #     offset = torch.cat((x, y), dim=1)
    #     modulate_ratio = torch.sigmoid(mask)
    #     shift_coor = None
    #
    #     return offset, shift_coor


def draw_image(img, file):
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(file, img)

def map_to_image(heat, ratio=5.):
    # img = np.tile(heat[:,:,None], (1,1,3))
    img = heat* ratio
    return img

def draw_heatmaps(heats, file, ratio=5.):
    # heat_max_cls = np.max(heats, axis=1)
    for i in range(100):
        file_i = file.replace('.', '_{}.'.format(i))
        heat_i = heats[0,i,...]
        # heat_0 = heat_max_cls[0, ...]
        image_0 = map_to_image(heat_i, ratio=ratio)
        draw_image(image_0, file_i)


class corner_filter(nn.Module):
    def __init__(self, dim_out, dim_in, mode, radius):
        super(corner_filter, self).__init__()
        self._init_layers(dim_out, dim_in, mode, radius)
        # -------- filter mode-----------
        # top_mode : 0
        # left_mode : 1
        # bottom_mode : 2
        # right_mode : 3
        # -------------------------------
    def _init_layers(self, dim_out, dim_in, mode, radius):
        self.groups = dim_in
        self.padding = radius
        kernel = self.gene_ker(radius, dim_out, mode)
        self.ker_w = nn.Parameter(data=kernel, requires_grad=False)
    def forward(self, x):
        after_filter = F.conv2d(x, self.ker_w, padding=self.padding, groups=self.groups)
        return after_filter
    @staticmethod
    def gene_ker(radius, dim_out, mode):
        H = W = 2*radius +1
        h_inds, w_inds = np.ogrid[:H,:W]
        dist_from_center = np.sqrt((h_inds - radius)**2 + (w_inds - radius)**2)
        circle_mask = dist_from_center <= radius

        if mode == 1 or mode == 3:
            dist_mode = np.abs(w_inds - radius) + h_inds - radius
            if mode == 1 :
                dist_mode = dist_mode[::-1, :]
        elif mode == 0 or mode == 2:
            dist_mode = np.abs(h_inds - radius) + w_inds - radius
            if mode ==0 :
                dist_mode = dist_mode[:,::-1]
        else:
            raise Exception('Wrong number for filter mode in class corner_filter.')

        # if mode == 1 or mode == 3:
        #     dist_mode = np.abs(h_inds - radius) + w_inds - radius
        #     if mode == 1 :
        #         dist_mode = dist_mode[:,::-1]
        # elif mode == 0 or mode == 2:
        #     dist_mode = np.abs(w_inds - radius) + h_inds - radius
        #     if mode ==0 :
        #         dist_mode = dist_mode[::-1, :]
        # else:
        #     raise Exception('Wrong number for filter mode in class corner_filter.')
        circle_mask[dist_mode<=0] = 0
        # circle_mask = (circle_mask * -1. / np.sum(circle_mask)).astype(np.float)
        circle_mask = (circle_mask * -1 ).astype(np.float)
        circle_mask[radius, radius] = -np.sum(circle_mask)
        kernel = torch.Tensor(circle_mask)
        kernel = kernel.expand(dim_out, 1,H, W)
        return kernel

class DCNPooling(DCNv2):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,mode,
                 dilation=1, deformable_groups=1):
        super(DCNPooling, self).__init__(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, deformable_groups)

        channels_ = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(self.in_channels,
                                          channels_,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=True)
        self.mode = mode
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        edge_zero = torch.zeros_like(o1)
        offset = torch.cat((o1,o2), dim=1)
        mask = torch.sigmoid(mask)
        # ***********************************
        #
        # 0: for top pooling (left extreme point)
        # 1: for left pooling (top extreme point)
        # 2: for bottom pooling (right extreme point)
        # 3: for right pooling (bottom extreme point)
        #
        # ***********************************
        keep = torch.zeros_like(edge_zero)
        if self.mode == 0:
            keep = (offset[:,1::2,:,:] >= 0).float()
            offset[:,0::2,:,:] *= edge_zero

        elif self.mode == 1:
            keep = (offset[:,0::2,:,:] >= 0).float()
            offset[:,1::2,:,:] *= edge_zero

        elif self.mode == 2:
            keep = (offset[:,1::2,:,:] <= 0).float()
            offset[:,0::2,:,:] *= edge_zero

        elif self.mode == 3:
            keep = (offset[:,0::2,:,:] <= 0).float()
            offset[:,1::2,:,:] *= edge_zero

        mask = keep * mask
        return dcn_v2_conv(input, offset, mask,
                           self.weight, self.bias,
                           self.stride,
                           self.padding,
                           self.dilation,
                           self.deformable_groups), offset


# class corner_filter(object):
#     def __init__(self, dim_out, dim_in, mode, radius):
#         super(corner_filter, self).__init__()
#         self._init_layers(dim_out, dim_in, mode, radius)
#         # -------- filter mode-----------
#         # top_mode : 0
#         # left_mode : 1
#         # bottom_mode : 2
#         # right_mode : 3
#         # -------------------------------
#     def _init_layers(self, dim_out, dim_in, mode, radius):
#         self.groups = dim_in
#         self.padding = radius
#         self.kernel = self.gene_ker(radius, dim_out, mode)
#         # self.ker_w = nn.Parameter(data=kernel, requires_grad=False)
#
#     def forward(self, x):
#         after_filter = F.conv2d(x, self.kernel, padding=self.padding, groups=self.groups)
#         return after_filter
#     def __call__(self, x):
#         return self.forward(x)
#     @staticmethod
#     def gene_ker(radius, dim_out,mode):
#         H = W = 2*radius +1
#         h_inds, w_inds = np.ogrid[:H,:W]
#         dist_from_center = np.sqrt((h_inds - radius)**2 + (w_inds - radius)**2)
#         circle_mask = dist_from_center <= radius
#
#         if mode == 0 or mode == 2:
#             dist_mode = np.abs(w_inds - radius) + h_inds - radius
#             if mode == 0 :
#                 dist_mode = dist_mode[::-1, :]
#         elif mode == 1 or mode == 3:
#             dist_mode = np.abs(h_inds - radius) + w_inds - radius
#             if mode ==1 :
#                 dist_mode = dist_mode[:,::-1]
#         else:
#             raise Exception('Wrong number for filter mode in class corner_filter.')
#         circle_mask[dist_mode<=0] = 0
#         circle_mask = (circle_mask * -1. / np.sum(circle_mask)).astype(np.float)
#
#         circle_mask[radius, radius] = -np.sum(circle_mask)
#         kernel = torch.Tensor(circle_mask).cuda()
#         kernel = kernel.expand(dim_out, 1,H, W)
#         return kernel