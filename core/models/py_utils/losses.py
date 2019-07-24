import torch
import torch.nn as nn

from .utils import _tranpose_and_gather_feat

def _sigmoid(x):
    return torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)

def _ae_loss(tag0, tag1, mask):
    num  = mask.sum(dim=1, keepdim=True).float()
    tag0 = tag0.squeeze()
    tag1 = tag1.squeeze()

    tag_mean = (tag0 + tag1) / 2

    tag0 = torch.pow(tag0 - tag_mean, 2) / (num + 1e-4)
    tag0 = tag0[mask].sum()
    tag1 = torch.pow(tag1 - tag_mean, 2) / (num + 1e-4)
    tag1 = tag1[mask].sum()
    pull = tag0 + tag1

    mask = mask.unsqueeze(1) + mask.unsqueeze(2)
    mask = mask.eq(2)
    num  = num.unsqueeze(2)
    num2 = (num - 1) * num
    dist = tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2)
    dist = 1 - torch.abs(dist)
    dist = nn.functional.relu(dist, inplace=True)
    dist = dist - 1 / (num + 1e-4)
    dist = dist / (num2 + 1e-4)
    dist = dist[mask]
    push = dist.sum()
    return pull, push

def _off_loss(off, gt_off, mask):
    num  = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_off)

    off    = off[mask]
    gt_off = gt_off[mask]
    
    off_loss = nn.functional.smooth_l1_loss(off, gt_off, reduction="sum")
    off_loss = off_loss / (num + 1e-4)
    return off_loss


def _dcn_off_loss(dcn_off, dcn_gt_off, mask, mode=None):
    num = mask.float().sum()
    mask_gt = mask.unsqueeze(2).expand_as(dcn_gt_off)
    dcn_gt_off = dcn_gt_off[mask_gt]
    cnt_point = dcn_off.size(2)//4
    y_offset = dcn_off[:, :, 1:cnt_point*2:2]
    x_offset = dcn_off[:, :, 2*cnt_point::2]

    mask_pred = mask.unsqueeze(2).expand_as(y_offset)
    y_offset = y_offset[mask_pred]
    x_offset = x_offset[mask_pred]

    dcn_gt_off_y = dcn_gt_off[1::2].view(-1,1).expand(-1,cnt_point).contiguous().view(-1)
    dcn_gt_off_x = dcn_gt_off[0::2].view(-1,1).expand(-1,cnt_point).contiguous().view(-1)
    dcn_off_loss = torch.Tensor([0.]).cuda()

    if mode == 'tl':
        dcn_off_loss +=torch.sum(torch.clamp(-y_offset, min=0) + torch.clamp(y_offset - dcn_gt_off_y, min=0))
        dcn_off_loss +=torch.sum(torch.clamp(-x_offset, min=0) + torch.clamp(x_offset - dcn_gt_off_x, min=0))
    elif mode =='br':
        dcn_off_loss += torch.sum(torch.clamp(y_offset, min=0) + torch.clamp( -dcn_gt_off_y - y_offset, min=0))
        dcn_off_loss += torch.sum(torch.clamp(x_offset, min=0) + torch.clamp(- dcn_gt_off_x - x_offset , min=0))
    else:
        raise Exception('Wrong mode for dcn_off_loss')
    dcn_off_loss = dcn_off_loss / (num + 1e-4)/cnt_point
    return dcn_off_loss

def _fea_off_loss(fea_off, target_range):
    pass


def _fea_cls_loss(fea_cls, target_cls):
    pass


def _focal_loss_mask(preds, gt, mask):
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    pos_mask = mask[pos_inds]
    neg_mask = mask[neg_inds]

    loss = 0
    for pred in preds:
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * pos_mask
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights * neg_mask

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

def _focal_loss(preds, gt):
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    loss = 0
    for pred in preds:
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

class CornerNet_Saccade_Loss(nn.Module):
    def __init__(self, pull_weight=1, push_weight=1, off_weight=1, focal_loss=_focal_loss_mask):
        super(CornerNet_Saccade_Loss, self).__init__()

        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.off_weight  = off_weight
        self.focal_loss  = focal_loss
        self.ae_loss     = _ae_loss
        self.off_loss    = _off_loss

    def forward(self, outs, targets):
        tl_heats = outs[0]
        br_heats = outs[1]
        tl_tags  = outs[2]
        br_tags  = outs[3]
        tl_offs  = outs[4]
        br_offs  = outs[5]
        atts     = outs[6]

        gt_tl_heat  = targets[0]
        gt_br_heat  = targets[1]
        gt_mask     = targets[2]
        gt_tl_off   = targets[3]
        gt_br_off   = targets[4]
        gt_tl_ind   = targets[5]
        gt_br_ind   = targets[6]
        gt_tl_valid = targets[7]
        gt_br_valid = targets[8]
        gt_atts     = targets[9]

        # focal loss
        focal_loss = 0

        tl_heats = [_sigmoid(t) for t in tl_heats]
        br_heats = [_sigmoid(b) for b in br_heats]

        focal_loss += self.focal_loss(tl_heats, gt_tl_heat, gt_tl_valid)
        focal_loss += self.focal_loss(br_heats, gt_br_heat, gt_br_valid)

        atts = [[_sigmoid(a) for a in att] for att in atts]
        atts = [[att[ind] for att in atts] for ind in range(len(gt_atts))]

        att_loss = 0
        for att, gt_att in zip(atts, gt_atts):
            att_loss += _focal_loss(att, gt_att) / max(len(att), 1)

        # tag loss
        pull_loss = 0
        push_loss = 0
        tl_tags   = [_tranpose_and_gather_feat(tl_tag, gt_tl_ind) for tl_tag in tl_tags]
        br_tags   = [_tranpose_and_gather_feat(br_tag, gt_br_ind) for br_tag in br_tags]
        for tl_tag, br_tag in zip(tl_tags, br_tags):
            pull, push = self.ae_loss(tl_tag, br_tag, gt_mask)
            pull_loss += pull
            push_loss += push
        pull_loss = self.pull_weight * pull_loss
        push_loss = self.push_weight * push_loss

        off_loss = 0
        tl_offs  = [_tranpose_and_gather_feat(tl_off, gt_tl_ind) for tl_off in tl_offs]
        br_offs  = [_tranpose_and_gather_feat(br_off, gt_br_ind) for br_off in br_offs]
        for tl_off, br_off in zip(tl_offs, br_offs):
            off_loss += self.off_loss(tl_off, gt_tl_off, gt_mask)
            off_loss += self.off_loss(br_off, gt_br_off, gt_mask)
        off_loss = self.off_weight * off_loss

        loss = (focal_loss + att_loss + pull_loss + push_loss + off_loss) / max(len(tl_heats), 1)
        return loss.unsqueeze(0)

class CornerNet_Loss(nn.Module):
    def __init__(self, pull_weight=1, push_weight=1, off_weight=1, dcn_off_weight=1.0, focal_loss=_focal_loss):
        super(CornerNet_Loss, self).__init__()

        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.off_weight  = off_weight
        self.dcn_off_weight = dcn_off_weight
        self.focal_loss  = focal_loss
        self.ae_loss     = _ae_loss
        self.off_loss    = _off_loss
        self.dcn_off_loss = _dcn_off_loss

    def forward(self, outs, targets):
        tl_heats = outs[0]
        br_heats = outs[1]
        tl_tags  = outs[2]
        br_tags  = outs[3]
        tl_offs  = outs[4]
        br_offs  = outs[5]
        tl_offsets = br_offsets=None
        if len(outs) >6:
            # for dcn
            tl_offsets = outs[6]
            br_offsets = outs[7]

        gt_tl_heat  = targets[0]
        gt_br_heat  = targets[1]
        gt_mask     = targets[2]
        gt_tl_off   = targets[3]
        gt_br_off   = targets[4]
        gt_tl_ind   = targets[5]
        gt_br_ind   = targets[6]
        hw_objs     = targets[7]

        # focal loss
        focal_loss = 0

        tl_heats = [_sigmoid(t) for t in tl_heats]
        br_heats = [_sigmoid(b) for b in br_heats]

        focal_loss += self.focal_loss(tl_heats, gt_tl_heat)
        focal_loss += self.focal_loss(br_heats, gt_br_heat)

        # tag loss
        pull_loss = 0
        push_loss = 0
        tl_tags   = [_tranpose_and_gather_feat(tl_tag, gt_tl_ind) for tl_tag in tl_tags]
        br_tags   = [_tranpose_and_gather_feat(br_tag, gt_br_ind) for br_tag in br_tags]
        for tl_tag, br_tag in zip(tl_tags, br_tags):
            pull, push = self.ae_loss(tl_tag, br_tag, gt_mask)
            pull_loss += pull
            push_loss += push
        pull_loss = self.pull_weight * pull_loss
        push_loss = self.push_weight * push_loss

        off_loss = 0
        tl_offs  = [_tranpose_and_gather_feat(tl_off, gt_tl_ind) for tl_off in tl_offs]
        br_offs  = [_tranpose_and_gather_feat(br_off, gt_br_ind) for br_off in br_offs]
        for tl_off, br_off in zip(tl_offs, br_offs):
            off_loss += self.off_loss(tl_off, gt_tl_off, gt_mask)
            off_loss += self.off_loss(br_off, gt_br_off, gt_mask)
        off_loss = self.off_weight * off_loss

        # dcn_offsets loss
        if tl_offsets is not None and br_offsets is not None:
            dcn_offset_loss = 0
            tl_offsets = [_tranpose_and_gather_feat(tl_offset, gt_tl_ind) for tl_offset in tl_offsets]
            br_offsets = [_tranpose_and_gather_feat(br_offset, gt_br_ind) for br_offset in br_offsets]

            for tl_offset, br_offset in zip(tl_offsets, br_offsets):
                dcn_offset_loss += self.dcn_off_loss(tl_offset, hw_objs, gt_mask, mode='tl')
                dcn_offset_loss += self.dcn_off_loss(br_offset, hw_objs, gt_mask, mode='br')

            dcn_offset_loss = self.dcn_off_weight * dcn_offset_loss

        loss = (focal_loss + pull_loss + push_loss + off_loss + dcn_offset_loss) / max(len(tl_heats), 1)
        return loss.unsqueeze(0)



