import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseDetector
from .test_mixins import RPNTestMixin, BBoxTestMixin, MaskTestMixin
from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2roi, bbox2result, build_assigner, build_sampler
import numpy as np
import copy
@DETECTORS.register_module
class TwoStageDetector(BaseDetector, RPNTestMixin, BBoxTestMixin,
                       MaskTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwoStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)
        else:
            raise NotImplementedError
        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)

        if mask_head is not None:
            self.mask_roi_extractor = builder.build_roi_extractor(
                mask_roi_extractor)
            self.mask_head = builder.build_head(mask_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(TwoStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_roi_extractor.init_weights()
            self.mask_head.init_weights()

    def extract_certain_feat(self,img, stage=1):
        x = self.backbone(img, stage=stage)
        return x
    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_bboxes_ignore,
                      gt_labels,
                      gt_masks=None,
                      proposals=None):
        if hasattr(self.neck,'with_sfa'):
            down_img_h, down_img_w = img.size(2)//2, img.size(3)//2
            down_img_h = int(np.ceil(down_img_h / 32) * 32)
            down_img_w = int(np.ceil(down_img_w / 32) * 32)
            down_img = F.interpolate(img, size=(down_img_h, down_img_w) ,mode='bilinear', align_corners=True)
            img_meta_orig = self.down_img_meta(img_meta)
            gt_bboxes_orig, gt_masks_orig = self.down_gt_bboxes_masks(gt_bboxes, gt_masks)
            x = self.extract_feat(down_img)
            if self.neck.with_sfa_loss:
                x_stage = self.extract_certain_feat(img, stage=1)

        else:
            x = self.extract_feat(img)
        losses = dict()



        # RPN forward and loss
        if self.with_rpn:
            if hasattr(self.neck, 'with_sfa') and self.neck.with_orig:
                rpn_outs_orig = self.rpn_head(x[1])
                rpn_loss_inputs_orig = rpn_outs_orig + (gt_bboxes_orig, img_meta_orig,
                                              self.train_cfg.rpn)
                rpn_losses_orig = self.rpn_head.loss(*rpn_loss_inputs_orig, scale='orig',
                                                     orig_w= self.neck.orig_loss_weight)
                losses.update(rpn_losses_orig)

                proposal_inputs_orig = rpn_outs_orig + (img_meta_orig, self.test_cfg.rpn)
                proposal_list_orig = self.rpn_head.get_bboxes(*proposal_inputs_orig)

            rpn_outs_sfa = self.rpn_head(x[0])
            rpn_loss_inputs_sfa = rpn_outs_sfa + (gt_bboxes, img_meta,
                                                  self.train_cfg.rpn)
            rpn_losses_sfa = self.rpn_head.loss(*rpn_loss_inputs_sfa, scale='sfa')
            losses.update(rpn_losses_sfa)

            proposal_inputs_sfa = rpn_outs_sfa + (img_meta, self.test_cfg.rpn)
            proposal_list_sfa = self.rpn_head.get_bboxes(*proposal_inputs_sfa)

        else:
            proposal_list = proposals


        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            sampling_results_orig = []
            sampling_results_sfa = []
            for i in range(num_imgs):
                if hasattr(self.neck, 'with_sfa') and self.neck.with_orig:
                    assign_result_orig = bbox_assigner.assign(
                        proposal_list_orig[i], gt_bboxes_orig[i], gt_bboxes_ignore[i],
                        gt_labels[i])
                    sampling_result_orig = bbox_sampler.sample(
                        assign_result_orig,
                        proposal_list_orig[i],
                        gt_bboxes_orig[i],
                        gt_labels[i],
                        feats=[lvl_feat[i][None] for lvl_feat in x[1]])
                    sampling_results_orig.append(sampling_result_orig)

                assign_result_sfa = bbox_assigner.assign(
                    proposal_list_sfa[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result_sfa = bbox_sampler.sample(
                    assign_result_sfa,
                    proposal_list_sfa[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x[0]])
                sampling_results_sfa.append(sampling_result_sfa)

        # bbox head forward and loss
        if self.with_bbox:
            if hasattr(self.neck, 'with_sfa') and self.neck.with_orig:
                rois_orig = bbox2roi([res.bboxes for res in sampling_results_orig])
                # TODO: a more flexible way to decide which feature maps to use
                bbox_feats = self.bbox_roi_extractor(
                    x[1][:self.bbox_roi_extractor.num_inputs], rois_orig)
                cls_score, bbox_pred = self.bbox_head(bbox_feats)

                bbox_targets = self.bbox_head.get_target(
                    sampling_results_orig, gt_bboxes_orig, gt_labels, self.train_cfg.rcnn)
                loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                                *bbox_targets, scale='orig', orig_w = self.neck.orig_loss_weight)
                losses.update(loss_bbox)
            # for sfa
            rois_sfa = bbox2roi([res.bboxes for res in sampling_results_sfa])
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[0][:self.bbox_roi_extractor.num_inputs], rois_sfa)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_targets = self.bbox_head.get_target(
                sampling_results_sfa, gt_bboxes, gt_labels, self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets, scale='sfa')
            losses.update(loss_bbox)

        if hasattr(self.neck, 'with_sfa') and self.neck.with_sfa_loss:
            if self.neck.with_rpn_clip:
                loss_sfa = self.neck.loss(x[0][0], x_stage, stage=1, proposal=rois_sfa)
            else:
                loss_sfa = self.neck.loss(x[0][0], x_stage, stage=1)
            losses.update(loss_sfa)
        # mask head forward and loss
        if self.with_mask:
            if hasattr(self.neck, 'with_sfa') and self.neck.with_orig:
                pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results_orig])
                mask_feats = self.mask_roi_extractor(
                    x[1][:self.mask_roi_extractor.num_inputs], pos_rois)
                mask_pred = self.mask_head(mask_feats)

                mask_targets = self.mask_head.get_target(
                    sampling_results_orig, gt_masks_orig, self.train_cfg.rcnn)
                pos_labels = torch.cat(
                    [res.pos_gt_labels for res in sampling_results_orig])
                loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                                pos_labels, scale='orig')
                losses.update(loss_mask)
            # for sfa
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results_sfa])
            mask_feats = self.mask_roi_extractor(
                x[0][:self.mask_roi_extractor.num_inputs], pos_rois)
            mask_pred = self.mask_head(mask_feats)

            mask_targets = self.mask_head.get_target(
                sampling_results_sfa, gt_masks, self.train_cfg.rcnn)
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results_sfa])
            loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            pos_labels, scale='sfa')
            losses.update(loss_mask)

        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."
        if hasattr(self.neck, 'with_sfa'):
            down_img_h, down_img_w = img.size(2) // 2, img.size(3) // 2
            down_img_h = int(np.ceil(down_img_h / 32) * 32)
            down_img_w = int(np.ceil(down_img_w / 32) * 32)
            down_img = F.interpolate(img, size=(down_img_h, down_img_w), mode='bilinear', align_corners=True)
            img_meta_orig = self.down_img_meta(img_meta)
            x = self.extract_feat(down_img)
        else:
            x = self.extract_feat(img)

        if hasattr(self.neck, 'with_sfa') and self.neck.with_orig:
            if self.neck.only_sfa_result:
                proposal_list_sfa = self.simple_test_rpn(
                    x[0], img_meta, self.test_cfg.rpn) if proposals is None else proposals
                det_bboxes_sfa, det_labels_sfa = self.simple_test_bboxes(
                    x[0], img_meta, proposal_list_sfa, self.test_cfg.rcnn, rescale=rescale)
                bbox_results_sfa = bbox2result(det_bboxes_sfa, det_labels_sfa,
                                               self.bbox_head.num_classes)
                if not self.with_mask:
                    return bbox_results_sfa

                else:
                    segm_results_sfa = self.simple_test_mask(
                        x[0], img_meta, det_bboxes_sfa, det_labels_sfa, rescale=rescale)

                    return bbox_results_sfa, segm_results_sfa
            elif self.neck.only_orig_result:
                proposal_list_orig = self.simple_test_rpn(
                    x[1], img_meta_orig, self.test_cfg.rpn) if proposals is None else proposals
                det_bboxes_orig, det_labels_orig = self.simple_test_bboxes(
                    x[1], img_meta_orig, proposal_list_orig, self.test_cfg.rcnn, rescale=rescale)
                bbox_results_orig = bbox2result(det_bboxes_orig, det_labels_orig,
                                                self.bbox_head.num_classes)

                if not self.with_mask:
                    return bbox_results_orig
                else:
                    segm_results_orig = self.simple_test_mask(
                        x[1], img_meta_orig, det_bboxes_orig, det_labels_orig, rescale=rescale)

                    return bbox_results_orig, segm_results_orig
            else:
                proposal_list_sfa = self.simple_test_rpn(
                    x[0], img_meta, self.test_cfg.rpn) if proposals is None else proposals

                proposal_list_orig = self.simple_test_rpn(
                    x[1], img_meta_orig, self.test_cfg.rpn) if proposals is None else proposals

                det_bboxes, det_labels= self.simple_test_bboxes_for_sfa_with_orig(
                    x[0], x[1], img_meta, img_meta_orig, proposal_list_sfa, proposal_list_orig,
                    self.test_cfg.rcnn, rescale=rescale)
                bbox_results = bbox2result(det_bboxes, det_labels,
                                               self.bbox_head.num_classes)
                if not self.with_mask:
                    return bbox_results
                else:
                    segm_results = self.simple_test_mask_for_sfa_with_orig(
                        x[0], x[1], img_meta, img_meta_orig, det_bboxes, det_labels, rescale=rescale,
                        out_flag=self.neck.segm_out_flag)
                    return bbox_results, segm_results

        else:
            if hasattr(self.neck, 'with_sfa'):
                proposal_list_sfa = self.simple_test_rpn(
                    x[0], img_meta, self.test_cfg.rpn) if proposals is None else proposals
                det_bboxes_sfa, det_labels_sfa = self.simple_test_bboxes(
                    x[0], img_meta, proposal_list_sfa, self.test_cfg.rcnn, rescale=rescale)
                bbox_results_sfa = bbox2result(det_bboxes_sfa, det_labels_sfa,
                                               self.bbox_head.num_classes)

                if not self.with_mask:
                    return bbox_results_sfa

                else:
                    segm_results_sfa = self.simple_test_mask(
                        x[0], img_meta, det_bboxes_sfa, det_labels_sfa, rescale=rescale)

                    return bbox_results_sfa, segm_results_sfa
            else:
                proposal_list_sfa = self.simple_test_rpn(
                    x, img_meta, self.test_cfg.rpn) if proposals is None else proposals
                det_bboxes_sfa, det_labels_sfa = self.simple_test_bboxes(
                    x, img_meta, proposal_list_sfa, self.test_cfg.rcnn, rescale=rescale)
                bbox_results_sfa = bbox2result(det_bboxes_sfa, det_labels_sfa,
                                               self.bbox_head.num_classes)

                if not self.with_mask:
                    return bbox_results_sfa

                else:
                    segm_results_sfa = self.simple_test_mask(
                        x, img_meta, det_bboxes_sfa, det_labels_sfa, rescale=rescale)

                    return bbox_results_sfa, segm_results_sfa

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(imgs), img_metas, proposal_list,
            self.test_cfg.rcnn)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(
                self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results

    def down_img_meta(self, img_meta):
        key_word = ['img_shape','pad_shape']
        img_meta_down = copy.deepcopy(img_meta)
        for i in range(len(img_meta)):
            for key in key_word:
                orig_key_v = list(img_meta[i][key])
                for j in range(len(orig_key_v)-1):
                    orig_key_v[j] = orig_key_v[j] //2
                img_meta_down[i][key] = tuple(orig_key_v)
            img_meta_down[i]['scale_factor'] = img_meta[i]['scale_factor']/2
        return img_meta_down

    def down_gt_bboxes_masks(self,gt_bboxes, gt_masks):
        gt_bboxes_orig =[]
        gt_masks_orig=[]
        num_imgs = len(gt_bboxes)
        for i in range(num_imgs):
            gt_bboxes_orig.append(gt_bboxes[i]/2)
            gt_masks_orig.append(gt_masks[i][:,0::2,0::2])
        return gt_bboxes_orig, gt_masks_orig


