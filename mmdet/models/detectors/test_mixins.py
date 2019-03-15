from mmdet.core import (bbox2roi, bbox_mapping, merge_aug_proposals,
                        merge_aug_bboxes, merge_aug_masks, multiclass_nms)


class RPNTestMixin(object):

    def simple_test_rpn(self, x, img_meta, rpn_test_cfg):
        rpn_outs = self.rpn_head(x)
        proposal_inputs = rpn_outs + (img_meta, rpn_test_cfg)
        proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        return proposal_list

    def aug_test_rpn(self, feats, img_metas, rpn_test_cfg):
        imgs_per_gpu = len(img_metas[0])
        aug_proposals = [[] for _ in range(imgs_per_gpu)]
        for x, img_meta in zip(feats, img_metas):
            proposal_list = self.simple_test_rpn(x, img_meta, rpn_test_cfg)
            for i, proposals in enumerate(proposal_list):
                aug_proposals[i].append(proposals)
        # after merging, proposals will be rescaled to the original image size
        merged_proposals = [
            merge_aug_proposals(proposals, img_meta, rpn_test_cfg)
            for proposals, img_meta in zip(aug_proposals, img_metas)
        ]
        return merged_proposals


class BBoxTestMixin(object):

    def simple_test_bboxes(self,
                           x,
                           img_meta,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
        roi_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        cls_score, bbox_pred = self.bbox_head(roi_feats)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        return det_bboxes, det_labels

    def simple_test_bboxes_for_sfa_with_orig(self,
                           x_sfa,
                           x_orig,
                           img_meta_sfa,
                           img_meta_orig,
                           proposals_sfa,
                           proposals_orig,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois_sfa = bbox2roi(proposals_sfa)
        rois_orig = bbox2roi(proposals_orig)
        roi_feats_sfa = self.bbox_roi_extractor(
            x_sfa[:len(self.bbox_roi_extractor.featmap_strides)], rois_sfa)
        roi_feats_orig = self.bbox_roi_extractor(
            x_orig[:len(self.bbox_roi_extractor.featmap_strides)], rois_orig)
        cls_score_sfa, bbox_pred_sfa = self.bbox_head(roi_feats_sfa)
        cls_score_orig, bbox_pred_orig = self.bbox_head(roi_feats_orig)
        img_shape_sfa = img_meta_sfa[0]['img_shape']
        scale_factor_sfa = img_meta_sfa[0]['scale_factor']
        img_shape_orig = img_meta_orig[0]['img_shape']
        scale_factor_orig = img_meta_orig[0]['scale_factor']
        det_bboxes, det_labels = self.bbox_head.get_det_bboxes_for_sfa_with_orig(
            rois_sfa,
            rois_orig,
            cls_score_sfa,
            bbox_pred_sfa,
            cls_score_orig,
            bbox_pred_orig,
            img_shape_sfa,
            scale_factor_sfa,
            img_shape_orig,
            scale_factor_orig,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        return det_bboxes, det_labels

    def aug_test_bboxes(self, feats, img_metas, proposal_list, rcnn_test_cfg):
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            # TODO more flexible
            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip)
            rois = bbox2roi([proposals])
            # recompute feature maps to save GPU memory
            roi_feats = self.bbox_roi_extractor(
                x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
            cls_score, bbox_pred = self.bbox_head(roi_feats)
            bboxes, scores = self.bbox_head.get_det_bboxes(
                rois,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)
        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(
            merged_bboxes, merged_scores, rcnn_test_cfg.score_thr,
            rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img)
        return det_bboxes, det_labels


class MaskTestMixin(object):

    def simple_test_mask(self,
                         x,
                         img_meta,
                         det_bboxes,
                         det_labels,
                         rescale=False,
                         is_orig=False):
        # image shape of the first image in the batch (only one)
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            _bboxes = (det_bboxes[:, :4] * scale_factor
                       if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            mask_feats = self.mask_roi_extractor(
                x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
            mask_pred = self.mask_head(mask_feats)
            segm_result = self.mask_head.get_seg_masks(
                mask_pred, _bboxes, det_labels, self.test_cfg.rcnn, ori_shape,
                scale_factor, rescale, is_orig=is_orig)
        return segm_result

    def simple_test_mask_for_sfa_with_orig(self,
                         x_sfa,
                         x_orig,
                         img_meta_sfa,
                         img_meta_orig,
                         det_bboxes,
                         det_labels,
                         rescale=False,
                         out_flag=0):
        # image shape of the first image in the batch (only one)
        ori_shape_sfa = img_meta_sfa[0]['ori_shape']
        scale_factor_sfa = img_meta_sfa[0]['scale_factor']
        ori_shape_orig = img_meta_orig[0]['ori_shape']
        scale_factor_orig = img_meta_orig[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            # for sfa
            _bboxes_sfa = (det_bboxes[:, :4] * scale_factor_sfa
                       if rescale else det_bboxes)
            mask_rois_sfa = bbox2roi([_bboxes_sfa])
            mask_feats_sfa = self.mask_roi_extractor(
                x_sfa[:len(self.mask_roi_extractor.featmap_strides)], mask_rois_sfa)
            mask_pred_sfa = self.mask_head(mask_feats_sfa)

            # for orig
            _bboxes_orig = (det_bboxes[:, :4] * scale_factor_orig
                           if rescale else det_bboxes[:, :4] /2.)
            mask_rois_orig = bbox2roi([_bboxes_orig])
            mask_feats_orig = self.mask_roi_extractor(
                x_orig[:len(self.mask_roi_extractor.featmap_strides)], mask_rois_orig)
            mask_pred_orig = self.mask_head(mask_feats_orig)

            segm_result_sfa = self.mask_head.get_seg_masks(
                mask_pred_sfa, _bboxes_sfa, det_labels, self.test_cfg.rcnn, ori_shape_sfa,
                scale_factor_sfa, rescale)
            segm_result_orig = self.mask_head.get_seg_masks(
                mask_pred_orig, _bboxes_orig, det_labels, self.test_cfg.rcnn, ori_shape_orig,
                scale_factor_orig, rescale, is_orig=True)
            if out_flag == 0:
                segm_result = []
                for i in range(len(segm_result_sfa)):
                    segm_result.append(segm_result_sfa[i] + segm_result_orig[i])
                return segm_result
            elif out_flag == 1:
                return segm_result_orig
            elif out_flag == 2:
                return segm_result_sfa
            else:
                raise Exception('Not right out_flag')
        return segm_result


    def aug_test_mask(self, feats, img_metas, det_bboxes, det_labels):
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
        else:
            aug_masks = []
            for x, img_meta in zip(feats, img_metas):
                img_shape = img_meta[0]['img_shape']
                scale_factor = img_meta[0]['scale_factor']
                flip = img_meta[0]['flip']
                _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                       scale_factor, flip)
                mask_rois = bbox2roi([_bboxes])
                mask_feats = self.mask_roi_extractor(
                    x[:len(self.mask_roi_extractor.featmap_strides)],
                    mask_rois)
                mask_pred = self.mask_head(mask_feats)
                # convert to numpy array to save memory
                aug_masks.append(mask_pred.sigmoid().cpu().numpy())
            merged_masks = merge_aug_masks(aug_masks, img_metas,
                                           self.test_cfg.rcnn)

            ori_shape = img_metas[0][0]['ori_shape']
            segm_result = self.mask_head.get_seg_masks(
                merged_masks,
                det_bboxes,
                det_labels,
                self.test_cfg.rcnn,
                ori_shape,
                scale_factor=1.0,
                rescale=False)
        return segm_result
