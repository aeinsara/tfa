import logging

import torch
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from torch import nn
from PIL import Image
import numpy as np
import cv2
from fsdet.modeling.roi_heads import build_roi_heads

# avoid conflicting with the existing GeneralizedRCNN module in Detectron2
from .build import META_ARCH_REGISTRY

__all__ = ["GeneralizedRCNN", "ProposalNetwork"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(
            cfg, self.backbone.output_shape()
        )
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = (
            torch.Tensor(cfg.MODEL.PIXEL_MEAN)
            .to(self.device)
            .view(num_channels, 1, 1)
        )
        pixel_std = (
            torch.Tensor(cfg.MODEL.PIXEL_STD)
            .to(self.device)
            .view(num_channels, 1, 1)
        )
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("froze backbone parameters")

        if cfg.MODEL.PROPOSAL_GENERATOR.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print("froze proposal generator parameters")

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.box_head.parameters():
                p.requires_grad = False
            print("froze roi_box_head parameters")

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores"
        """
        
        if "instances" in batched_inputs[0]:
            gt_instances = [
                x["instances"].to(self.device) for x in batched_inputs
            ]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN,
                "'targets' in the model inputs is now renamed to 'instances'!",
                n=10,
            )
            gt_instances = [
                x["targets"].to(self.device) for x in batched_inputs
            ]
        else:
            gt_instances = None
        if not self.training:
            return self.inference(batched_inputs,gt_instances)
        images = self.preprocess_image(batched_inputs,gt_instances)
        

        features = self.backbone(images.tensor)

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [
                x["proposals"].to(self.device) for x in batched_inputs
            ]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(
            images, features, proposals, gt_instances
        )

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(
        self, batched_inputs, gt_instances, detected_instances=None, do_postprocess=True
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs,gt_instances)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [
                    x["proposals"].to(self.device) for x in batched_inputs
                ]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [
                x.to(self.device) for x in detected_instances
            ]
            results = self.roi_heads.forward_with_given_boxes(
                features, detected_instances
            )

        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
        else:
            return results

    def preprocess_image(self, batched_inputs, gt_instances):
#         print("gt_instances == ",  gt_instances)
#         for i,img in enumerate(batched_inputs):
#             image = img["image"]
#             if gt_instances is None or len(gt_instances)==0 or len(gt_instances[i].gt_boxes.tensor.data)==0:
#                 continue
#             x1,y1,x2,y2 = gt_instances[i].gt_boxes.tensor.data[0].tolist()
#             image = np.transpose(batched_inputs[i]['image'].cpu().detach().numpy(), (1, 2, 0))
#             image = self.crop(image, int(x1), int(y1), int(x2-x1), int(y2-y1), 1)
# #             images.append(torch.from_numpy(image.transpose((2, 0, 1))).to(self.device))
#             batched_inputs[i]['image'] = torch.from_numpy(image.transpose((2, 0, 1)))
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(
            images, self.backbone.size_divisibility
        )
        return images
    def crop(self, image, x1, y1, w, h, scale):
        # create a mask
        mask = np.zeros(image.shape[:2], np.uint8)
        if scale == 1:
            new_x1 = x1
            new_y1 = y1
            new_x2 = x1+w
            new_y2 = y1+h
        else:
            new_x1 = int(max(x1 - ((scale / 2) * w), 0))
            new_y1 = int(max(y1 - ((scale / 2) * h), 0))
            new_x2 = int(min((x1 + (scale / 2) * w + w), image.shape[1]))
            new_y2 = int(min((y1 + (scale / 2) * h + h), image.shape[0]))
        
        # ............ For useing mask image ..............
        mask[new_y1:new_y2, new_x1:new_x2] = 255

        # compute the bitwise AND using the mask
        masked_img = cv2.bitwise_and(image,image,mask = mask)
        return masked_img


@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(
            cfg, self.backbone.output_shape()
        )

        pixel_mean = (
            torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(-1, 1, 1)
        )
        pixel_std = (
            torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(-1, 1, 1)
        )
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]: Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        if "instances" in batched_inputs[0]:
            gt_instances = [
                x["instances"].to(self.device) for x in batched_inputs
            ]
#         for i,img in enumerate(batched_inputs):
#             image = img["image"]
#             if gt_instances is None:
#                 continue
#             x1,y1,x2,y2 = gt_instances[i].gt_boxes.tensor.data[0].tolist()
#             image = np.transpose(batched_inputs[i]['image'].cpu().detach().numpy(), (1, 2, 0))
#             image = self.crop(image, int(x1), int(y1), int(x2-x1), int(y2-y1), 1)
# #             images.append(torch.from_numpy(image.transpose((2, 0, 1))).to(self.device))
#             batched_inputs[i]['image'] = torch.from_numpy(image.transpose((2, 0, 1)))
        
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(
            images, self.backbone.size_divisibility
        )
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [
                x["instances"].to(self.device) for x in batched_inputs
            ]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN,
                "'targets' in the model inputs is now renamed to 'instances'!",
                n=10,
            )
            gt_instances = [
                x["targets"].to(self.device) for x in batched_inputs
            ]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances
        )
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results
    
    def crop(self, image, x1, y1, w, h, scale):
        # create a mask
        mask = np.zeros(image.shape[:2], np.uint8)
        if scale == 1:
            new_x1 = x1
            new_y1 = y1
            new_x2 = x1+w
            new_y2 = y1+h
        else:
            new_x1 = int(max(x1 - ((scale / 2) * w), 0))
            new_y1 = int(max(y1 - ((scale / 2) * h), 0))
            new_x2 = int(min((x1 + (scale / 2) * w + w), image.shape[1]))
            new_y2 = int(min((y1 + (scale / 2) * h + h), image.shape[0]))
        
        # ............ For useing mask image ..............
        mask[new_y1:new_y2, new_x1:new_x2] = 255

        # compute the bitwise AND using the mask
        masked_img = cv2.bitwise_and(image,image,mask = mask)
        return masked_img