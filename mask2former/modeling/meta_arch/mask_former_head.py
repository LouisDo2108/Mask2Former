# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer_decoder.maskformer_transformer_decoder import build_transformer_decoder
from ..pixel_decoder.fpn import build_pixel_decoder
import copy
def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


@SEM_SEG_HEADS_REGISTRY.register()
class MaskFormerHead(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "sem_seg_head" in k and not k.startswith(prefix + "predictor"):
                    newk = k.replace(prefix, prefix + "pixel_decoder.")
                    # logger.debug(f"{k} ==> {newk}")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        # extra parameters
        transformer_predictor: nn.Module,
        transformer_in_feature: str,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor
        self.transformer_in_feature = transformer_in_feature

        self.num_classes = num_classes

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        # figure out in_channels to transformer predictor
        if cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "transformer_encoder":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        elif cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "pixel_embedding":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        elif cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "multi_scale_pixel_decoder":  # for maskformer2
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        else:
            transformer_predictor_in_channels = input_shape[cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE].channels

        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "pixel_decoder": build_pixel_decoder(cfg, input_shape),
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "transformer_in_feature": cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE,
            "transformer_predictor": build_transformer_decoder(
                cfg,
                transformer_predictor_in_channels,
                mask_classification=True,
            ),
        }

    def forward(self, features, mask=None):
        return self.layers(features, mask)

    def layers(self, features, mask=None):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)
        if self.transformer_in_feature == "multi_scale_pixel_decoder": # Default
            predictions = self.predictor(multi_scale_features, mask_features, mask)
        else:
            if self.transformer_in_feature == "transformer_encoder":
                assert (
                    transformer_encoder_features is not None
                ), "Please use the TransformerEncoderPixelDecoder."
                predictions = self.predictor(transformer_encoder_features, mask_features, mask)
            elif self.transformer_in_feature == "pixel_embedding":
                predictions = self.predictor(mask_features, mask_features, mask)
            else:
                predictions = self.predictor(features[self.transformer_in_feature], mask_features, mask)
        return predictions


@SEM_SEG_HEADS_REGISTRY.register()
class MaskFormerDualDecoderHeadWithZissWeighting(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "sem_seg_head" in k and not k.startswith(prefix + "predictor"):
                    newk = k.replace(prefix, prefix + "pixel_decoder.")
                    # logger.debug(f"{k} ==> {newk}")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        # extra parameters
        transformer_predictor_easy: nn.Module,
        transformer_predictor_hard: nn.Module,
        transformer_in_feature: str,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        self.pixel_decoder = pixel_decoder
        self.predictor_easy = transformer_predictor_easy
        self.predictor_hard = transformer_predictor_hard
        self.transformer_in_feature = transformer_in_feature

        self.num_classes = num_classes

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        # figure out in_channels to transformer predictor
        if cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "transformer_encoder":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        elif cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "pixel_embedding":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        elif cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "multi_scale_pixel_decoder":  # for maskformer2
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        else:
            transformer_predictor_in_channels = input_shape[cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE].channels

        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "pixel_decoder": build_pixel_decoder(cfg, input_shape),
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "transformer_in_feature": cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE,
            "transformer_predictor_easy": build_transformer_decoder(
                cfg,
                transformer_predictor_in_channels,
                mask_classification=True,
            ),
            "transformer_predictor_hard": build_transformer_decoder(
                cfg,
                transformer_predictor_in_channels,
                mask_classification=True,
            ),
        }

    def forward(self, features, mask=None, hardness_weight=None):
        return self.layers(features, mask, hardness_weight)

    def layers(self, features, mask=None, hardness_weight=None):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)
        
        # If hardness_weight is None, raise error, 
        # since there is no hardness_pred
        if hardness_weight is None:
            raise ValueError("hardness_weight is None")
        
        predictions = {}
        if self.transformer_in_feature == "multi_scale_pixel_decoder": # Default
            predictions_easy = self.predictor_easy(
                multi_scale_features, mask_features, mask,
                hardness_weight=(1.0-hardness_weight),
            )
            predictions_hard = self.predictor_hard(
                multi_scale_features, mask_features, mask,
                hardness_weight=hardness_weight,
            )
            
            # Summing easy and hard predictions
            for k in predictions_easy.keys():
                if k != 'aux_outputs':
                    predictions[k] = predictions_easy[k] + predictions_hard[k]
                else:
                    predictions['aux_outputs'] = []
                    for aux_easy, aux_hard in zip(predictions_easy[k], predictions_hard[k]):
                        predictions['aux_outputs'].append({
                            "pred_logits": aux_easy["pred_logits"] + aux_hard["pred_logits"],
                            "pred_masks": aux_easy["pred_masks"] + aux_hard["pred_masks"],
                        })
        else:
            if self.transformer_in_feature == "transformer_encoder":
                assert (
                    transformer_encoder_features is not None
                ), "Please use the TransformerEncoderPixelDecoder."
                predictions = self.predictor(transformer_encoder_features, mask_features, mask)
            elif self.transformer_in_feature == "pixel_embedding":
                predictions = self.predictor(mask_features, mask_features, mask)
            else:
                predictions = self.predictor(features[self.transformer_in_feature], mask_features, mask)
        return predictions
    

@SEM_SEG_HEADS_REGISTRY.register()
class MaskFormerDualDecoderHeadWithZissRanking(MaskFormerDualDecoderHeadWithZissWeighting):
    def forward(self, features, mask=None, hardness_weight=None):
        return self.layers(features, mask, hardness_weight)

    def layers(self, features, mask=None, hardness_weight=None):
        mask_features, transformer_encoder_features, multi_scale_features, hardness_pred = self.pixel_decoder.forward_features(features)
        
        # If hardness_weight is None, use the hardness_pred instead
        if hardness_weight is None:
            hardness_weight = hardness_pred.sigmoid().squeeze(-1) # Same shape as the gt: tensor size of (2)
        
        predictions = {}
        if self.transformer_in_feature == "multi_scale_pixel_decoder": # Default
            predictions_easy = self.predictor_easy(
                multi_scale_features, mask_features, mask,
                hardness_weight=(1.0-hardness_weight),
            )
            predictions_hard = self.predictor_hard(
                multi_scale_features, mask_features, mask,
                hardness_weight=hardness_weight,
            )
            
            # Summing easy and hard predictions
            for k in predictions_easy.keys():
                if k != 'aux_outputs':
                    predictions[k] = predictions_easy[k] + predictions_hard[k]
                else:
                    predictions['aux_outputs'] = []
                    for aux_easy, aux_hard in zip(predictions_easy[k], predictions_hard[k]):
                        predictions['aux_outputs'].append({
                            "pred_logits": aux_easy["pred_logits"] + aux_hard["pred_logits"],
                            "pred_masks": aux_easy["pred_masks"] + aux_hard["pred_masks"],
                        })
            del predictions_easy, predictions_hard
        else:
            if self.transformer_in_feature == "transformer_encoder":
                assert (
                    transformer_encoder_features is not None
                ), "Please use the TransformerEncoderPixelDecoder."
                predictions = self.predictor(transformer_encoder_features, mask_features, mask)
            elif self.transformer_in_feature == "pixel_embedding":
                predictions = self.predictor(mask_features, mask_features, mask)
            else:
                predictions = self.predictor(features[self.transformer_in_feature], mask_features, mask)
        return predictions, hardness_pred
