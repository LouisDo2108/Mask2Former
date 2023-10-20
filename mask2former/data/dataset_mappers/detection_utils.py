import logging

import numpy as np
import torch

from detectron2.data import transforms as T
from detectron2.data.detection_utils import \
    annotations_to_instances as d2_anno_to_inst
from detectron2.data.detection_utils import \
    transform_instance_annotations as d2_transform_inst_anno


def transform_instance_annotations(
    annotation, transforms, image_size, *, keypoint_hflip_indices=None
):
    s_alpha = None
    if "s_alpha" in annotation:
        s_alpha = annotation["s_alpha"]
        
    srf = None
    if "srf" in annotation:
        s_alpha = annotation["srf"]
        
    sb = None
    if "sb" in annotation:
        s_alpha = annotation["sb"]
        
    f_ratio = None
    if "f_ratio" in annotation:
        s_alpha = annotation["f_ratio"]
        
    annotation = d2_transform_inst_anno(
        annotation,
        transforms,
        image_size,
        keypoint_hflip_indices=keypoint_hflip_indices,
    )

    if "beziers" in annotation:
        beziers = transform_beziers_annotations(annotation["beziers"], transforms)
        annotation["beziers"] = beziers
    
    if s_alpha:
        annotation["s_alpha"] = s_alpha
        
    if srf:
        annotation["srf"] = srf
        
    if sb:
        annotation["sb"] = sb
        
    if f_ratio:
        annotation["f_ratio"] = f_ratio    
    
    return annotation


def transform_beziers_annotations(beziers, transforms):
    """
    Transform keypoint annotations of an image.

    Args:
        beziers (list[float]): Nx16 float in Detectron2 Dataset format.
        transforms (TransformList):
    """
    # (N*2,) -> (N, 2)
    beziers = np.asarray(beziers, dtype="float64").reshape(-1, 2)
    beziers = transforms.apply_coords(beziers).reshape(-1)

    # This assumes that HorizFlipTransform is the only one that does flip
    do_hflip = (
        sum(isinstance(t, T.HFlipTransform) for t in transforms.transforms) % 2 == 1
    )
    if do_hflip:
        raise ValueError("Flipping text data is not supported (also disencouraged).")

    return beziers


def annotations_to_instances(annos, image_size, mask_format="polygon"):
    instance = d2_anno_to_inst(annos, image_size, mask_format)

    if not annos:
        return instance

    # add attributes
    if "beziers" in annos[0]:
        beziers = [obj.get("beziers", []) for obj in annos]
        instance.beziers = torch.as_tensor(beziers, dtype=torch.float32)

    if "rec" in annos[0]:
        text = [obj.get("rec", []) for obj in annos]
        instance.text = torch.as_tensor(text, dtype=torch.int32)
        
    if "s_alpha" in annos[0]:
        s_alpha = [obj.get("s_alpha", []) for obj in annos]
        instance.s_alpha = torch.as_tensor(s_alpha, dtype=torch.float32)
        
    if "srf" in annos[0]:
        srf = [obj.get("srf", []) for obj in annos]
        instance.srf = torch.as_tensor(srf, dtype=torch.float32)
        
    if "sb" in annos[0]:
        sb = [obj.get("sb", []) for obj in annos]
        instance.sb = torch.as_tensor(sb, dtype=torch.float32)
        
    if "f_ratio" in annos[0]:
        f_ratio = [obj.get("f_ratio", []) for obj in annos]
        instance.f_ratio = torch.as_tensor(f_ratio, dtype=torch.float32)

    return instance



def build_augmentation(cfg, is_train):
    """
    With option to don't use hflip

    Returns:
        list[Augmentation]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert (
            len(min_size) == 2
        ), "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)

    augmentation = []
    augmentation.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        if cfg.INPUT.HFLIP_TRAIN:
            augmentation.append(T.RandomFlip())
        logger.info("Augmentations used in training: " + str(augmentation))
    return augmentation


build_transform_gen = build_augmentation
"""
Alias for backward-compatibility.
"""
