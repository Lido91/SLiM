# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Labelled datasets for the downstream protocols: NTU .npz files or sign-pose packs.

`data_path` decides: a directory with sign packs (<data_path>/train/meta.json, e.g. WLASL-2000)
gives SignPose_Classification, anything else the original NTU_Classification.
"""

import os

from slim.data.augmentations import DownstreamAugmentation
from slim.data.datasets import NTU_Classification, SignPose_Classification, is_sign_pack
from slim.data.skeletons import get_skeleton, is_sign_skeleton


def split_from_dataset_str(dataset_str):
    """'NTU:split=VAL' -> 'val'. NTU has no validation split, so VAL and TEST both read its test set."""
    for token in (dataset_str or "").split(":")[1:]:
        key, _, value = token.partition("=")
        if key == "split":
            return value.lower()
    return "test"


def _sign_settings(data_cfg):
    if data_cfg is None or not is_sign_skeleton(data_cfg.get("skeleton")):
        raise ValueError("sign-pose packs need an eval config with crops.skeleton set (e.g. slim_sign_eval.yaml)")
    return data_cfg.skeleton, data_cfg.get("pose_norm", "part")


def build_downstream_transform(split, data_path, data_cfg=None):
    out_size = data_cfg.global_crops_size[0] if data_cfg is not None else 64
    skeleton = None
    if is_sign_pack(os.path.join(data_path, "train")):
        name, norm = _sign_settings(data_cfg)
        skeleton = get_skeleton(name, norm=norm)
    return DownstreamAugmentation(out_size=out_size, split=split, skeleton=skeleton)


def build_classification_dataset(data_path, split, num_classes, benchmark, transform, data_cfg=None, sample_ratio=1.0):
    """split: 'train', 'val' or 'test'."""
    if is_sign_pack(os.path.join(data_path, "train")):
        name, norm = _sign_settings(data_cfg)
        if split == "val":
            split = "dev" if is_sign_pack(os.path.join(data_path, "dev")) else "test"
        return SignPose_Classification(data_path, split=split, transform=transform, skeleton=name, norm=norm,
                                       sample_ratio=sample_ratio)
    return NTU_Classification(
        root_path=data_path,
        split="train" if split == "train" else "test",
        num_classes=num_classes,
        benchmark=benchmark,
        min_buffer=150,
        transform=transform,
        sample_ratio=sample_ratio,
    )
