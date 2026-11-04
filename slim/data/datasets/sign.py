# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Sign-language pose datasets (RTMPose keypoints packed by tools/pack_sign_poses.py).

A pack is a directory holding
    poses.npy    float32 (F, 69, 3) raw (x, y, score) frames of every clip, concatenated
    offsets.npy  int64 (N + 1,) clip i is poses[offsets[i]:offsets[i + 1]]
    meta.json    {"format": "slim-sign-pack-v1", "dataset", "split", "names", "labels", "classes"}
`poses.npy` is memory-mapped, so DataLoader workers share it through the page cache.
"""

import json
import logging
import os
import random
from enum import Enum

import numpy as np
import torch
from torch.utils.data import Dataset

from slim.data.skeletons import get_skeleton, normalize_sign_pose


logger = logging.getLogger("slim")

PACK_FORMAT = "slim-sign-pack-v1"


def is_sign_pack(path):
    return os.path.isfile(os.path.join(path, "meta.json"))


class SignPosePack:
    def __init__(self, root):
        with open(os.path.join(root, "meta.json")) as f:
            self.meta = json.load(f)
        if self.meta.get("format") != PACK_FORMAT:
            raise RuntimeError(f"{root} is not a {PACK_FORMAT} pack")
        self.root = root
        self.poses = np.load(os.path.join(root, "poses.npy"), mmap_mode="r")
        self.offsets = np.load(os.path.join(root, "offsets.npy"))
        self.names = self.meta["names"]
        self.labels = self.meta.get("labels")

    def __len__(self):
        return len(self.offsets) - 1

    def raw(self, i):
        return np.array(self.poses[self.offsets[i]:self.offsets[i + 1]])


def _to_clip(raw, skeleton, norm):
    """(T, 69, 3) raw -> normalised (C=3, T, V, M=1) float32 for the selected layout."""
    seq = normalize_sign_pose(raw, norm=norm)[:, skeleton.joints]
    return np.ascontiguousarray(seq.transpose(2, 0, 1)[..., None])


class SignPose(Dataset):
    """Unlabelled sign clips for pre-training. `root` is one pack dir or several joined by ','.

    Uses only what the packs hold; pack the *training* split of each dataset so evaluation
    clips are never seen during pre-training.
    """

    class Split(Enum):
        TRAIN = "train"
        TEST = "test"

    def __init__(self, root, split=None, transform=None, skeleton="sign69", norm="part", **kwargs):
        self.transform = transform
        self.skeleton = get_skeleton(skeleton, norm=norm)
        self.norm = norm
        self.packs = [SignPosePack(r) for r in root.split(",") if r]
        self.index = [(p, i) for p, pack in enumerate(self.packs) for i in range(len(pack))]
        for pack in self.packs:
            logger.info(f"SignPose: {len(pack):,d} clips from {pack.root} "
                        f"({pack.meta.get('dataset')}/{pack.meta.get('split')})")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        p, i = self.index[index]
        seq = _to_clip(self.packs[p].raw(i), self.skeleton, self.norm)

        if self.transform is not None:
            output = self.transform(seq, seq.shape[1])
        else:
            tensor_seq = torch.from_numpy(seq).float()
            output = {"global_crops": [tensor_seq, tensor_seq], "local_crops": []}
        output["target"] = ()
        return output, ()


class SignPose_Classification(Dataset):
    """Labelled sign clips (e.g. WLASL-2000) for linear probing, fine-tuning and k-NN.

    `data_path` holds one pack per split: <data_path>/{train,dev,test}. Labels index the
    `classes` list stored in the packs. Returns (clip, valid-person mask, label) like
    NTU_Classification, with a single always-valid person.
    """

    def __init__(self, data_path, split="train", transform=None, skeleton="sign69", norm="part",
                 sample_ratio=1.0, seed=42):
        self.transform = transform
        self.skeleton = get_skeleton(skeleton, norm=norm)
        self.norm = norm
        self.pack = SignPosePack(os.path.join(data_path, split))
        if self.pack.labels is None:
            raise RuntimeError(f"{self.pack.root} has no labels; pack it with a label key")
        self.label = np.asarray(self.pack.labels, dtype=np.int64)
        self.indices = np.flatnonzero(self.label >= 0)  # -1 = class absent from the train vocab
        if len(self.indices) < len(self.label):
            logger.info(f"{self.pack.root}: dropping {len(self.label) - len(self.indices)} clips "
                        f"whose class is not in the training vocabulary")

        if sample_ratio < 1.0:  # class-balanced subset for the semi-supervised protocol
            rng = np.random.RandomState(seed)
            selected = []
            for lbl in np.unique(self.label[self.indices]):
                lbl_indices = self.indices[self.label[self.indices] == lbl]
                rng.shuffle(lbl_indices)
                selected.extend(lbl_indices[:max(1, int(len(lbl_indices) * sample_ratio))])
            self.indices = np.array(selected)
            rng.shuffle(self.indices)
            logger.info(f"Subsampled to {sample_ratio * 100:g}% per class -> {len(self.indices)} samples.")
        logger.info(f"Loaded {len(self.indices)} labelled clips from {self.pack.root}")

    def __len__(self):
        return len(self.indices)

    def get_targets(self):
        return self.label[self.indices]

    def __getitem__(self, index):
        i = self.indices[index]
        seq = _to_clip(self.pack.raw(i), self.skeleton, self.norm)
        if self.transform is not None:
            output = self.transform(seq, seq.shape[1])
        else:
            output = torch.from_numpy(seq).float()
        return output, torch.ones(1, dtype=torch.bool), self.label[i]
