# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Skeleton-aware augmentations (SAA) for 2D sign-language pose.

Clips are numpy (C, T, V, M) with C = (x, y, score). A joint with score 0 is missing (it was
zeroed by `normalize_sign_pose`) and must stay all-zero after every augmentation.
"""

import math
import random

import numpy as np


def _valid(data):
    return data[2] > 0  # (T, V, M)


def sign_flip(data, skeleton, p=0.5):
    """Mirror x and swap left/right joints (arms, hands, face contour)."""
    if random.random() < p:
        data = data[:, :, skeleton.flip_order, :].copy()
        data[0] = -data[0]
    return data


def sign_bone_scale(data, skeleton, scale_range=(0.85, 1.15), p=0.5):
    """Scale bone lengths per anatomical group and rebuild joints from the roots.

    Joints are visited parent-first. When a bone touches a missing joint it is not scaled; the
    child just inherits its parent's displacement, so missing joints never inject coordinates.
    """
    if random.random() >= p:
        return data.copy()
    C, T, V, M = data.shape
    scales = np.ones((V, M), dtype=data.dtype)
    for m in range(M):
        for group in skeleton.scale_groups:
            scales[group, m] = random.uniform(*scale_range)

    xy = data[:2]
    valid = _valid(data)
    new_xy = xy.copy()
    shift = np.zeros_like(xy)  # displacement new - old, propagated down the tree
    for v in skeleton.topo_order:
        parent = skeleton.parents[v]
        if parent < 0:
            continue
        both = (valid[:, v] & valid[:, parent])[None]
        scaled = new_xy[:, :, parent] + scales[v] * (xy[:, :, v] - xy[:, :, parent])
        new_xy[:, :, v] = np.where(both, scaled, xy[:, :, v] + shift[:, :, parent])
        shift[:, :, v] = new_xy[:, :, v] - xy[:, :, v]

    out = data.copy()
    out[:2] = new_xy * valid[None]
    return out


def sign_rotate(data, theta=0.3, p=0.5):
    """In-plane rotation of (x, y). Every part shares one angle, so part-local frames stay consistent."""
    if random.random() < p:
        a = random.uniform(-theta, theta)
        c, s = math.cos(a), math.sin(a)
        x, y = data[0].copy(), data[1].copy()
        data = data.copy()
        data[0] = c * x - s * y
        data[1] = s * x + c * y
    return data


def sign_noise(data, std=0.01, p=0.5):
    if random.random() < p:
        data = data.copy()
        data[:2] += np.random.normal(0, std, data[:2].shape).astype(data.dtype) * _valid(data)[None]
    return data


def sign_part_drop(data, skeleton, p=0.2):
    """Zero one whole part (a hand or the face), the 2D analogue of a missed detection.

    Replaces NTU's axis-drop: with only (x, y) there is no redundant axis to drop.
    """
    if skeleton.drop_parts and random.random() < p:
        data = data.copy()
        data[:, :, random.choice(skeleton.drop_parts), :] = 0
    return data
