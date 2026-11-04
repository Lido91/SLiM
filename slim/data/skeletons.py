# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Joint layouts for sign-language pose (RTMPose / COCO-WholeBody-133 keypoints).

Packed sign data (tools/pack_sign_poses.py) always stores the 69-joint Uni-Sign layout below,
in this order, as raw (x, y, score). A layout is a subset of it, selected at load time.

    0-8    body   nose, l/r ear, l/r shoulder, l/r elbow, l/r wrist   (COCO 0, 3-10)
    9-29   left hand, 21 joints in COCO hand order                    (COCO 91-111)
    30-50  right hand                                                 (COCO 112-132)
    51-68  face: 9 jaw points, 8 inner-lip points, nose tip           (COCO 23-39:2, 83-90, 53)

Hand order: 0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky (MCP -> tip).
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


UNISIGN69_COCO_INDICES = (
    [0] + list(range(3, 11))
    + list(range(91, 112))
    + list(range(112, 133))
    + list(range(23, 40, 2)) + list(range(83, 91)) + [53]
)

BODY = list(range(0, 9))
LEFT_HAND = list(range(9, 30))
RIGHT_HAND = list(range(30, 51))
FACE = list(range(51, 69))
FACE_ROOT = 68  # nose tip, the face origin in Uni-Sign


def _hand_parents(offset):
    # wrist is the root; each finger is a chain wrist -> MCP -> PIP -> DIP -> tip
    parents = {offset: -1}
    for finger in range(5):
        base = offset + 1 + 4 * finger
        parents[base] = offset
        for j in range(1, 4):
            parents[base + j] = base + j - 1
    return parents


@dataclass
class SkeletonSpec:
    name: str
    joints: List[int]                 # indices into the packed 69-joint layout
    flip_pairs: List[tuple]           # (a, b) pairs in *layout* indices, swapped by mirroring
    parents: List[int]                # per-layout-joint parent for bone scaling, -1 = root
    scale_groups: List[List[int]]     # joints whose incoming bone shares one random scale
    mask_groups: List[List[int]]      # ordered anatomical chains for Semantic Tube Masking
    drop_parts: List[List[int]] = field(default_factory=list)  # parts zeroed by part-drop

    @property
    def num_joints(self):
        return len(self.joints)

    @property
    def flip_order(self):
        order = np.arange(self.num_joints)
        for a, b in self.flip_pairs:
            order[a], order[b] = b, a
        return order

    @property
    def topo_order(self):
        """Joints ordered so that every parent comes before its children."""
        order, seen = [], set()

        def visit(v):
            if v in seen:
                return
            if self.parents[v] >= 0:
                visit(self.parents[v])
            seen.add(v)
            order.append(v)

        for v in range(self.num_joints):
            visit(v)
        return order


def _build_unisign(name, use_face, norm):
    """Build a sign layout. `norm` decides how parts are chained for bone scaling:

    - "part"   (Uni-Sign): hands are wrist-relative and the face nose-relative, so each part is
               its own tree with a fixed root.
    - "global": every joint lives in one body-bbox frame, so hands hang off the body wrists
               and the face off the nose.
    """
    joints = BODY + LEFT_HAND + RIGHT_HAND + (FACE if use_face else [])
    pos = {j: i for i, j in enumerate(joints)}

    packed_parents = {0: -1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 3, 6: 4, 7: 5, 8: 6}
    packed_parents.update(_hand_parents(9))
    packed_parents.update(_hand_parents(30))
    if use_face:
        packed_parents.update({j: FACE_ROOT for j in FACE if j != FACE_ROOT})
        packed_parents[FACE_ROOT] = -1
    if norm == "global":
        packed_parents[9] = 7     # left-hand wrist  <- body left wrist
        packed_parents[30] = 8    # right-hand wrist <- body right wrist
        if use_face:
            packed_parents[FACE_ROOT] = 0
    elif norm != "part":
        raise ValueError(f"unknown pose normalisation: {norm}")
    parents = [pos[packed_parents[j]] if packed_parents[j] >= 0 else -1 for j in joints]

    flip_pairs = [(1, 2), (3, 4), (5, 6), (7, 8)] + [(9 + k, 30 + k) for k in range(21)]
    if use_face:
        jaw = list(range(51, 60))
        flip_pairs += [(jaw[k], jaw[8 - k]) for k in range(4)]
        flip_pairs += [(60, 64), (61, 63), (65, 67)]  # inner lip, 62 / 66 / nose tip are central
    flip_pairs = [(pos[a], pos[b]) for a, b in flip_pairs]

    palm = [h + k for h in (9, 30) for k in (1, 5, 9, 13, 17)]
    fingers = [h + 1 + 4 * f + j for h in (9, 30) for f in range(5) for j in range(1, 4)]
    scale_groups = [[1, 2], [3, 4], [5, 6, 7, 8], palm, fingers]
    if use_face:
        scale_groups.append([j for j in FACE if j != FACE_ROOT])
    scale_groups = [[pos[j] for j in g] for g in scale_groups]

    mask_groups = [[1, 0, 2], [3, 5, 7], [4, 6, 8], LEFT_HAND, RIGHT_HAND]
    drop_parts = [LEFT_HAND, RIGHT_HAND]
    if use_face:
        mask_groups += [list(range(51, 60)), list(range(60, 69))]
        drop_parts.append(FACE)
    mask_groups = [[pos[j] for j in g] for g in mask_groups]
    drop_parts = [[pos[j] for j in g] for g in drop_parts]

    return SkeletonSpec(name, joints, flip_pairs, parents, scale_groups, mask_groups, drop_parts)


SIGN_LAYOUTS = {
    "sign69": dict(use_face=True),   # Uni-Sign: body + both hands + face
    "sign51": dict(use_face=False),  # body + both hands
}


def is_sign_skeleton(name: Optional[str]) -> bool:
    return name in SIGN_LAYOUTS


def get_skeleton(name: str, norm: str = "part") -> SkeletonSpec:
    if name not in SIGN_LAYOUTS:
        raise ValueError(f"unknown sign skeleton {name!r}, expected one of {sorted(SIGN_LAYOUTS)}")
    return _build_unisign(name, norm=norm, **SIGN_LAYOUTS[name])


def normalize_sign_pose(raw, norm="part", thr=0.3):
    """Normalise a packed (T, 69, 3) raw (x, y, score) sequence like Uni-Sign's `load_part_kp`.

    The scale is the side of the square bbox of all confident body joints over the clip.
    - "part":   body -> [-1, 1] in that bbox; hands relative to their wrist and face relative
                to the nose tip, divided by the same scale (Uni-Sign's exact recipe).
    - "global": every joint mapped into the body bbox frame like the body.
    Joints with score <= thr are zeroed in all three channels. Returns float32 (T, 69, 3).
    """
    raw = np.asarray(raw, dtype=np.float32)
    out = np.zeros_like(raw)
    body = raw[:, BODY]
    valid_coords = body[body[..., 2] > thr][:, :2]
    if len(valid_coords) < 4:
        return out
    xmin, ymin = valid_coords.min(0)
    xmax, ymax = valid_coords.max(0)
    scale = max(xmax - xmin, ymax - ymin)
    if scale == 0:
        return out
    origin = np.array([(xmin + xmax - scale) / 2, (ymin + ymax - scale) / 2], dtype=np.float32)

    out[..., 2] = raw[..., 2]
    if norm == "global":
        out[..., :2] = ((raw[..., :2] - origin) / scale - 0.5) * 2
    elif norm == "part":
        out[:, BODY, :2] = ((body[..., :2] - origin) / scale - 0.5) * 2
        for part, root in ((LEFT_HAND, LEFT_HAND[0]), (RIGHT_HAND, RIGHT_HAND[0]), (FACE, FACE_ROOT)):
            out[:, part, :2] = (raw[:, part, :2] - raw[:, root:root + 1, :2]) / scale
    else:
        raise ValueError(f"unknown pose normalisation: {norm}")
    out = np.clip(out, -1, 1)
    out[out[..., 2] <= thr] = 0
    return out
