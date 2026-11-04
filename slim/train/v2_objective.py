"""Validity-aware teacher/student objective for SLiM V2, on CPU or distributed CUDA."""
import torch
from torch import distributed as dist
from torch.nn import functional as F

from slim.fsdp import reshard_fsdp_model


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


@torch.no_grad()
def balanced_assignments(logits, temperature, iterations=3):
    """Stable distributed Sinkhorn, including ranks with no valid local targets.

    Always executes the same collectives, including when every rank is empty.
    Prototype and batch dimensions are normalized in fp32.
    """
    if temperature <= 0:
        raise ValueError("teacher temperature must be positive")
    q = logits.float().div(temperature).t()
    count = q.new_tensor([q.shape[1]])
    if dist.is_initialized():
        dist.all_reduce(count)
    if count.item() == 0:
        return logits.float()
    peak = q.max() if q.numel() else q.new_tensor(float('-inf'))
    if dist.is_initialized():
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    q = (q - peak).exp().clamp_min(1e-12)
    total = q.sum()
    if dist.is_initialized():
        dist.all_reduce(total)
    q /= total.clamp_min(1e-12)
    for _ in range(iterations):
        rows = q.sum(1, keepdim=True)
        if dist.is_initialized():
            dist.all_reduce(rows)
        q /= rows.clamp_min(1e-12)
        q /= q.shape[0]
        q /= q.sum(0, keepdim=True).clamp_min(1e-12)
        q /= count
    return (q * count).t()


def distributed_mean(total, count):
    """Correct gradient scale when each rank has a different number of valid terms."""
    denominator = total.new_tensor(float(count))
    if dist.is_initialized():
        dist.all_reduce(denominator)
    return total * world_size() / denominator.clamp_min(1)


def soft_ce(logits, targets, temperature=0.1):
    return -(targets * F.log_softmax(logits.float() / temperature, -1)).sum(-1)


def parent_targets(targets, parents, batch_size):
    """Crop-major local rows -> original (unswapped) teacher-global rows."""
    if parents.numel() % batch_size or ((parents < 0) | (parents > 1)).any():
        raise ValueError("each V2 local crop must name its parent global view (0 or 1)")
    sample = torch.arange(parents.numel(), device=parents.device) % batch_size
    return targets[parents * batch_size + sample]


def forward_backward(meta, images, teacher_temp):
    if meta.cfg.train.centering != "sinkhorn_knopp":
        raise ValueError("SLiM V2 currently uses validity-aware Sinkhorn targets")
    if not (meta.do_dino and meta.do_ibot and meta.ibot_separate_head):
        raise ValueError("V2 requires DINO and separate-head joint MFP")
    device = next(meta.student.parameters()).device
    clips = images["collated_global_crops"].to(device)
    masks = images["collated_masks"].to(device)
    global_usable = images["global_usable"].to(device)
    target_masks = images.get("mfp_masks", masks).to(device)
    masked = target_masks.flatten().nonzero().flatten()
    b = clips.shape[0] // 2
    with torch.no_grad():
        teacher = meta.teacher.backbone(clips, is_training=True)
        tq = meta.teacher.dino_head(teacher["x_norm_clstoken"][global_usable])
        assignments = balanced_assignments(tq, teacher_temp)
        cls_targets = assignments.new_zeros(clips.shape[0], tq.shape[-1])
        cls_targets[global_usable] = assignments
        tp = teacher["x_norm_patchtokens"].flatten(0, 1)[masked]
        patch_targets = balanced_assignments(meta.teacher.ibot_head(tp), teacher_temp)
    reshard_fsdp_model(meta.teacher)
    student = meta.student.backbone(clips, masks=masks, is_training=True)
    cls_logits = meta.student.dino_head(student["x_norm_clstoken"])
    other_targets = cls_targets.reshape(2, b, -1).flip(0).flatten(0, 1)
    usable_pair = global_usable & global_usable.reshape(2, b).flip(0).flatten()
    dino_sum = soft_ce(cls_logits[usable_pair], other_targets[usable_pair]).sum()
    pair_count = usable_pair.sum()
    for key in (k for k in images if k.startswith("local_crops_")):
        local = images[key].to(device)
        local_masks = images[key.replace("crops", "masks")].to(device)
        parents = images[key.replace("crops", "parents")].to(device)
        usable = images[key.replace("crops", "usable")].to(device)
        parent_usable = parent_targets(global_usable[:, None], parents, b).flatten()
        usable = usable & parent_usable
        out = meta.student.backbone(local, masks=local_masks, is_training=True)
        logits = meta.student.dino_head(out["x_norm_clstoken"])
        targets = parent_targets(cls_targets, parents, b)
        dino_sum = dino_sum + soft_ce(logits[usable], targets[usable]).sum()
        pair_count = pair_count + usable.sum()
    dino = distributed_mean(dino_sum, pair_count.item()) * meta.dino_loss_weight
    sp = student["x_norm_patchtokens"].flatten(0, 1)[masked]
    patch_logits = meta.student.ibot_head(sp)
    patch_ce = soft_ce(patch_logits, patch_targets)
    weights = images["masks_weight"].to(device)
    masked_views = target_masks.any(1).sum()
    ibot = distributed_mean((patch_ce * weights).sum(), masked_views.item()) * meta.ibot_loss_weight
    koleo = student["x_norm_clstoken"].sum() * 0
    if meta.do_koleo:
        for features, usable in zip(student["x_norm_clstoken"].chunk(2), global_usable.chunk(2)):
            features = features[usable]
            if features.shape[0] > 1:
                koleo = koleo + meta.koleo_loss(features) * 0.5 * meta.cfg.dino.koleo_loss_weight
    total = dino + ibot + koleo
    if not torch.isfinite(total):
        raise FloatingPointError("non-finite SLiM V2 objective")
    meta.backprop_loss(total)
    meta.fsdp_synchronize_streams()
    return {"dino_loss": dino.detach(), "ibot_loss": ibot.detach(), "koleo_loss": koleo.detach()}
