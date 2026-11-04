"""Short single-GPU V1/V2 teacher/student compute benchmark; never starts a long run.

Measures actual forward, MFP/DINO heads, backward, AdamW and EMA on synthetic
already-collated crops. Data loading, checkpoint I/O and multi-GPU scaling are excluded.
"""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slim.configs import slim_default_config
from slim.train.ssl_meta_arch import SSLMetaArch


def inputs(batch, device, cap):
    output = {}
    x = torch.randn(2 * batch, 3, 64, 69, 1, device=device) * 0.1
    x[:, 2] = 1
    mask = torch.zeros(2 * batch, 1104, device=device, dtype=torch.bool)
    for i in range(batch):
        mask[i, torch.randperm(1104, device=device)[:770]] = True
    targets = mask.clone()
    if cap:
        for row in targets:
            ids = row.nonzero().flatten()
            if len(ids) > cap:
                keep = ids[torch.randperm(len(ids), device=device)[:cap]]
                row.zero_()
                row[keep] = True
    indices = targets.flatten().nonzero().flatten()
    output.update(collated_global_crops=x, collated_masks=mask, mfp_masks=targets,
                  global_usable=torch.ones(2 * batch, device=device, dtype=torch.bool),
                  mask_indices_list=indices, n_masked_patches=torch.tensor([len(indices)], device=device),
                  masks_weight=(1 / targets.sum(1).clamp_min(1))[:, None].expand_as(targets)[targets],
                  upperbound=int(mask.sum()))
    for frames, per_parent in ((32, 1), (16, 2), (8, 1)):
        count = 2 * per_parent
        local = torch.randn(count * batch, 3, frames, 69, 1, device=device) * 0.1
        local[:, 2] = 1
        output[f"local_crops_{frames}"] = local
        output[f"local_masks_{frames}"] = torch.zeros(count * batch, frames // 4 * 69, device=device, dtype=torch.bool)
        output[f"local_parents_{frames}"] = torch.tensor([p for p in (0, 1) for _ in range(per_parent * batch)], device=device)
        output[f"local_usable_{frames}"] = torch.ones(count * batch, device=device, dtype=torch.bool)
    return output


def run(architecture, cap, args):
    torch.manual_seed(0)
    config = OmegaConf.merge(OmegaConf.create(slim_default_config),
                             OmegaConf.load(ROOT / 'slim/configs/pretrain/slim_sign_v2_openasl.yaml'))
    config.student.arch = architecture
    config.compute_precision.grad_scaler = False
    model = SSLMetaArch(config).cuda()
    model.teacher.load_state_dict(model.student.state_dict())
    optimizer = torch.optim.AdamW(model.student.parameters(), lr=1e-4)
    data = inputs(args.batch_size, torch.device('cuda'), cap)
    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            losses = model.forward_backward(data, 0.07)
        torch.nn.utils.clip_grad_norm_(model.student.parameters(), 3, foreach=True)
        optimizer.step()
        model.update_teacher(0.994)
        return {k:float(v) for k,v in losses.items()}
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(args.steps):
        begin = time.perf_counter()
        losses = step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - begin)
    result = dict(architecture=architecture, target_cap_per_view=cap, batch_size=args.batch_size,
                  step_seconds=times, median_step_seconds=float(torch.tensor(times).median()),
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                  last_losses=losses, masked_targets=int(data['n_masked_patches']),
                  parameters=sum(p.numel() for p in model.student.backbone.parameters()))
    del optimizer, model, data
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required for this actual SSL compute benchmark')
    torch.cuda.set_device(0)
    with tempfile.TemporaryDirectory() as temp:
        dist.init_process_group('nccl', init_method='file://' + temp + '/rendezvous', rank=0, world_size=1)
        try:
            results = []
            for arch, cap in (('vit_tiny', 0), ('slim_v2', 0), ('slim_v2', 128)):
                item = run(arch, cap, args)
                results.append(item)
                print(json.dumps(item), flush=True)
            payload = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, dtype='bf16',
                           measured='teacher/student, heads, backward, AdamW, EMA; 2 globals + 8 locals',
                           excluded='data loading, checkpoint I/O and multi-GPU communication',
                           warning='Shared GPUs can make latency noisy; this is not downstream accuracy evidence.',
                           results=results)
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2) + '\n')
        finally:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
