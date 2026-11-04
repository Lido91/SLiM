"""SHuBERT-style rank-1 LoRA adaptation of SLiM sign encoders on WLASL.

Run with torchrun; see scripts/eval_wlasl_lora_compare.py. The full model is DDP,
the original encoder is frozen, and dev/test samplers never pad or duplicate IDs.
"""

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from slim.data.augmentations import DownstreamAugmentation
from slim.data.datasets.sign import SignPose_Classification
from slim.data.skeletons import get_skeleton
from slim.hub import load_encoder_from_pth


ENCODER_CONFIG = {"img_size": [64, 69], "patch_size": [4, 1]}
PAPER = "https://arxiv.org/html/2411.16765v3#S4.SS3"


class LoRALinear(nn.Module):
    def __init__(self, base, rank=1, alpha=1.0):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        self.base = base.requires_grad_(False)
        self.a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + nn.functional.linear(nn.functional.linear(x, self.a), self.b) * self.scale


def add_lora(module, rank=1, alpha=1.0, prefix=""):
    names = []
    for name, child in list(module.named_children()):
        qualified = prefix + name
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank, alpha))
            names.append(qualified)
        else:
            names.extend(add_lora(child, rank, alpha, qualified + "."))
    return names


class SignLoRAClassifier(nn.Module):
    def __init__(self, encoder, classes=2000, rank=1, alpha=1.0):
        super().__init__()
        self.encoder = encoder.requires_grad_(False)
        self.lora_layers = add_lora(self.encoder, rank, alpha)
        if not self.lora_layers:
            raise RuntimeError("No encoder Linear layers received LoRA")
        self.head = nn.Sequential(nn.BatchNorm1d(encoder.embed_dim), nn.Linear(encoder.embed_dim, classes))

    def forward(self, clips):
        features = self.encoder.forward_features(clips)
        if "x_norm_embedding" in features:
            return self.head(features["x_norm_embedding"])
        patches = features["x_norm_patchtokens"]
        # SLiM has joint tokens at each temporal patch. Average joints first, then
        # time, to obtain the temporal-average representation used by the head.
        temporal = patches.reshape(patches.shape[0], clips.shape[2] // 4, 69, -1).mean(dim=2)
        return self.head(temporal.mean(dim=1))


class ExactEvalSampler(Sampler):
    """Rank-strided evaluation: each real example is evaluated exactly once."""
    def __init__(self, dataset, rank=0, world_size=1):
        self.n, self.rank, self.world_size = len(dataset), rank, world_size

    def __iter__(self):
        return iter(range(self.rank, self.n, self.world_size))

    def __len__(self):
        return len(range(self.rank, self.n, self.world_size))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_torch_save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def seed_worker(_):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])


def frozen_digest(model):
    h = hashlib.sha256()
    for name, p in model.encoder.named_parameters():
        if not p.requires_grad:
            h.update(name.encode())
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def metrics_from_counts(counts):
    counts = counts.cpu().double()
    present = counts[:, 0] > 0
    n = counts[:, 0].sum().item()
    if not n:
        raise RuntimeError("Empty evaluation")
    result = {"samples": int(n), "classes_present": int(present.sum())}
    for column, k in enumerate((1, 5, 10), start=1):
        result[f"pi_top{k}"] = 100 * counts[:, column].sum().item() / n
        result[f"pc_top{k}"] = 100 * (counts[present, column] / counts[present, 0]).mean().item()
    return result


def check_data(data_path):
    manifests = {}
    all_names = set()
    classes = None
    for split in ("train", "dev", "test"):
        path = Path(data_path) / split / "meta.json"
        meta = json.loads(path.read_text())
        names = meta["names"]
        if len(names) != len(set(names)) or all_names.intersection(names):
            raise RuntimeError(f"Duplicate video IDs within/across WLASL splits: {split}")
        all_names.update(names)
        if classes is None:
            classes = meta["classes"]
        if meta["classes"] != classes or len(classes) != 2000:
            raise RuntimeError("Inconsistent WLASL-2000 vocabulary")
        if len(names) != len(meta["labels"]) or min(meta["labels"]) < 0 or max(meta["labels"]) >= 2000:
            raise RuntimeError("Invalid/missing WLASL labels")
        manifests[split] = {"samples": len(names), "meta_sha256": sha256(path), "files": {}}
        for name in ("poses.npy", "offsets.npy"):
            stat = (path.parent / name).stat()
            manifests[split]["files"][name] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return manifests


def make_loader(args, split, rank, world_size):
    spec = get_skeleton("sign69", norm="part")
    dataset = SignPose_Classification(args.data_path, split=split, skeleton="sign69", norm="part",
                                     transform=DownstreamAugmentation(64, split, spec), sample_ratio=1.0)
    if split == "train":
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                     shuffle=True, seed=args.seed, drop_last=True)
    else:
        sampler = ExactEvalSampler(dataset, rank, world_size)
    generator = torch.Generator()
    loader = DataLoader(dataset, batch_size=args.batch_size // world_size, sampler=sampler,
                        num_workers=args.workers, pin_memory=True, drop_last=(split == "train"),
                        worker_init_fn=seed_worker, generator=generator)
    return loader, sampler, generator


def amp_context(args):
    return torch.autocast("cuda", dtype=torch.float16) if not args.fp32 else contextlib.nullcontext()


@torch.inference_mode()
def evaluate(model, loader, args, rank, world_size, predictions=False, max_batches=0):
    # Bypass DDP forward here: rank-strided evaluation has unequal last batches.
    # SyncBatchNorm uses saved running stats in eval and makes no collectives.
    model.eval()
    counts = torch.zeros((2000, 4), device="cuda", dtype=torch.float64)
    losses = torch.zeros(2, device="cuda", dtype=torch.float64)
    records = []
    local_indices = list(loader.sampler)
    position = 0
    for batch_index, (clips, _, labels) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        clips, labels = clips.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        with amp_context(args):
            logits = model(clips)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite evaluation logits")
        loss = nn.functional.cross_entropy(logits.float(), labels, reduction="sum")
        top = logits.topk(10, dim=1).indices
        correct = top.eq(labels[:, None])
        counts[:, 0].scatter_add_(0, labels, torch.ones_like(labels, dtype=torch.float64))
        for column, k in enumerate((1, 5, 10), start=1):
            counts[:, column].scatter_add_(0, labels, correct[:, :k].any(1).double())
        losses[0] += loss
        losses[1] += labels.numel()
        if predictions:
            top_cpu, labels_cpu = top.cpu().tolist(), labels.cpu().tolist()
            for j in range(labels.numel()):
                index = local_indices[position + j]
                pack_index = int(loader.dataset.indices[index])
                records.append({"index": index, "id": loader.dataset.pack.names[pack_index],
                                "label": labels_cpu[j], "top10": top_cpu[j]})
        position += labels.numel()
    if world_size > 1:
        dist.all_reduce(counts)
        dist.all_reduce(losses)
    metrics = metrics_from_counts(counts)
    metrics["loss"] = (losses[0] / losses[1]).item()
    if not max_batches and metrics["samples"] != len(loader.dataset):
        raise RuntimeError("Evaluation sample count mismatch")
    if predictions and world_size > 1:
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(records, gathered, dst=0)
        records = [r for shard in gathered for r in shard] if rank == 0 else []
    if predictions and rank == 0:
        records.sort(key=lambda r: r["index"])
        if len(records) != len(loader.dataset) or len({r["id"] for r in records}) != len(records):
            raise RuntimeError("Prediction ID coverage/uniqueness failure")
    return metrics, records


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pretrain-step", type=int, required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--data-path", default="data/sign/WLASL")
    p.add_argument("--dev-split", choices=("dev", "test"), default="dev",
                   help="Split used for epoch selection/early stopping; test gives test-as-dev scores")
    p.add_argument("--epochs", type=int, default=125)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128, help="Global batch across all ranks")
    p.add_argument("--workers", type=int, default=4, help="Workers per GPU")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--lora-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--rank", type=int, default=1, help="LoRA rank")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--mode", choices=("train", "test", "smoke"), default="train")
    p.add_argument("--smoke-steps", type=int, default=4)
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    p.add_argument("--wandb-project", default="SLiM")
    p.add_argument("--wandb-entity", default="cvpr-eccv-iccv")
    p.add_argument("--wandb-group", default="wlasl-lora-r1")
    return p.parse_args()


def main():
    args = parse_args()
    if args.dev_split == "test" and args.mode == "test":
        raise ValueError("Test is already used as dev; report test-as-dev scores from training")
    score_prefix = "test_as_dev" if args.dev_split == "test" else "dev"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(2)
    if world_size > 1:
        dist.init_process_group("nccl")
    if args.batch_size % world_size or args.batch_size // world_size < 2:
        raise ValueError("Global batch must divide world size, with at least 2 samples per GPU")
    output = Path(args.output_dir).resolve()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    # Identical initialization and sampler seeds for all three pretrained snapshots.
    seed_all(args.seed)
    encoder = load_encoder_from_pth(args.checkpoint, ENCODER_CONFIG)
    encoder_config = encoder.get_config() if hasattr(encoder, "get_config") else ENCODER_CONFIG
    model = SignLoRAClassifier(encoder, rank=args.rank, alpha=args.alpha)
    before_digest = frozen_digest(model)
    if world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda()
    encoder_trainable = [p for p in model.encoder.parameters() if p.requires_grad]
    head_trainable = list(model.head.parameters())
    optimizer = torch.optim.Adam([
        {"params": head_trainable, "lr": args.head_lr},
        {"params": encoder_trainable, "lr": args.lora_lr},
    ], weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=not args.fp32)
    config = {key: value for key, value in vars(args).items()
              if key not in {"mode", "resume", "output_dir", "smoke_steps"}}
    config.update({"checkpoint": str(Path(args.checkpoint).resolve()),
                   "checkpoint_sha256": sha256(args.checkpoint), "data_path": str(Path(args.data_path).resolve()),
                   "data": check_data(args.data_path), "world_size": world_size, "encoder": encoder_config,
                   "lora_layers": model.lora_layers, "lora_parameters": sum(p.numel() for p in encoder_trainable),
                   "head_parameters": sum(p.numel() for p in head_trainable), "frozen_sha256": before_digest,
                   "paper": PAPER, "selection": f"max {score_prefix} pi_top1, earliest epoch on ties",
                   "selection_split": args.dev_split, "independent_test": args.dev_split == "dev",
                   "scheduler": "constant (paper does not specify a downstream schedule)",
                   "patience_note": "20 non-improving epochs is an implementation choice; paper gives no patience",
                   "sampling": "SLiM sign69 part normalization; existing train augmentations; 64 frames; deterministic dev/test",
                   "pooling": ("validity-aware part-balanced joint detail plus CLS; BN + Linear" if encoder_config.get("arch") == "slim_v2"
                               else "last contextual patch layer, mean joints then time; BN1d(256) + Linear(256,2000)"),
                   "precision": "float32" if args.fp32 else "float16 autocast + GradScaler",
                   "sync_bn": world_size > 1, "drop_last_train": True, "eval_padding": False,
                   "code_sha256": sha256(__file__), "torch": torch.__version__})
    config_path = output / "config.json"
    if config_path.exists():
        saved_config = json.loads(config_path.read_text())
        if saved_config != config:
            raise RuntimeError("Refusing to reuse output with a different protocol/data/checkpoint/code")
        if args.mode == "train" and not args.resume:
            raise RuntimeError("Existing training output: pass --resume")
    elif rank == 0:
        atomic_json(config_path, config)
    if world_size > 1:
        dist.barrier()

    start_epoch, best, bad_epochs, global_step = 0, {"pi_top1": -1.0}, 0, 0
    best_epoch, run = -1, None
    if args.mode == "test" or (args.resume and (output / "last.pth").exists()):
        path = output / ("best.pth" if args.mode == "test" else "last.pth")
        saved = torch.load(path, map_location="cpu")
        if saved["config"] != config:
            raise RuntimeError("Checkpoint protocol mismatch")
        model.load_state_dict(saved["model"], strict=True)
        best, best_epoch = saved["best_dev"], saved["best_epoch"]
        if args.mode != "test":
            optimizer.load_state_dict(saved["optimizer"])
            scaler.load_state_dict(saved["scaler"])
            start_epoch, bad_epochs = saved["epoch"] + 1, saved["bad_epochs"]
            global_step = saved["global_step"]
            if len(saved["rng_states"]) != world_size:
                raise RuntimeError("Resume requires same world size")
            restore_rng(saved["rng_states"][rank])
    else:
        seed_all(args.seed + rank)

    if args.mode == "test":
        if (output / "test_metrics.json").exists():
            raise RuntimeError("Test already evaluated; refusing a repeated test")
        test_loader, _, _ = make_loader(args, "test", rank, world_size)
        test_metrics, records = evaluate(model, test_loader, args, rank, world_size, predictions=True)
        if rank == 0:
            with open(output / "test_predictions.jsonl.tmp", "w") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")
            os.replace(output / "test_predictions.jsonl.tmp", output / "test_predictions.jsonl")
            atomic_json(output / "test_metrics.json", {"pretrain_step": args.pretrain_step,
                        "selected_epoch": best_epoch + 1, "best_dev": best, "test": test_metrics,
                        "checkpoint_sha256": config["checkpoint_sha256"]})
            print("FINAL_TEST " + json.dumps(test_metrics), flush=True)
        if world_size > 1:
            dist.barrier()
            dist.destroy_process_group()
        return

    if rank == 0 and args.wandb_mode != "disabled":
        import wandb
        id_path = output / "wandb_run.json"
        previous = json.loads(id_path.read_text()) if id_path.exists() else {}
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         group=args.wandb_group, name=f"wlasl_lora_r{args.rank}_pt{args.pretrain_step}_seed{args.seed}_{score_prefix}",
                         id=previous.get("id"), resume="allow" if previous else None,
                         config=config, dir=str(output), mode=args.wandb_mode)
        atomic_json(id_path, {"id": run.id, "url": run.url, "project": args.wandb_project,
                              "entity": args.wandb_entity, "group": args.wandb_group})
    train_loader, train_sampler, train_generator = make_loader(args, "train", rank, world_size)
    dev_loader, _, dev_generator = make_loader(args, args.dev_split, rank, world_size)
    ddp = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False) if world_size > 1 else model
    total_start = time.monotonic()
    epochs_completed = start_epoch
    if rank == 0:
        print(f"START pt={args.pretrain_step} rank={args.rank} global_batch={args.batch_size} world={world_size} "
              f"LoRA_params={config['lora_parameters']} head_params={config['head_parameters']} "
              f"train={len(train_loader.dataset)} {score_prefix}={len(dev_loader.dataset)} "
              f"selection_split={args.dev_split} steps/epoch={len(train_loader)}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        if bad_epochs >= args.patience:
            break
        epoch_start = time.monotonic()
        model.train()
        train_sampler.set_epoch(epoch)
        train_generator.manual_seed(args.seed + epoch * world_size + rank)
        dev_generator.manual_seed(args.seed + epoch * world_size + rank)
        totals = torch.zeros(3, device="cuda", dtype=torch.float64)
        for batch_index, (clips, _, labels) in enumerate(train_loader):
            clips, labels = clips.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(args):
                logits = ddp(clips)
                loss = nn.functional.cross_entropy(logits.float(), labels, label_smoothing=args.label_smoothing)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            scaler.scale(loss).backward()
            if args.mode == "smoke":
                # With B=0, A's first gradient can be zero, but must be connected.
                if any(p.grad is None for p in model.parameters() if p.requires_grad):
                    raise RuntimeError("A trainable parameter is disconnected from the loss")
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            totals[0] += loss.detach() * labels.numel()
            totals[1] += logits.argmax(1).eq(labels).sum()
            totals[2] += labels.numel()
            if rank == 0 and (batch_index % 10 == 0 or batch_index + 1 == len(train_loader)):
                progress = {"phase": args.mode, "pretrain_step": args.pretrain_step, "epoch": epoch + 1,
                            "selection_split": args.dev_split,
                            "batch": batch_index + 1, "batches_per_epoch": len(train_loader),
                            "global_step": global_step, "loss_rank0": loss.item(),
                            "elapsed_epoch_seconds": time.monotonic() - epoch_start,
                            "updated_unix": time.time()}
                atomic_json(output / "progress.json", progress)
                print("PROGRESS " + json.dumps(progress), flush=True)
                if run:
                    run.log({"train/batch_loss_rank0": loss.item(), "train/global_step": global_step,
                             "epoch": epoch + (batch_index + 1) / len(train_loader)})
            if args.mode == "smoke" and batch_index + 1 >= args.smoke_steps:
                break
        if world_size > 1:
            dist.all_reduce(totals)
        metrics, _ = evaluate(model, dev_loader, args, rank, world_size,
                              max_batches=2 if args.mode == "smoke" else 0)
        improved = metrics["pi_top1"] > best["pi_top1"]
        if improved:
            best, best_epoch, bad_epochs = metrics, epoch, 0
        else:
            bad_epochs += 1
        row = {"epoch": epoch + 1, "global_step": global_step, "selection_split": args.dev_split,
               "train_loss": (totals[0] / totals[2]).item(),
               "train_pi_top1": (100 * totals[1] / totals[2]).item(), "train_samples": int(totals[2].item()),
               "dev": metrics, "best_epoch": best_epoch + 1, "best_dev": best,
               "bad_epochs": bad_epochs, "seconds": time.monotonic() - epoch_start,
               "max_gpu_memory_gib_rank0": torch.cuda.max_memory_allocated() / 2**30,
               "scaler_scale": scaler.get_scale()}
        epochs_completed = epoch + 1
        if args.mode == "smoke":
            after_digest = frozen_digest(model)
            if after_digest != before_digest:
                raise RuntimeError("Frozen original encoder parameters changed")
            if not any(layer.b.detach().abs().sum().item() > 0 for layer in model.encoder.modules()
                       if isinstance(layer, LoRALinear)):
                raise RuntimeError("No LoRA weights updated during GPU smoke")
            # All ranks must have the same updated adapters and classifier.
            trainable = torch.cat([p.detach().float().flatten() for p in model.parameters() if p.requires_grad])
            mismatch = torch.zeros((), device="cuda")
            if world_size > 1:
                reference = trainable.clone()
                dist.broadcast(reference, src=0)
                mismatch = (reference - trainable).abs().max()
                dist.all_reduce(mismatch, op=dist.ReduceOp.MAX)
            if mismatch.item() > 1e-7:
                raise RuntimeError("DDP adapter/head weights diverged between ranks")
            row.update({"frozen_weights_unchanged": True, "max_ddp_parameter_difference": mismatch.item()})
            if rank == 0:
                atomic_torch_save(output / "smoke_reload.pth", {"model": model.state_dict(),
                                  "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()})
            if world_size > 1:
                dist.barrier()
            loaded = torch.load(output / "smoke_reload.pth", map_location="cpu")
            model.load_state_dict(loaded["model"], strict=True)
            optimizer.load_state_dict(loaded["optimizer"])
            scaler.load_state_dict(loaded["scaler"])
            reloaded = torch.cat([p.detach().float().flatten() for p in model.parameters() if p.requires_grad])
            if not torch.equal(trainable, reloaded):
                raise RuntimeError("Saved adapters/head do not round-trip exactly")
            row["checkpoint_roundtrip_exact"] = True
            if rank == 0:
                atomic_json(output / "smoke_result.json", row)
                print("SMOKE_OK " + json.dumps(row), flush=True)
            break

        states = [None] * world_size
        if world_size > 1:
            dist.all_gather_object(states, rng_state())
        else:
            states[0] = rng_state()
        if rank == 0:
            if frozen_digest(model) != before_digest:
                raise RuntimeError("Frozen encoder changed")
            saved = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                     "epoch": epoch, "global_step": global_step, "best_dev": best, "best_epoch": best_epoch,
                     "bad_epochs": bad_epochs, "rng_states": states, "config": config}
            atomic_torch_save(output / "last.pth", saved)
            if improved:
                atomic_torch_save(output / "best.pth", saved)
            with open(output / "metrics.jsonl", "a") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")
            atomic_json(output / "progress.json", {"phase": "epoch_finished", "pretrain_step": args.pretrain_step,
                        **row, "updated_unix": time.time()})
            print("EPOCH " + json.dumps(row), flush=True)
            if run:
                run.log({"epoch": epoch + 1, "train/loss": row["train_loss"], "train/pi_top1": row["train_pi_top1"],
                         **{score_prefix + "/" + k: v for k, v in metrics.items()},
                         "best/" + score_prefix + "_pi_top1": best["pi_top1"],
                         "best/epoch": best_epoch + 1, "epoch_seconds": row["seconds"]})
        if world_size > 1:
            dist.barrier()

    if rank == 0:
        if args.mode == "train":
            summary = {"pretrain_step": args.pretrain_step, "best_epoch": best_epoch + 1, "best_dev": best,
                       "selection_split": args.dev_split, "score_role": score_prefix,
                       "independent_test": args.dev_split == "dev",
                       "epochs_completed": epochs_completed,
                       "stopped_early": bad_epochs >= args.patience, "elapsed_seconds": time.monotonic() - total_start,
                       "checkpoint_sha256": config["checkpoint_sha256"], "wandb_url": run.url if run else None}
            atomic_json(output / "dev_summary.json", summary)
            print("TRAIN_DONE " + json.dumps(summary), flush=True)
        if run:
            run.summary["best/" + score_prefix + "_pi_top1"] = best["pi_top1"]
            run.summary["best/epoch"] = best_epoch + 1
            run.summary["selection_split"] = args.dev_split
            run.finish()
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
