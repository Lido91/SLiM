# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import logging
import math
import os
from functools import partial

from fvcore.common.checkpoint import PeriodicCheckpointer
import torch

from slim.data import SamplerType, make_data_loader, make_dataset
from slim.data import collate_data_and_cast, DataAugmentationDINO, MaskingGenerator
from slim.data.skeletons import get_skeleton, is_sign_skeleton
import slim.distributed as distributed
from slim.fsdp import FSDPCheckpointer
from slim.logging import MetricLogger
from slim.logging.wandb import track_training
from slim.utils.config import setup
from slim.utils.utils import CosineScheduler

from slim.train.ssl_meta_arch import SSLMetaArch


torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
logger = logging.getLogger("slim")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("SLiM pre-training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (offset by rank)")
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )

    return parser


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


def build_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[
        : cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH
    ] = 0  # mimicking the original schedules

    logger.info("Schedulers ready.")

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier


def do_test(cfg, model, iteration):
    new_state_dict = model.teacher.state_dict()

    if distributed.is_main_process():
        iterstring = str(iteration)
        eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)
        os.makedirs(eval_dir, exist_ok=True)
        # save teacher checkpoint
        teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
        backbone = model.teacher["backbone"]
        payload = {"teacher": new_state_dict}
        if hasattr(backbone, "get_config"):
            payload["encoder_config"] = backbone.get_config()
        torch.save(payload, teacher_ckp_path)


def do_train(cfg, model, resume=False, metric_sink=None):
    model.train()
    inputs_dtype = torch.half
    fp16_scaler = model.fp16_scaler  # for mixed precision training

    # setup optimizer

    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)

    # checkpointer
    checkpointables = {"optimizer": optimizer}
    if cfg.student.arch == "slim_v2":
        from .v2_state import V2TrainState
        checkpointables["v2_state"] = V2TrainState(cfg)
        if fp16_scaler is not None:
            checkpointables["scaler"] = fp16_scaler
    checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, save_to_disk=True, **checkpointables)

    if cfg.student.arch == "slim_v2" and (checkpointer.has_checkpoint() or cfg.MODEL.WEIGHTS):
        checkpoint_path = checkpointer.get_checkpoint_file() if resume and checkpointer.has_checkpoint() else cfg.MODEL.WEIGHTS
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if "v2_state" not in checkpoint:
                raise ValueError("V2 training requires a V2 checkpoint; V1 weights cannot resume this architecture")
            checkpointables["v2_state"].load_state_dict(checkpoint["v2_state"])
            del checkpoint
    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH
    if cfg.train.get("max_steps", 0):
        max_iter = min(max_iter, int(cfg.train.max_steps))
    if start_iter >= max_iter:
        logger.info("Requested training budget already complete at iteration %s", start_iter)
        return {}

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period=3 * OFFICIAL_EPOCH_LENGTH,
        max_iter=max_iter,
        max_to_keep=3,
    )

    # setup data preprocessing
    n_people = cfg.crops.num_people
    img_size = cfg.crops.global_crops_size  # T, V
    patch_size = cfg.student.patch_size
    n_tokens = img_size[0] // patch_size[0] * img_size[1] // patch_size[1]

    # Sign pose: joint layout drives STM groups, augmentations and the dataset; NTU keeps defaults.
    skeleton, mask_kwargs, dataset_kwargs = None, {}, {}
    if is_sign_skeleton(cfg.crops.skeleton):
        skeleton = get_skeleton(cfg.crops.skeleton, norm=cfg.crops.pose_norm)
        assert img_size[1] == skeleton.num_joints and patch_size[1] == 1, (
            f"crops.global_crops_size[1] must be {skeleton.num_joints} joints with P_J = 1 for {skeleton.name}"
        )
        # sqrt(group size) weights give hands, arms and face ~equal per-joint mask rates
        mask_kwargs = dict(joint_groups=skeleton.mask_groups,
                           group_weights=[len(g) ** 0.5 for g in skeleton.mask_groups])
        dataset_kwargs = dict(skeleton=cfg.crops.skeleton, norm=cfg.crops.pose_norm)

    mask_generator = MaskingGenerator(
        input_size=(img_size[0] // patch_size[0], img_size[1] // patch_size[1]),
        min_num_patches=8,
        max_num_patches=0.5 * n_tokens,
        **mask_kwargs,
    )

    local_mask_generator = {
        'local_crops_32': MaskingGenerator(
            input_size=(img_size[0] // 2 // patch_size[0], img_size[1] // patch_size[1]),
            min_num_patches=4,
            max_num_patches=0.5 * n_tokens,
            **mask_kwargs,
        ),
        'local_crops_16': MaskingGenerator(
            input_size=(img_size[0] // 4 // patch_size[0], img_size[1] // patch_size[1]),
            min_num_patches=2,
            max_num_patches=0.5 * n_tokens,
            **mask_kwargs,
        ),
        'local_crops_8': MaskingGenerator(
            input_size=(img_size[0] // 8 // patch_size[0], img_size[1] // patch_size[1]),
            min_num_patches=1,
            max_num_patches=0.5 * n_tokens,
            **mask_kwargs,
        )
    }

    data_transform = DataAugmentationDINO(
        cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size[0],
        local_crops_size=cfg.crops.local_crops_size[0],
        split='train',
        skeleton=skeleton,
        max_window=cfg.crops.max_window_frames,
        parent_matching=cfg.student.arch == "slim_v2",
        min_global_overlap=cfg.crops.get("min_global_overlap", 0.0),
        noise_std=cfg.crops.get("noise_std", 0.005),
    )

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        local_mask_ratio_tuple=cfg.ibot.local_mask_ratio_min_max,
        local_masking=cfg.ibot.local_masking,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        n_people=n_people,
        mask_generator=mask_generator,
        local_mask_generator=local_mask_generator,
        respect_validity=cfg.student.arch == "slim_v2",
        patch_size=patch_size[0],
        max_mfp_tokens_per_view=cfg.ibot.get("max_mfp_tokens_per_view", 0),
        dtype=inputs_dtype,
    )

    # setup data loader

    dataset = make_dataset(
        dataset_str=cfg.train.dataset_path,
        transform=data_transform,
        target_transform=lambda _: (),
        **dataset_kwargs,
    )
    sampler_type = SamplerType.SHARDED_INFINITE
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        seed=(torch.initial_seed() - distributed.get_global_rank()) if cfg.student.arch == "slim_v2" else start_iter,
        sampler_type=sampler_type,
        sampler_advance=start_iter * cfg.train.batch_size_per_gpu if cfg.student.arch == "slim_v2" else 0,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # training loop

    iteration = start_iter

    logger.info("Starting training from iteration {}".format(start_iter))
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file, metric_sink=metric_sink)
    header = "Training"

    for data in metric_logger.log_every(
        data_loader,
        10,
        header,
        max_iter,
        start_iter,
    ):
        current_batch_size = data["collated_global_crops"].shape[0] / 2
        if iteration > max_iter:
            return

        # apply schedules

        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        mom = momentum_schedule[iteration]
        teacher_temp = teacher_temp_schedule[iteration]
        last_layer_lr = last_layer_lr_schedule[iteration]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # compute losses

        optimizer.zero_grad(set_to_none=True)
        loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)

        # clip gradients

        if fp16_scaler is not None:
            if cfg.optim.clip_grad:
                fp16_scaler.unscale_(optimizer)
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            if cfg.optim.clip_grad:
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            optimizer.step()

        # perform teacher EMA update

        model.update_teacher(mom)

        # logging

        if distributed.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
        loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

        if math.isnan(sum(loss_dict_reduced.values())):
            logger.info("NaN detected")
            raise AssertionError
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(teacher_temp=teacher_temp)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(current_batch_size=current_batch_size)
        metric_logger.update(global_batch_size=current_batch_size * distributed.get_global_size())
        metric_logger.update(epoch=(iteration + 1) / OFFICIAL_EPOCH_LENGTH)
        metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)

        # checkpointing and testing

        if cfg.evaluation.eval_period_iterations > 0 and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0:
            do_test(cfg, model, f"training_{iteration}")
            torch.cuda.synchronize()
        periodic_checkpointer.step(iteration)

        iteration = iteration + 1
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    cfg = setup(args)

    with track_training(cfg, enabled=not args.eval_only) as metric_sink:
        model = SSLMetaArch(cfg).to(torch.device("cuda"))
        model.prepare_for_distributed_training()

        logger.info("Model:\n{}".format(model))
        if args.eval_only:
            iteration = (
                FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
                .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
                .get("iteration", -1)
                + 1
            )
            return do_test(cfg, model, f"manual_{iteration}")

        do_train(cfg, model, resume=not args.no_resume, metric_sink=metric_sink)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
