"""Checkpoint configuration guard and per-rank RNG state for V2 pretraining."""
import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

import slim.distributed as distributed


class V2TrainState:
    def __init__(self, cfg):
        keys = ("student", "crops", "dino", "ibot", "optim")
        self.signature = {key: OmegaConf.to_container(cfg[key], resolve=True) for key in keys}
        self.signature["train"] = {k: cfg.train[k] for k in
                                   ("dataset_path", "batch_size_per_gpu", "OFFICIAL_EPOCH_LENGTH", "centering")}
        self.signature["world_size"] = distributed.get_global_size()
        self.signature["seed"] = torch.initial_seed() - distributed.get_global_rank()
        roots = str(cfg.train.dataset_path).split("root=", 1)[-1].split(":", 1)[0]
        self.signature["stores"] = {}
        for root in roots.split(","):
            path = Path(root) / "meta.json"
            if path.exists():
                self.signature["stores"][str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()

    def state_dict(self):
        return dict(signature=self.signature, python=random.getstate(), numpy=np.random.get_state(),
                    torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)

    def load_state_dict(self, state):
        if self.signature != state["signature"]:
            changed = [k for k in self.signature if self.signature[k] != state["signature"].get(k)]
            raise ValueError(f"V2 resume configuration/data changed: {changed}; use a new output directory")
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if state["cuda"] is not None:
            torch.cuda.set_rng_state(state["cuda"])
