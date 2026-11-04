"""Optional, rank-zero W&B logging for pre-training."""

from contextlib import contextmanager
import json
import logging
from pathlib import Path

from omegaconf import OmegaConf

import slim.distributed as distributed


logger = logging.getLogger("slim")


@contextmanager
def track_training(cfg, enabled=True):
    """Yield a metric callback, reusing the run ID when an output dir resumes."""
    settings = cfg.get("wandb", {})
    if not enabled or not settings.get("enabled", False) or not distributed.is_main_process():
        yield None
        return

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb.enabled=true requires installing the wandb package") from exc

    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "wandb_run.json"
    previous = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    run_id = settings.get("id") or previous.get("id") or wandb.util.generate_id()
    mode = settings.get("mode", "online")
    run = wandb.init(
        project=settings.get("project", "SLiM"),
        entity=settings.get("entity") or None,
        name=settings.get("name") or output_dir.name,
        id=run_id,
        resume="allow" if mode == "online" else None,
        mode=mode,
        dir=str(output_dir),
        config=OmegaConf.to_container(cfg, resolve=True),
        settings=wandb.Settings(init_timeout=60),
    )
    exit_code = 0
    try:
        metadata = dict(id=run.id, project=run.project, entity=run.entity, url=run.url, mode=mode)
        temporary = metadata_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(metadata, indent=2) + "\n")
        temporary.replace(metadata_path)
        run.define_metric("iteration")
        run.define_metric("*", step_metric="iteration")
        logger.info("W&B run: %s", run.url or run.id)

        def record(metrics):
            run.log(dict(metrics), step=int(metrics["iteration"]))

        yield record
    except BaseException:
        exit_code = 1
        raise
    finally:
        run.finish(exit_code=exit_code)
