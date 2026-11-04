#!/usr/bin/env python3
"""Compare three teacher snapshots with an explicit dev or test-as-dev selection split."""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


STEPS = (62500, 75000, 87500)
ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpus", default="0,1,2", help="One GPU for each requested checkpoint, in step order")
    parser.add_argument("--data-path", default=str(ROOT / "data/sign/WLASL"))
    parser.add_argument("--dev-split", default="dev", choices=("dev", "test"))
    parser.add_argument("--pretrain-dir", default=str(ROOT / "outputs/slim_sign_openasl"))
    parser.add_argument("--wandb-group", default="wlasl-lora-r1-62500-75000-87500")
    parser.add_argument("--wandb-mode", default="online", choices=("online", "offline", "disabled"))
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = open(output / "comparison.lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("This comparison already has a running controller")
    devices = args.gpus.split(",")
    if len(devices) != len(STEPS) or len(devices) != len(set(devices)):
        raise ValueError("Specify three different GPU IDs, one for each checkpoint")
    gpu_for_step = dict(zip(STEPS, devices))
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                        "PYTHONUNBUFFERED": "1", "XFORMERS_DISABLED": "1", "WANDB_CONSOLE": "off"})
    checkpoints = {step: Path(args.pretrain_dir).resolve() / "eval" / f"training_{step - 1}" / "teacher_checkpoint.pth"
                   for step in STEPS}
    if not all(path.is_file() for path in checkpoints.values()):
        raise FileNotFoundError("A requested teacher snapshot is missing")
    score_prefix = "test_as_dev" if args.dev_split == "test" else "dev"
    plan = {"steps": list(STEPS), "selection": f"best {score_prefix} P-I top1; ties prefer earlier pretrain step",
            "selection_split": args.dev_split, "independent_test": args.dev_split == "dev",
            "test": "used as dev for every epoch" if args.dev_split == "test" else "only selected checkpoint, best dev epoch, one test pass",
            "protocol": "SHuBERT-style rank1 LoRA, time mean + BN + Linear; SLiM RTMPose69 inputs",
            "gpus": args.gpus, "gpu_for_step": {str(s): g for s, g in gpu_for_step.items()},
            "layout": "three parallel single-GPU runs", "epochs": 125, "patience": 20, "seed": 0, "global_batch": 128,
            "head_lr": 1e-4, "lora_lr": 1e-5, "optimizer": "Adam", "weight_decay": 1e-4,
            "label_smoothing": 0.1, "lora_alpha": 1.0, "scheduler": "constant",
            "wandb_group": args.wandb_group, "checkpoints": {str(s): str(p) for s, p in checkpoints.items()},
            "data_path": str(Path(args.data_path).resolve())}
    plan_path = output / "comparison_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError("Output has a different comparison plan")
    write_json(plan_path, plan)

    def command(step, mode):
        return [sys.executable, "-m", "slim.eval.wlasl_lora",
                "--checkpoint", str(checkpoints[step]), "--pretrain-step", str(step),
                "--output-dir", str(output / f"pt{step}"), "--data-path", args.data_path, "--dev-split", args.dev_split,
                "--wandb-group", args.wandb_group, "--wandb-mode", args.wandb_mode, "--mode", mode]

    def environment_for(step):
        return dict(environment, CUDA_VISIBLE_DEVICES=gpu_for_step[step])

    def train_one(step):
        run_dir = output / f"pt{step}"
        run_dir.mkdir(exist_ok=True)
        summary_path = run_dir / "dev_summary.json"
        if not summary_path.exists():
            cmd = command(step, "train")
            if (run_dir / "config.json").exists():
                cmd.append("--resume")
            print(f"RUN GPU={gpu_for_step[step]} " + json.dumps(cmd), flush=True)
            with open(run_dir / "console.log", "a", buffering=1) as log:
                subprocess.run(cmd, cwd=ROOT, env=environment_for(step), stdout=log, stderr=subprocess.STDOUT, check=True)
        return json.loads(summary_path.read_text())

    summaries, failures = [], []
    write_json(output / "status.json", {"phase": "training", "active_steps": list(STEPS),
               "selection_split": args.dev_split, "independent_test": args.dev_split == "dev",
               "gpu_for_step": plan["gpu_for_step"], "completed_steps": [], "updated_unix": time.time()})
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(train_one, step): step for step in STEPS}
        for future in as_completed(futures):
            step = futures[future]
            try:
                summary = future.result()
                summaries.append(summary)
                summaries.sort(key=lambda s: s["pretrain_step"])
                write_json(output / f"{score_prefix}_comparison.json", summaries)
                print("DEV_RESULT " + json.dumps(summary), flush=True)
            except Exception as error:
                failures.append({"pretrain_step": step, "error": str(error)})
                print(f"FAILED pt={step} {error}", flush=True)
            completed = [s["pretrain_step"] for s in summaries]
            failed = [f["pretrain_step"] for f in failures]
            write_json(output / "status.json", {"phase": "training", "completed_steps": completed,
                       "selection_split": args.dev_split, "independent_test": args.dev_split == "dev",
                       "failed": failures, "active_steps": [s for s in STEPS if s not in completed + failed],
                       "updated_unix": time.time()})
    if failures:
        write_json(output / "status.json", {"phase": "failed", "failures": failures,
                   "completed_steps": [s["pretrain_step"] for s in summaries], "updated_unix": time.time()})
        raise RuntimeError("One or more adaptation runs failed; see per-checkpoint console.log")

    winner = max(summaries, key=lambda s: (s["best_dev"]["pi_top1"], -s["pretrain_step"]))
    selected_step = winner["pretrain_step"]
    write_json(output / "selection.json", {"pretrain_step": selected_step, "dev_result": winner,
                "selection": plan["selection"], "selection_split": args.dev_split,
                "independent_test": args.dev_split == "dev", "all_dev": summaries, "selected_unix": time.time()})
    if args.dev_split == "test":
        result = {"selection": plan["selection"], "selection_split": "test", "independent_test": False,
                  "test_as_dev_comparison": summaries, "selected_pretrain_step": selected_step,
                  "selected_epoch": winner["best_epoch"], "test_as_dev": winner["best_dev"]}
        write_json(output / "results.json", result)
        with open(output / "test_as_dev_comparison.csv", "w", newline="") as f:
            fields = ["pretrain_step", "best_epoch", "epochs_completed", "pi_top1", "pc_top1", "pi_top5", "pc_top5"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in summaries:
                writer.writerow({key: row[key] if key in row else row["best_dev"][key] for key in fields})
        write_json(output / "status.json", {"phase": "complete", "selected_pretrain_step": selected_step,
                   "selection_split": "test", "independent_test": False,
                   "completed_steps": list(STEPS), "updated_unix": time.time()})
        print("COMPLETE_TEST_AS_DEV " + json.dumps(result), flush=True)
        lock.close()
        return
    write_json(output / "status.json", {"phase": "final_test", "pretrain_step": selected_step,
               "completed_steps": list(STEPS), "updated_unix": time.time()})
    test_path = output / f"pt{selected_step}" / "test_metrics.json"
    if not test_path.exists():
        with open(output / f"pt{selected_step}" / "test_console.log", "a", buffering=1) as log:
            subprocess.run(command(selected_step, "test"), cwd=ROOT, env=environment_for(selected_step),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    test_result = json.loads(test_path.read_text())
    result = {"selection": plan["selection"], "dev_comparison": summaries,
              "selected_pretrain_step": selected_step, "final_test": test_result}
    write_json(output / "results.json", result)
    with open(output / "dev_comparison.csv", "w", newline="") as f:
        fields = ["pretrain_step", "best_epoch", "epochs_completed", "pi_top1", "pc_top1", "pi_top5", "pc_top5"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row[key] if key in row else row["best_dev"][key] for key in fields})
    # Attach the single final test result to the selected W&B run without training/test selection feedback.
    if args.wandb_mode == "online":
        import wandb
        run_meta = json.loads((output / f"pt{selected_step}" / "wandb_run.json").read_text())
        remote = wandb.Api().run(f"{run_meta['entity']}/{run_meta['project']}/{run_meta['id']}")
        for key, value in test_result["test"].items():
            remote.summary["test/" + key] = value
        remote.summary["selected_pretrain_step"] = selected_step
        remote.summary.update()
    write_json(output / "status.json", {"phase": "complete", "selected_pretrain_step": selected_step,
               "completed_steps": list(STEPS), "updated_unix": time.time()})
    print("COMPLETE " + json.dumps(result), flush=True)
    lock.close()


if __name__ == "__main__":
    main()
