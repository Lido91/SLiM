"""Pack RTMPose sign-language poses (Uni-Sign layout) into a memory-mapped SLiM sign pack.

Reads a Uni-Sign label file (gzip pickle: name -> {"video_path", "text", ...}) and the matching
COCO-WholeBody-133 pose pickles ({"keypoints": (T, 1, 133, 2), "scores": (T, 1, 133)}, optional
"start" / "end"), keeps the 69 Uni-Sign joints as raw (x, y, score) and writes

    <out>/poses.npy  <out>/offsets.npy  <out>/meta.json

Usage:
    python tools/pack_sign_poses.py --dataset CSL_Daily --split train
    python tools/pack_sign_poses.py --dataset WLASL --split test       # labels = WLASL-2000 classes
    python tools/pack_sign_poses.py --dataset OpenASL --split train --out /data/x/openasl_train

Presets resolve paths inside the Uni-Sign repo (--unisign-root, default $UNISIGN_ROOT or
/home/student/hwu/Workplace/Uni-Sign); --labels / --poses override them for other sources.
"""

import argparse
import gzip
import json
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slim.data.datasets.sign import PACK_FORMAT  # noqa: E402
from slim.data.skeletons import UNISIGN69_COCO_INDICES  # noqa: E402

PRESETS = {
    "CSL_Daily": dict(labels="data/CSL_Daily/labels.{split}", poses="dataset/CSL_Daily/pose_format"),
    "WLASL": dict(labels="data/WLASL/labels-2000.{split}", poses="dataset/WLASL/pose_format/{split}",
                  label_key="text"),
    "How2Sign": dict(labels="data/How2Sign/labels.{split}", poses="data/How2Sign/{split}-pose-rtmpose-192"),
    "OpenASL": dict(labels="data/OpenASL/labels.{split}", poses="data/OpenASL/pose-rtmpose-192"),
}


def load_labels(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def load_clip(pose_path):
    """-> float32 (T, 69, 3) raw (x, y, score), or an error string."""
    if not os.path.exists(pose_path):
        return "missing"
    with open(pose_path, "rb") as f:
        pose = pickle.load(f)
    kps = np.asarray(pose["keypoints"], dtype=np.float32)
    scores = np.asarray(pose["scores"], dtype=np.float32)
    if kps.ndim == 4:  # (T, person, 133, 2): Uni-Sign uses person 0
        kps, scores = kps[:, 0], scores[:, 0]
    start, end = pose.get("start", 0), pose.get("end", len(scores))
    kps, scores = kps[start:end, UNISIGN69_COCO_INDICES], scores[start:end, UNISIGN69_COCO_INDICES]
    if len(scores) == 0:
        return "empty"
    clip = np.concatenate([kps, scores[..., None]], axis=-1)
    bad = ~np.isfinite(clip).all(-1)
    clip[bad] = 0  # non-finite joint -> missing
    return clip


def _work(args):
    name, path = args
    return name, load_clip(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help=f"preset name ({', '.join(PRESETS)}) or any name with --labels/--poses")
    p.add_argument("--split", required=True, help="train / dev / test")
    p.add_argument("--out", help="output pack dir (default: ./data/sign/<dataset>/<split>)")
    p.add_argument("--unisign-root", default=os.environ.get("UNISIGN_ROOT", "/home/student/hwu/Workplace/Uni-Sign"))
    p.add_argument("--labels", help="label file, '{split}' is substituted (overrides the preset)")
    p.add_argument("--poses", help="pose pickle dir, '{split}' is substituted (overrides the preset)")
    p.add_argument("--label-key", help="label-file field holding the class name (WLASL preset: text)")
    p.add_argument("--vocab-split", default="train", help="split whose label file defines the class list")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--limit", type=int, default=0, help="pack only the first N clips (quick tests)")
    a = p.parse_args()

    preset = PRESETS.get(a.dataset, {})

    def resolve(pattern, split):
        path = pattern.format(split=split)
        return path if os.path.isabs(path) else os.path.join(a.unisign_root, path)

    labels_pattern = a.labels or preset.get("labels")
    poses_pattern = a.poses or preset.get("poses")
    if not labels_pattern or not poses_pattern:
        p.error(f"unknown dataset {a.dataset!r}: pass --labels and --poses")
    label_key = a.label_key or preset.get("label_key")
    out = a.out or os.path.join("data", "sign", a.dataset, a.split)

    labels_path = resolve(labels_pattern, a.split)
    pose_dir = resolve(poses_pattern, a.split)
    entries = load_labels(labels_path)
    names = list(entries)
    if a.limit:
        names = names[:a.limit]
    jobs = [(n, os.path.join(pose_dir, entries[n]["video_path"].replace(".mp4", ".pkl"))) for n in names]
    print(f"{a.dataset}/{a.split}: {len(jobs)} clips from {labels_path}, poses in {pose_dir}")

    clips, kept, errors = [], [], {}
    with Pool(a.workers) as pool:
        for i, (name, clip) in enumerate(pool.imap(_work, jobs, chunksize=16)):
            if isinstance(clip, str):
                errors[clip] = errors.get(clip, 0) + 1
            else:
                clips.append(clip)
                kept.append(name)
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{len(jobs)}")
    if errors:
        print(f"skipped: {errors}")
    if not clips:
        raise SystemExit("nothing to pack")

    classes, labels = None, None
    if label_key:
        vocab_entries = entries if a.vocab_split == a.split else load_labels(resolve(labels_pattern, a.vocab_split))
        classes = sorted({str(v[label_key]) for v in vocab_entries.values()})
        class_id = {c: i for i, c in enumerate(classes)}
        labels = [class_id.get(str(entries[n][label_key]), -1) for n in kept]
        unknown = sum(1 for lbl in labels if lbl < 0)
        print(f"{len(classes)} classes from the {a.vocab_split} labels"
              + (f"; {unknown} clips with an unseen class get label -1" if unknown else ""))

    os.makedirs(out, exist_ok=True)
    lengths = np.array([len(c) for c in clips], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    poses = np.lib.format.open_memmap(os.path.join(out, "poses.npy"), mode="w+", dtype=np.float32,
                                      shape=(int(offsets[-1]), len(UNISIGN69_COCO_INDICES), 3))
    for clip, o in zip(clips, offsets[:-1]):
        poses[o:o + len(clip)] = clip
    poses.flush()
    np.save(os.path.join(out, "offsets.npy"), offsets)

    meta = {
        "format": PACK_FORMAT,
        "layout": "unisign69",
        "coco_wholebody_indices": UNISIGN69_COCO_INDICES,
        "channels": ["x", "y", "score"],
        "dataset": a.dataset,
        "split": a.split,
        "labels_file": labels_path,
        "pose_dir": pose_dir,
        "num_clips": len(kept),
        "num_frames": int(offsets[-1]),
        "skipped": errors,
        "names": kept,
        "labels": labels,
        "classes": classes,
    }
    with open(os.path.join(out, "meta.json"), "w") as f:  # written last: its presence marks a complete pack
        json.dump(meta, f, ensure_ascii=False)
    print(f"wrote {out}: {len(kept)} clips, {offsets[-1]:,d} frames, "
          f"length mean {lengths.mean():.0f} / median {np.median(lengths):.0f}")


if __name__ == "__main__":
    main()
