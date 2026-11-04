> **Sign-language fork:** this checkout includes RTMPose sign-data support and an
> experimental hierarchical **SLiM Sign V2** (`student.arch: slim_v2`). V2 is a local
> extension of the [original SLiM](https://github.com/KAIST-VICLab/SLiM), with a different encoder/checkpoint architecture.
> See [SLiM Sign V2](#slim-sign-v2) for implementation, validation and commands.

<div align="center">
<h2>[NeurIPS 2026] Less is More: Compact-Token Masked Feature Prediction for Skeleton Representation Learning</h2>

<div>    
    <a href='https://jeonghyeokdo.github.io/' target='_blank'>Jeonghyeok Do</a><sup>1</sup>&nbsp;&nbsp;&nbsp;&nbsp;
    <a href='https://therealchenyun.github.io/' target='_blank'>Yun Chen</a><sup>1</sup>&nbsp;&nbsp;&nbsp;&nbsp;
    <a href='https://geunhyukyouk.github.io/' target='_blank'>Geunhyuk Youk</a><sup>1</sup>&nbsp;&nbsp;&nbsp;&nbsp;
    <a href='https://www.viclab.kaist.ac.kr' target='_blank'>Munchurl Kim</a><sup>1†</sup>
</div>
<br>
<div>
    <sup>†</sup>Corresponding author
</div>
<div>
    <sup>1</sup>Korea Advanced Institute of Science and Technology, South Korea
</div>

<div>
    <h4 align="center">
        <a href="https://kaist-viclab.github.io/SLiM_site/" target='_blank'>
        <img src="https://img.shields.io/badge/🏠-Project%20Page-blue">
        </a>
        <img src="https://img.shields.io/badge/2026-NeurIPS-brightgreen">
        <a href="https://arxiv.org/abs/2603.10648" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-2603.10648-b31b1b.svg">
        </a>
        <a href="https://youtu.be/cePWsYhS08E" target='_blank'>
        <img src="https://img.shields.io/badge/Presentation-%23FF0000.svg?logo=YouTube&logoColor=white">
        </a>
        <a href="https://huggingface.co/JeonghyeokDo/SLiM" target='_blank'>
        <img src="https://img.shields.io/badge/🤗-Models-yellow">
        </a>
        <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/KAIST-VICLab/SLiM">
    </h4>
</div>
</div>

---

Official PyTorch implementation of **"Less is More: Compact-Token Masked Feature Prediction for Skeleton Representation Learning"**.

**SLiM (Skeleton Less is More)** is a unified self-supervised framework that combines masked feature prediction (MFP) and contrastive learning (CL) on a compact token grid.

* 🚀 **Extreme Efficiency:** A completely decoder-free architecture that reduces inference computational costs by **7.89×**.
* 🧠 **Robust Representation:** Introduces **Semantic Tube Masking (STM)** and **Skeleton-Aware Augmentations (SAA)** to capture anatomically consistent motion dynamics without shortcut learning.
* 🏆 **State-of-the-Art:** Best linear-evaluation accuracy on NTU-60 and NTU-120 (second best on PKU-MMD II), the best 1% semi-supervised results, and the best retrieval accuracy.

---

## SLiM: Decoder-Free Unified Framework
![motivation](assets/motivation.jpg)

---

## Overview of SLiM Framework
![overview](assets/overview.jpg)

---

## Less Cost, More Accuracy
Token budget and computational cost on NTU-60 (GFLOPs), with NTU-60 linear-evaluation accuracy (%).

| Method | Venue | Tokens (N<sub>T</sub> × N<sub>J</sub>) | Inference: Encoder | Training: Encoder | Training: Decoder | X-Sub | X-View |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SkeletonMAE | ICMEW'23 | 30 × 25 | 28.32 | 1.97 | 17.70 | 74.8 | 77.7 |
| MAMP | ICCV'23 | 30 × 25 | 28.32 | 1.97 | 17.70 | 84.9 | 89.1 |
| S-JEPA | ECCV'24 | 30 × 25 | 28.32 | 1.97 | 17.70 | 85.3 | 89.8 |
| GFP | ICCV'25 | 30 × 25 | 28.32 | 1.97 | 1.57 | 85.9 | 92.0 |
| **SLiM (Ours)** | – | 8 × 25 | **3.59** | 3.59 | – | **87.9** | **93.2** |

---

## 📧 News
- **Sep 25, 2026:** SLiM accepted to NeurIPS 2026. :tada:
- **Aug 4, 2026:** Code and NTU-60 pre-trained weights released
- **Mar 11, 2026:** This repository is created

---

## Method

SLiM patchifies a 64-frame clip with `P_T = 8`, `P_J = 1` into a compact `8 x 25` token grid and
trains a ViT encoder (8 blocks, dim 256, 8 heads) with two objectives on top of an EMA teacher:

* **MFP** — Masked Feature Prediction: masked student tokens predict the teacher's prototype
  assignment for the same positions. No coordinate decoder is used.
* **GLCL** — Global-Local Contrastive Learning: cross-view prototype matching over CLS tokens.
* **STM** — Semantic Tube Masking: masks are anatomically coherent spatio-temporal tubes rather
  than independent joints.
* **SAA** — Skeleton-Aware Augmentations: bone-aware scaling, skeleton-aware mirroring and rotation.

Naming note: the code descends from DINOv2, so MFP is implemented in the iBOT patch branch
(`ibot.*` config keys, `ibot_head`) and GLCL in the DINO CLS branch (`dino.*`, `dino_head`).

## Installation

```bash
git clone https://github.com/Lido91/SLiM.git
cd SLiM
conda create -n slim python=3.9 -y
conda activate slim
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Tested with Python 3.9, PyTorch 2.0.0, CUDA 11.8, xFormers 0.0.19 on 4x NVIDIA RTX A6000.
xFormers is optional: without it the temporal-RoPE attention falls back to PyTorch
`scaled_dot_product_attention` (same math), and the code also runs on PyTorch >= 2.1 (FSDP
stream / reshard API changes are handled). Loading weights from HuggingFace Hub additionally needs
`pip install huggingface_hub safetensors`.

## Data

Place the preprocessed `.npz` files under `./data` (or point `DATA_ROOT` elsewhere):

```
data/
├── ntu/
│   ├── NTU60_XSub.npz
│   └── NTU60_XView.npz
└── ntu120/
    ├── NTU120_CS.npz     # X-Sub
    └── NTU120_CV.npz     # X-Set
```

Each file holds the standard NTU arrays:

| key | shape | dtype |
| --- | --- | --- |
| `x_train` / `x_test` | `(N, 300, 150)` — 300 frames x (2 people x 25 joints x 3 coords) | float32 |
| `y_train` / `y_test` | `(N, num_classes)` one-hot | int |

The dataset reshapes this to `(N, 3, 300, 25, 2)` internally. Raw NTU RGB+D 60/120 skeletons are
available from [ROSE Lab](https://rose1.ntu.edu.sg/dataset/actionRecognition/); this is the same
preprocessed layout used by CrosSCLR / AimCLR / CMD.

## Pre-trained weights

Encoder-only checkpoints (6.37 M parameters, ~25 MB). The prototype heads are discarded, as they
are at transfer time. One checkpoint per evaluation protocol — the evaluated pre-training
checkpoint with the highest linear accuracy on that protocol — with that protocol's trained
linear probe under `linear/`.

All checkpoints live in one repository, [JeonghyeokDo/SLiM](https://huggingface.co/JeonghyeokDo/SLiM),
one subfolder per protocol:

| subfolder | Protocol | Epoch | Top-1 |
| --- | --- | ---: | ---: |
| `ntu60_xsub`  | NTU-60 X-Sub  | 120 | 87.9 |
| `ntu60_xview` | NTU-60 X-View | 150 | 93.2 |

```python
from slim.hub import SLiMEncoder

model = SLiMEncoder.from_pretrained("JeonghyeokDo/SLiM", subfolder="ntu60_xsub").eval().cuda()
feats = model.get_intermediate_layers(clips, 4, return_class_token=True)  # clips: (B, 3, 64, 25, M)
```

`from_pretrained` also takes a local directory holding `config.json` and `model.safetensors`, so
it works on your own exports (see [Pre-training](#pre-training)). Sanity-check any checkpoint —
downloaded or your own — with:

```bash
python tools/verify_checkpoints.py       # needs a GPU — see the xFormers note above
```

### Trained linear probes

Each subfolder carries the linear-classifier grid trained for the linear-evaluation table under
`linear/`. Download it into `checkpoints/linear/` and the encoder into `checkpoints/`;
evaluating the pair reproduces the reported accuracy without retraining the probe:

```bash
CLASSIFIER=checkpoints/linear/slim_ntu60_best_xsub.pth \
  scripts/eval_linear.sh checkpoints/slim_ntu60_best_xsub.pth outputs/lin_ntu60_xsub ntu60 xsub
```

| Classifier | Encoder | Protocol | Top-1 |
| --- | --- | --- | ---: |
| `slim_ntu60_best_xsub.pth`  | `slim_ntu60_best_xsub.pth`  | NTU-60 X-Sub  | 87.88 |
| `slim_ntu60_best_xview.pth` | `slim_ntu60_best_xview.pth` | NTU-60 X-View | 93.24 |

Retraining a probe from scratch instead lands within ~0.25 points of these values — probe training
depends on data order and on fp16/cuDNN algorithm selection, so small differences between
environments are expected. k-NN retrieval needs no classifier and runs at 72.3 (X-Sub) and
89.8 (X-View) on NTU-60 from the released encoder weights alone.

## Pre-training

Pre-train once per evaluation protocol, on that protocol's **training** clips only (`split=TRAIN`,
labels unused). The protocol's test clips are therefore never seen before evaluation.

```bash
DATA_ROOT=./data scripts/pretrain.sh ntu60_xsub   outputs/slim_ntu60_xsub   0,1,2,3
DATA_ROOT=./data scripts/pretrain.sh ntu60_xview  outputs/slim_ntu60_xview  0,1,2,3
DATA_ROOT=./data scripts/pretrain.sh ntu120_xsub  outputs/slim_ntu120_xsub  0,1,2,3
DATA_ROOT=./data scripts/pretrain.sh ntu120_xset  outputs/slim_ntu120_xset  0,1,2,3
```

| Run | Config | `.npz` | Pre-training clips |
| --- | --- | --- | ---: |
| `ntu60_xsub`  | `slim_ntu60_xsub.yaml`  | `ntu/NTU60_XSub.npz`    | 40,091 |
| `ntu60_xview` | `slim_ntu60_xview.yaml` | `ntu/NTU60_XView.npz`   | 37,646 |
| `ntu120_xsub` | `slim_ntu120_xsub.yaml` | `ntu120/NTU120_CS.npz`  | 63,026 |
| `ntu120_xset` | `slim_ntu120_xset.yaml` | `ntu120/NTU120_CV.npz`  | 54,468 |

Evaluate each encoder on its own protocol only — an encoder pre-trained on the X-Sub training
split has seen clips that appear in the X-View test split, and vice versa.

An epoch is a fixed 1250 iterations (`train.OFFICIAL_EPOCH_LENGTH`) rather than one pass over the
data, so the schedule below and the total compute are the same for every run above; only the
number of passes over the pre-training set changes.

150 epochs, AdamW, base LR `2e-4` with `sqrt_wrt_1024` scaling, 20-epoch warmup, cosine decay to
`1e-6`, effective batch 768 (4 GPUs x 192). Training resumes automatically from the last
checkpoint in the output directory. Teacher checkpoints are written to
`outputs/<run>/eval/training_<iter>/teacher_checkpoint.pth` every 12,500 iterations; export the
encoder from one with:

```bash
python tools/export_encoder.py \
    outputs/slim_ntu60_xsub/eval/training_187499/teacher_checkpoint.pth \
    checkpoints/ntu60_xsub.pth
```

If you change the number of GPUs, adjust `train.batch_size_per_gpu` so the effective batch stays
768, otherwise the scaled learning rate will not match the paper.

## Evaluation

```bash
# Linear evaluation
scripts/eval_linear.sh checkpoints/slim_ntu60_best_xsub.pth  outputs/lin_ntu60_xsub   ntu60  xsub
scripts/eval_linear.sh checkpoints/slim_ntu60_best_xview.pth outputs/lin_ntu60_xview  ntu60  xview
scripts/eval_linear.sh checkpoints/ntu120_xsub.pth  outputs/lin_ntu120_xsub  ntu120 xsub
scripts/eval_linear.sh checkpoints/ntu120_xset.pth  outputs/lin_ntu120_xset  ntu120 xview

# Semi-supervised fine-tuning, 1 % and 10 % of the labels (2 GPUs)
scripts/eval_semi.sh checkpoints/slim_ntu60_best_xsub.pth outputs/semi_001_xsub ntu60 xsub 0.01
scripts/eval_semi.sh checkpoints/slim_ntu60_best_xsub.pth outputs/semi_010_xsub ntu60 xsub 0.1

# Action retrieval, k-NN
scripts/eval_knn.sh checkpoints/slim_ntu60_best_xsub.pth outputs/knn_ntu60_xsub ntu60 xsub
```

`xview` selects X-View on NTU-60 and X-Set on NTU-120. Accuracies are appended to
`<output_dir>/results_eval_linear.json` after every evaluation period.

**Protocols.** Following DINOv2, linear evaluation feeds the frozen encoder into a grid of
linear classifiers — {1, 4} last blocks x {with, without} average-pooled patch tokens x 13
learning rates — trained for 10 epochs x 2000 iterations with SGD (momentum 0.9, batch 16 per
GPU); the best classifier of the grid is reported. Person features are summed over the valid
people in a clip, with padded people masked out. Semi-supervised evaluation fine-tunes the whole
encoder with a single head (last block + avgpool), head LR `1e-3` and backbone LR `1e-5`.

Learning rates are scaled by `batch_size * world_size / 256`, so the **number of GPUs is part of
the recipe**: linear evaluation uses 4 GPUs x batch 16, semi-supervised fine-tuning 2 GPUs x
batch 128. The default GPU lists in the scripts already match; change them and the effective
learning rates change with them.

## Sign-language pose (RTMPose, Uni-Sign layout)

The same framework runs on 2D sign-language pose: RTMPose COCO-WholeBody-133 keypoints as used
by Uni-Sign (CSL-Daily, WLASL, How2Sign, OpenASL pose pickles). Nothing changes on the NTU path.

| | NTU (original) | Sign pose |
| --- | --- | --- |
| Joints | Kinect-25, 3D | 69 = body 9 + hands 2 x 21 + face 18 (`sign69`), or 51 without face (`sign51`) |
| Channels | x, y, z | x, y, RTMPose score (missing joints are all-zero) |
| Normalisation | origin at spine | Uni-Sign `load_part_kp`: body in its bbox, hands wrist-relative, face nose-relative (`pose_norm: part`), or one body-bbox frame (`global`) |
| Token grid | 64 frames, P_T = 8 -> 8 x 25 | 64 frames, P_T = 4 -> 16 x 69 |
| STM groups | trunk, arms, legs | head, 2 arms, 2 hands (COCO finger order), jaw, mouth; groups drawn with sqrt(size) weights so every part gets the target mask rate |
| SAA | 3D rotation, limb scaling, L/R mirror | in-plane rotation (+-0.2 rad), bone scaling on hand / arm / face trees, L/R mirror incl. hands and face contour; part-drop (a hand or the face) replaces axis-drop |
| Long clips | - | the two global views come from one random <= 128-frame window (`crops.max_window_frames`) |
| MFP prototypes | 65,536 | 16,384 (1,104 tokens per view) |
| Downstream | NTU-60/120 | WLASL-2000 linear / semi / k-NN (classifier picked on dev, reported on test) |

The encoder is the same `vit_tiny` (6.38 M parameters); temporal positions are RoPE, so it also
takes full-length clips (any multiple of P_T frames) at inference.

**Environment.** On top of an existing PyTorch env (e.g. Uni-Sign's torch 2.1), without touching it:

```bash
python -m venv --system-site-packages .venv
.venv/bin/pip install omegaconf==2.3.0 fvcore==0.1.5.post20221221 torchmetrics==0.10.3
printf '#!/bin/sh\nexec "$(dirname "$0")/python" -m torch.distributed.run "$@"\n' > .venv/bin/torchrun
chmod +x .venv/bin/torchrun && source .venv/bin/activate
```

**Data.** Pack each split once into a memory-mapped `data/sign/<dataset>/<split>/` (paths default
to the Uni-Sign repo layout, `--unisign-root` / `--labels` / `--poses` override them):

```bash
for sp in train dev test; do python tools/pack_sign_poses.py --dataset WLASL --split $sp; done      # 2000 classes
for sp in train dev test; do python tools/pack_sign_poses.py --dataset CSL_Daily --split $sp; done
python tools/pack_sign_poses.py --dataset How2Sign --split train
python tools/pack_sign_poses.py --dataset OpenASL --split train
```

**Pre-training** (training clips only, no labels / text):

```bash
scripts/pretrain_sign.sh wlasl     outputs/slim_sign_wlasl  0,1,2,3   # WLASL train
scripts/pretrain_sign.sh csl_daily outputs/slim_sign_csl    0,1,2,3   # CSL-Daily train
scripts/pretrain_sign.sh openasl   outputs/slim_sign_openasl 0,1,2,3  # OpenASL train
scripts/pretrain_sign.sh asl       outputs/slim_sign_asl    0,1,2,3   # How2Sign + OpenASL + WLASL train
```

The configs use 64 clips per GPU: peak memory grows ~0.58 GB per clip at the default grid
(measured at batch 8 / 16 on an L40S), so ~38 GB at 64. With less memory lower
`train.batch_size_per_gpu` (the LR follows the sqrt rule), or use `student.patch_size=[8,1]` for the
paper's 8 x V grid at about half the cost.

Enable Weights & Biases logging with the optional `wandb` package installed and an account
logged in (`python -m wandb login`):

```bash
scripts/pretrain_sign.sh openasl outputs/slim_sign_openasl 0,1,2,3 \
  wandb.enabled=true wandb.project=SLiM
```

Only rank 0 creates a W&B run. Losses, learning rates, teacher momentum/temperature, batch
sizes, timing and peak memory are recorded every 10 iterations, with the same values as the
local `training_metrics.json`. `wandb_run.json` stores the run URL and ID for resuming. To start
from scratch, use a new output directory (or archive the old one) and pass `--no-resume`
before any `key=value` overrides. Model checkpoints remain on local storage.

**Evaluation** on WLASL-2000 (the benchmark argument is ignored, pass `-`):

```bash
python tools/export_encoder.py outputs/slim_sign_wlasl/eval/training_187499/teacher_checkpoint.pth checkpoints/sign_wlasl.pth
scripts/eval_linear.sh checkpoints/sign_wlasl.pth outputs/lin_sign_wlasl  wlasl - 0,1,2,3
scripts/eval_knn.sh    checkpoints/sign_wlasl.pth outputs/knn_sign_wlasl  wlasl - 0,1,2,3
scripts/eval_semi.sh   checkpoints/sign_wlasl.pth outputs/semi_sign_wlasl wlasl - 0.1 0,1
```

**SHuBERT-style WLASL-2000 LoRA comparison** (full WLASL training split):

```bash
.venv/bin/python scripts/eval_wlasl_lora_compare.py \
  --output-dir outputs/wlasl_lora_r1_62500_75000_87500 \
  --gpus 0,1,2
```

This runs three experiments concurrently, one per GPU, for the teacher snapshots at
62,500, 75,000 and 87,500 updates. Each run uses
rank-1 LoRA on every encoder Linear, with the original encoder weights frozen. The last
patch representations are averaged over joints and time, then classified by BN + Linear.
The [SHuBERT ISLR protocol](https://arxiv.org/html/2411.16765v3#S4.SS3) supplies Adam,
global batch 128, at most 125 epochs, head LR 1e-4, LoRA LR 1e-5, weight decay 1e-4 and
label smoothing 0.1. Implementation choices not specified in the paper are alpha=1,
no adapter dropout, constant learning rates, patience=20, and the existing SLiM sign
augmentations with 64-frame inputs. This adapts the evaluation recipe to RTMPose69
SLiM; it does not reproduce SHuBERT's RGB-feature encoder.

All three models are selected and compared using dev P-I Top-1. The winner alone is
then evaluated once on test, reporting per-instance/per-class Top-1/5/10. Evaluation
never pads the split or duplicates videos, and macro accuracy averages present classes.
Multi-GPU training wraps the complete model in DDP and uses SyncBatchNorm. The output
contains checkpoint/data/code fingerprints, per-epoch metrics, W&B run links, and atomic
`best.pth` / `last.pth` files with the full encoder, adapters, head, optimizer, scaler and
per-rank RNG states. Repeating the same command resumes unfinished runs at the last
completed epoch and skips finished runs. `results.json` and `dev_comparison.csv` hold
final results. The original pretraining remains stopped.

To use the test split directly for model selection, add `--dev-split test` and use a
fresh output directory and W&B group. This evaluates all 2,878 test clips every epoch
for early stopping and best-epoch selection, without evaluating the dev split. Metrics
are logged as `test_as_dev/*`; `test_as_dev_comparison.csv` and `results.json` explicitly
mark that no independent test score remains. There is no additional final test pass.


Feature extraction for other uses: `slim.hub.load_encoder_from_pth(path, {"img_size": [64, 69],
"patch_size": [4, 1]})`, input `(B, 3, T, 69, 1)` normalised with `slim.data.skeletons.normalize_sign_pose`.

---

## Repository layout

```
slim/
├── configs/            default_config.yaml + per-dataset pre-training and eval configs
├── data/
│   ├── augmentations.py    SAA + view sampling (DataAugmentationDINO, DownstreamAugmentation)
│   ├── masking.py          STM (MaskingGenerator)
│   ├── collate.py          view stacking and mask assignment
│   ├── skeletons.py        sign-pose joint layouts, Uni-Sign normalisation
│   ├── sign_augmentations.py  2D sign-pose SAA
│   └── datasets/           ntu.py (NTU), sign.py (packed sign-pose clips)
├── layers/             ViT blocks, temporal-RoPE attention, prototype head
├── loss/               MFP (iBOTPatchLoss), GLCL (DINOLoss), KoLeo
├── models/             SLiM encoder (vit_tiny: 8 blocks, dim 256, 8 heads)
├── train/              SSLMetaArch (student/teacher, EMA) and the training loop
├── eval/               linear.py, finetune.py (semi-supervised), knn.py (retrieval), datasets.py
└── hub.py              HuggingFace Hub loading
scripts/                pre-training and evaluation launchers
tools/                  checkpoint export and verification, pack_sign_poses.py
checkpoints/            exported encoders; linear/ holds trained linear probes
```

---

## Results
Please visit our [project page](https://kaist-viclab.github.io/SLiM_site/) for more experimental results.

## Reference
```BibTeX
@inproceedings{do2026less,
  title={Less is More: Compact-Token Masked Feature Prediction for Skeleton Representation Learning},
  author={Do, Jeonghyeok and Chen, Yun and Youk, Geunhyuk and Kim, Munchurl},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```

## Acknowledgements
This codebase is built upon [DINOv2](https://github.com/facebookresearch/dinov2).

## License
Apache License 2.0, inherited from DINOv2. See [LICENSE](LICENSE).


## SLiM Sign V2

V2 retains the original sign69 data layout (body 9, left/right hands 21 each, face 18)
and the 4-frame temporal patch. Its data flow is:

```text
[B, 3, 64, 69, 1] normalized pose + confidence
  -> 16 x 69 joint embeddings (256 dimensions)
  -> shared spatial attention within each part/time patch
  -> learned query pooling: body 4, left hand 8, right hand 8, face 8
  -> 16 x 28 = 448 summaries -> 8-layer global Transformer with temporal RoPE
  -> global part context fused back into all 16 x 69 joint features
  -> validity-aware, part-balanced joint pooling + CLS -> 256D downstream embedding
```

The 1,104 joint features remain available as `x_norm_patchtokens` for joint-level
MFP; `x_norm_parttokens` contains the 448 summaries. Only summaries, CLS and four
register tokens enter global attention. Four parts share a local attention block,
processed in one padded batch; joint positions and part queries distinguish them.
This retains a direct fine-feature path, but does **not** establish that compression
is lossless or improves sign recognition. The existing 18 face points cover jaw,
inner lips and nose tip; eyebrows/eye landmarks are not in this input layout.

**Training.** An EMA teacher sees unmasked global crops. The student masks joint
patches **before local spatial mixing**, so masked coordinates cannot leak through
pooling. Detection validity excludes invalid coordinates from patch embedding,
attention, pooling and MFP targets. Globals overlap by at least 50%; each local
crop matches only its parent global, with crop-major sample identity preserved.
The default V2 local crops are unmasked. Confidence-based validity remains distinct
from artificial masking; a partially detected temporal patch is usable when at
least one constituent frame contains a detected joint.

The loss is `DINO + joint MFP/iBOT + 0.1 * KoLeo`. V2 uses stable fp32 distributed
Sinkhorn and per-crop mean MFP, including ranks with no valid local targets.
The default `ibot.max_mfp_tokens_per_view=128` uniformly samples up to 128 valid
masked **joint prediction targets** per global view. All drawn masks remain hidden
from the student, including masks not selected as targets. Set the budget to `0`
for all-target supervision. This is an explicit stochastic supervision budget,
not a change from joint-level targets to part-only targets. Gentler sign noise
(`crops.noise_std=0.002`) is used in V2. V1 configs remain available.

Pack OpenASL train as described above, then start a **new** output directory:

```bash
bash scripts/pretrain_sign_v2.sh outputs/slim_sign_v2_openasl 0,1,2,3   wandb.enabled=true wandb.project=SLiM wandb.name=slim_sign_v2_openasl
```

For a bounded smoke run, append `train.max_steps=20`; the learning-rate/teacher
schedules still use the declared full training budget. Repeating the same command
resumes the saved iteration, optimizer, scaler, per-rank RNG and sampler position.
V2 guards the architecture, loss/crop settings, schedule, seed, batch, world size
and store metadata fingerprints. Worker-prefetch augmentation state is not replayed
bit-for-bit. V1 checkpoints cannot resume V2; use fresh pretraining for this encoder.
The output/checkpoint format used by V1 is otherwise preserved.

Teacher exports include `encoder_config`; `tools/export_encoder.py` preserves it:

```bash
python tools/export_encoder.py   outputs/slim_sign_v2_openasl/eval/training_12499/teacher_checkpoint.pth   checkpoints/slim_sign_v2.pth
```

```python
from slim.hub import load_encoder_from_pth
encoder = load_encoder_from_pth("checkpoints/slim_sign_v2.pth").eval().cuda()
# clips: [B,3,T,69,1], with T divisible by 4
embedding = encoder.forward_features(clips)["x_norm_embedding"]  # [B,256]
```

`slim.eval.wlasl_lora` automatically loads V2 metadata and uses this embedding for
its BN + Linear classifier. Example using the existing test-as-dev diagnostic:

```bash
python -m slim.eval.wlasl_lora --checkpoint checkpoints/slim_sign_v2.pth   --pretrain-step 12500 --output-dir outputs/wlasl_lora_v2   --dev-split test --wandb-group wlasl-lora-v2
```

Use `slim/configs/eval/slim_sign_v2_eval.yaml` with the generic evaluation tools.
`get_intermediate_layers` returns the context-fused joint features at selected
global blocks. The dedicated WLASL classifier uses validity-aware V2 pooling;
the older generic evaluation heads retain their own configured pooling.

**Validation (2026-11-04).** 25 CPU unit checks pass, including pre-mixing mask
isolation, missing-data/padding invariance, empty-target loss, EMA, sampler resume
across ranks/epochs, configuration guards, checkpoint export/load and LoRA forward.
Bounded real OpenASL GPU checks passed: one GPU for 2 updates then resume to 4;
two GPUs for 2 updates. GPU checks used the full encoder and smaller prototype
heads for smoke testing. No formal V2 pretraining or WLASL accuracy run has started.

Single L40S compute benchmark on synthetic, already-collated crops: bf16, batch 16,
2 globals + 8 locals, 2 warmup and
7 timed updates; actual teacher/student, heads, backward, AdamW and EMA:

| Encoder / supervision | Median step | Peak allocated GPU memory |
| --- | ---: | ---: |
| V1, all masked joint targets | 0.390 s | 11,592 MiB |
| V2, all masked joint targets | 0.316 s | 8,827 MiB |
| V2, default 128-target budget | 0.199 s | 5,772 MiB |

The V2 default measured about **1.96x** compute throughput and **50%** less allocated
memory. Architecture alone measured 1.23x with all-target supervision. Data loading,
checkpoint I/O and multi-GPU scaling are excluded; shared-GPU load introduces timing
noise. Batch 2 is dominated by overhead and did not improve in this short benchmark.
This is implementation/performance evidence, not downstream-quality evidence.
Reproduce with `python tools/benchmark_sign_v2.py --batch-size 16 --output /tmp/slim_v2_benchmark.json`.
