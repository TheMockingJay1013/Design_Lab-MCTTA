# MC-TTA — Modality-Collaborative Test-Time Adaptation

PyTorch implementation of **Modality-Collaborative Test-Time Adaptation (MC-TTA)** for multimodal video action recognition (CVPR 2024: [Modality-Collaborative Test-Time Adaptation for Action Recognition](https://openaccess.thecvf.com/content/CVPR2024/html/Xiong_Modality-Collaborative_Test-Time_Adaptation_for_Action_Recognition_CVPR_2024_paper.html)). This repository is a course-style reproduction: dual-stream **I3D (RGB + optical flow)**, **teacher–student** adaptation with memory banks, **SSFR** (self-assembled source-friendly feature reconstruction), **multimodal prototype alignment**, and **cross-modal relative consistency**, without access to source video at test time (multimodal video test-time adaptation, MVTTA).

## What this project does

In MVTTA, only a **source-pretrained model** and **unlabeled target videos** are available for one-epoch adaptation. MC-TTA uses separate teacher and student memory banks to build pseudo-prototypes and target-prototypes, aligns them to reduce multimodal domain shift, and uses cross-modal consistency so one modality can stabilize another under shift. See the paper for full methodology.

## Installation

Requires **Python 3** and a **CUDA**-capable GPU for practical training (CPU is not supported in the provided commands). Install dependencies:

```bash
pip install -r requirements.txt
```

You may see a `torch.load` `FutureWarning` about `weights_only` when loading checkpoints; for **trusted local** `.pth` files this is expected and safe to ignore unless you harden loading for untrusted files.

## Data and pretrained weights (not in Git)

Large assets are **excluded** from this repository (see [.gitignore](.gitignore)): `downloads/`, `checkpoints/`, `pretrained/`, and benchmark video trees under `data/ucf_hmdb_full/` (and related split names).

| Asset | How to obtain |
|--------|----------------|
| **UCF-HMDB_full** (or full processed `data_root` tree) | **Google Drive ([add your link here](https://drive.google.com/drive/folders/1o4gJfjZYhJ140uKcUGR6bLIoVtnA-VFX?usp=sharing)):** `________________________` |
| **I3D Kinetics** RGB / flow checkpoints (`rgb_kinetics.pt`, `flow_kinetics.pt`) | Follow [docs/I3D_PRETRAINING.md](docs/I3D_PRETRAINING.md) and [scripts/convert_kinetics_i3d_to_pytorch.py](scripts/convert_kinetics_i3d_to_pytorch.py) |
| **Source checkpoint** after pretraining | Run pretraining below; output is written to `./checkpoints/` (local only) |

**Expected `data_root` layout** (used by [data/video_dataset.py](data/video_dataset.py)):

```text
data_root/
  <domain>/          # e.g. ucf, hmdb
    train/<class>/<video>.avi
    test/<class>/<video>.avi
```

After unpacking, verify counts:

```bash
python scripts/verify_dataset.py --data_root ./data/ucf_hmdb_full --domain ucf --split train
```

**Optional — build from official sources** instead of Drive: [scripts/setup_dataset.py](scripts/setup_dataset.py) (downloads and organizes UCF101 / HMDB51 into the TA3N-style split; may use `downloads/` as a staging area).

## Repository layout

| Path | Role |
|------|------|
| [config/default.yaml](config/default.yaml) | Paper-aligned defaults (e.g. λ₁=1, λ₂=0.5, `accum_steps=64`, SSFR α/β). |
| [config/adapt_stable.yaml](config/adapt_stable.yaml) | Conservative adaptation (smaller λ, normalized L_ma, grad clip) for stable runs. |
| [engine.py](engine.py) | Teacher–student loop, EMA, adaptation step. |
| [mctta/](mctta/) | SSFR, memory banks, L_CE / L_ma / L_cmr. |
| [models/](models/) | I3D, MLP heads, `SourceModel`, Kinetics weight loading. |
| [data/](data/) | `VideoDataset`, TV-L1 flow (`flow_method: tvl1`). |
| [scripts/pretrain.py](scripts/pretrain.py) | Source-domain pretraining. |
| [scripts/adapt.py](scripts/adapt.py) | One-epoch TTA; logs source-only, teacher pseudo-acc during adapt, post student & teacher multimodal acc. |
| [scripts/evaluate.py](scripts/evaluate.py) | Checkpoint evaluation. |
| [tests/](tests/) | `pytest` for core components. |

## Pretraining (source: UCF)

From the repository root:

```bash
cd /path/to/Design_Lab-MCTTA   # or your clone of this repo

python scripts/pretrain.py \
  --config config/default.yaml \
  --source_domain ucf \
  --data_root ./data/ucf_hmdb_full \
  --output_dir ./checkpoints \
  --i3d_rgb_pretrained ./pretrained/rgb_kinetics.pt \
  --i3d_flow_pretrained ./pretrained/flow_kinetics.pt \
  --batch_size 4 \
  --device cuda
```

Adjust `--data_root`, `--output_dir`, and `--i3d_*` paths to match your machine.

## Adaptation (target: HMDB, stable config)

Uses the source checkpoint from pretraining and [config/adapt_stable.yaml](config/adapt_stable.yaml) (`accum_steps=64`, λ₁=0.2, λ₂=0.1, etc.):

```bash
cd /path/to/Design_Lab-MCTTA

python scripts/adapt.py \
  --config config/adapt_stable.yaml \
  --source_checkpoint ./checkpoints/source_ucf_best.pth \
  --target_domain hmdb \
  --data_root ./data/ucf_hmdb_full \
  --device cuda
```

**Metrics logged:** source-only (student multimodal); during adaptation, accuracy of **teacher pseudo-labels**; after adaptation, **student** and **teacher (EMA)** multimodal accuracy on the target split.

### SSFR toggles (`config` → `ssfr:`)

- `pairwise_metric`: `cosine_distance` (default) or `cosine_similarity_pdf`.
- `entropy_mode`: `low_entropy_threshold` (default) or `high_entropy_threshold`.

## Results (reproduction run, UCF → HMDB, UCF-HMDB full)

The following matches a local run with `adapt_stable.yaml` (not the paper’s default λ₁=1.0, λ₂=0.5 in `default.yaml`). Numbers are **project reproduction**, not a claim to match the published Table 1 benchmarks (e.g. 88.61% / 91.07% on U→H / H→U in the paper under their full protocol).

**Table 1–style summary (target HMDB, multimodal accuracy, %).**

| Methods | U→HMDB |
|---------|--------|
| Source-only | 73.33 |
| During adaptation (teacher pseudo-label accuracy) | 74.72 |
| MC-TTA — post-adapt student | **75.56** |
| MC-TTA — post-adapt teacher (EMA) | 74.72 |

**One-line summary:** `source_only=73.33% | post_student=75.56% | post_teacher=74.72%`

## Citation

```bibtex
@inproceedings{xiong2024mctta,
  title={Modality-Collaborative Test-Time Adaptation for Action Recognition},
  author={Xiong, Baochen and Yang, Xiaoshan and Song, Yaguang and Wang, Yaowei and Xu, Changsheng},
  booktitle={CVPR},
  year={2024}
}
```
