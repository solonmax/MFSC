# MFSC + TUA: Multi-scale Frequency–Spatial Cooperation with Task-specific Uncertainty Allocation

**Official PyTorch implementation of**  
**"MFSC + TUA for Universal Image Restoration"**  
(*submitted to XXXX, 2025*)

---

## 🧩 Introduction

This project provides the official implementation of **MFSC (Multi-scale Frequency–Spatial Cooperation)** and **TUA (Task-specific Uncertainty Allocation)** for **universal image restoration**, covering deraining, dehazing, desnowing, deblurring and low-light enhancement within a single unified framework.

MFSC injects *multi-band spectral priors* via adaptive frequency–spatial projection, and TUA dynamically modulates per-task residual uncertainty to stabilize cross-degradation generalization.

---

## 📦 Environment

```bash
conda create -n mfsc python=3.10 -y
conda activate mfsc
pip install -r requirements.txt
We tested on Ubuntu 22.04 + CUDA 11.8 + PyTorch 2.2.
python test.py \
    --dataroot ./datasets \
    --phase test \
    --pretrained ./checkpoints/mfsc_tua.pth

Datasets
We follow standard evaluation splits:
| Task       | Dataset                        |
| ---------- | ------------------------------ |
| Deraining  | Rain100L / Rain100H / Test2800 |
| Dehazing   | SOTS-Outdoor / SOTS-Indoor     |
| Desnowing  | Snow100K                       |
| Low-light  | LOL-v1                         |
| Deblurring | GoPro                          |

Results 
