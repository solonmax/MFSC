# MFSC: Multi-Scale Frequency–Spatial Cooperation for Universal Image Restoration

Official PyTorch implementation for the paper:

**Uncertainty-Aware Frequency–Spatial Diffusion for Universal Image Restoration**

---

## Introduction

Universal image restoration aims to handle multiple degradations (rain, snow, haze, blur, low-light) using **a single model**.  
However, existing diffusion/transformer based approaches usually treat:

* spatial-domain features and frequency-domain features **independently**
* all degradation types **equally** during optimization

which leads to poor cross-task generalization.

This repository provides the official implementation of:

- **MFSC** — Multi-scale Frequency–Spatial Cooperation
- **TUA** — Task-Specific Uncertainty Allocation

MFSC explicitly enables collaborative spectral–spatial reasoning across scales  
while TUA introduces uncertainty-aware loss weights to handle task imbalance.

---

## Environment
* Python 3.79
* Pytorch 1.12
*  ## Install
```bash
conda create -n mfsc python=3.7
conda activate mfsc
pip install -r requirements.txt
## Dependencies
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia
pip install opencv-python
pip install scikit-image
pip install tensorboard
pip install matplotlib 
pip install tqdm
## Dataset Preparation

Preparing the train and test datasets following our paper Dataset Construction section as:

```bash
Datasets/Restoration
|--syn_rain
|  |--train
      |--input
      |--target
|  |--test
|--Snow100K
|--Deblur
|--LOL
|--RESIDE
   |--OTS_ALPHA
      |--haze
      |--clear
   |--SOTS/outdoor
```
Then get into the `data/universal_dataset.py` file and modify the dataset paths. 
## Train 
```
python train.py
```
## Test and Calculate the Metric
Note that the dataset of SOTS can not calculate the metric online as the number of input and gt images is different. 
Please use eval/SOTS.m.  <br>
The pretrained weight of model-300.pt is used to test with timestep 3, **check double times whether you loaded, the ``result_folder'' !!!**. <br>
Notably, change the 'task' id in test.py Line 43 to your task, low-light enhancement for 'light_only', deblur for 'blur', dehaze for 'fog', derain for 'rain', desnow for 'snow' 
```
python test.py
```
## Results & Checkpoints

Checkpoints will be released upon acceptance.


