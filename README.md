# 🌍 GeoFAR: Geography‑informed Frequency‑Aware Representations for Climate Downscaling

Official implementation for the ICLR26 paper: [GeoFAR](https://openreview.net/forum?id=0WHpOekph0&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2026%2FConference%2FAuthors%23your-submissions))

## 📘 Introduction

Climate super‑resolution aims to reconstruct high‑resolution climate
fields from coarse‑resolution inputs. However, deep learning models struggle to recover **geography‑dependent high‑frequency
structures**, such as those induced by complex terrain, coastlines, and
regional atmospheric dynamics.

We introduce **GeoFAR (Geography‑informed Frequency‑Aware Representation)**:

-   🌊 **Frequency‑Aware Representation (FAR)** to model multi‑frequency
    spatial structures.
-   🗺 **Geography‑aware implicit representations (Geo‑INR)** to
    incorporate terrain and geographic priors.
-   ⚙️ Compatibility with both **deterministic and generative
    super‑resolution backbones**.

GeoFAR improves reconstruction fidelity for climate variables and
significantly enhances the recovery of **high‑frequency details** in climate downscaling tasks.

------------------------------------------------------------------------

## 🧰 Installation

Clone the repository:

``` bash
git clone https://github.com/eceo-epfl/GeoFAR.git
cd GeoFAR
```

Create the conda environment:

``` bash
conda create -n geofar python=3.10
conda activate geofar
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

Or install the repository in editable mode:

``` bash
pip install -e .
```

------------------------------------------------------------------------

## 📦 Dataset

GeoFAR supports [CERRA](https://cds.climate.copernicus.eu/datasets/reanalysis-cerra-single-levels?tab=download), [ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download), [PRISM](https://prism.oregonstate.edu/), and customized datasets for super‑resolution
experiments.

### 🇪🇺 CERRA Downscaling

We evaluate GeoFAR on the **CERRA regional climate dataset** under two experimental settings.

#### Single-variable setting

We perform downscaling on **2m temperature**:

- 🌡 **2m temperature (`t2m`)**

#### Multi-variable setting

We jointly downscale multiple surface variables:

- 🌡 2m temperature (`t2m`)
- 🧭 10m wind (`10u`, `10v`)
- 💧 2m relative humidity (`rh2m`)
- 🌍 surface pressure (`sp`)

#### Dataset structure

The expected directory structure is:

    dataset/
    ├── cerra/
    │   ├── low-res/
    │   │   ├── train
    │   │   ├── val
    │   │   └── test
    │   └── high-res/
    │       ├── train
    │       ├── val
    │       └── test

Additional preprocessing scripts can be found in:

    tools/preprocess/

### 🌐 ERA5 Downscaling

We also evaluate GeoFAR on **global climate downscaling tasks** using ERA5 data.

Instructions for downloading and preprocessing ERA5 datasets can be found in the **climate-learn documentation**:

https://climatelearn.readthedocs.io/en/latest/user-guide/tasks_and_datasets.html


### 🇺🇸 ERA5 → PRISM Downscaling

For **global-to-local downscaling**, we use ERA5 as the low-resolution input and PRISM as the high-resolution target.

Dataset preparation follows the same procedure described in the  **climate-learn documentation**:

https://climatelearn.readthedocs.io/en/latest/user-guide/tasks_and_datasets.html


------------------------------------------------------------------------

## 🧠 Supported Models

To our knowledge, **GeoFAR provides one of the most comprehensive deep-learning toolboxes for climate downscaling**, covering classical interpolation baselines, generic image super-resolution backbones, and climate-specific downscaling models. This repository provides a unified benchmarking framework for **CNN-, GAN-, Transformer-, diffusion-, and neural-operator-based** super-resolution methods.

| Model | Venue | Brief description | Link |
|---|---|---|---|
| ResNet | CVPR 2016 | Residual CNN backbone widely used as a strong SR/downscaling baseline. | [Paper](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) |
| U-Net | MICCAI 2015 | Encoder-decoder architecture with skip connections for dense prediction and super-resolution tasks. | [Paper](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) |
| ViT | ICLR 2021 | Vision Transformer using patch tokens and global self-attention for representation learning. | [Paper](https://openreview.net/forum?id=YicbFdNTTy) |
| EDSR | CVPRW 2017 | Enhanced deep residual network optimized for high-fidelity image super-resolution. | [Paper](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/html/Lim_Enhanced_Deep_Residual_CVPR_2017_paper.html) |
| FFL | ICCV 2021 | The focal frequency loss to mitigate frequency bias of SR. | [Paper](https://arxiv.org/abs/2012.12821) |
| SRGAN | CVPR 2017 | GAN-based super-resolution model designed to generate perceptually realistic textures. | [Paper](https://openaccess.thecvf.com/content_cvpr_2017/html/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.html) |
| SwinIR | ICCVW 2021 | Transformer-based image restoration network built on Swin Transformer blocks. | [Paper](https://arxiv.org/abs/2108.10257) |
| SRFormer | ICCV 2023 | Transformer SR model with permuted self-attention to enlarge receptive fields. | [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhou_SRFormer_Permuted_Self-Attention_for_Single_Image_Super-Resolution_ICCV_2023_paper.pdf) |
| DeepSD | KDD 2017 | Early deep-learning-based statistical downscaling method for climate data. | [Paper](https://dl.acm.org/doi/10.1145/3097983.3098004) |
| FACL | NeurIPS 2025 | A frequency domain loss for precpitation Nowcasting | [Paper](https://openreview.net/forum?id=0aN7VWwp4g) |
| SmCL | JMLR 2023 | Hard-Constrained deep learning for climate downscaling. | [Paper](https://www.jmlr.org/papers/volume24/23-0158/23-0158.pdf) |
| DSFNO | JMLR 2024 | Fourier Neural Operator designed for arbitrary-resolution climate downscaling. | [Paper](https://arxiv.org/abs/2305.14452) |
| ClimateDiffuse | arXiv 2024 | Diffusion-based generative model for climate super-resolution. | [Paper](https://arxiv.org/abs/2404.17752) |
| GeoFAR | ICLR 2026 | Geography-informed frequency-aware representation framework for climate downscaling. | [Paper](https://openreview.net/forum?id=0WHpOekph0) |

### Notes

- **Generic SR methods:** ResNet, U-Net, ViT, EDSR, SRGAN, SwinIR, SRFormer  
- **Climate-specific baselines:** DeepSD, DSFNO, ClimateDiffuse, FFL, FACL, SmCL
- **GeoFAR variants:** The framework supports multiple enhanced backbones such as **GeoFAR[U-Net]**, **GeoFAR[ViT]**, **GeoFAR[SRGAN]**, and **GeoFAR[DSFNO]**.

------------------------------------------------------------------------

## 🚀 Usage

### Training

Example command for training on the CERRA 2x single-variable setting:

``` bash
python experiments/downscaling/cerra_cerra_downscale.py {low-res-path} {high-res-path} vit t2m --ratio=2 --bs=2 --gpu=0
```



------------------------------------------------------------------------

### Evaluation

Run evaluation by specifying a checkpoint path:

``` bash
python experiments/downscaling/cerra_cerra_downscale.py {low-res-path} {high-res-path} vit t2m --ratio=2 --bs=2 --gpu=0 --checkpoint={checkpoint-path}
```

------------------------------------------------------------------------

## 📑 Citation

If you find GeoFAR useful in your research, please cite:

``` bibtex
@inproceedings{
xu2026geofar,
title={GeoFAR: Geography-Informed Frequency-Aware Super-Resolution for Climate Data},
author={Chang Xu and Gencer Sumbul and Li Mi and Robin Zbinden and Devis Tuia},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=0WHpOekph0}
}
```

------------------------------------------------------------------------

⭐ If you find this repository useful, please consider giving it a star!
