<<<<<<< HEAD
<div align="center">
  <img alt="BiMamba-Seg遥感分割框架示意图" src="https://github.com/user-attachments/assets/f1bf34c9-36a4-435a-b0cc-e08b8a120220" width=500 />
</div>
<h1 align="center">BiMamba-Seg: Bidirectional Cross-Modal Modulation and Dual-Axis Mamba for Open-Vocabulary Remote Sensing Segmentation</h1>
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue.svg" />
  <img src="https://img.shields.io/badge/PyTorch-2.4.0-red.svg" />
  <img src="https://img.shields.io/badge/CUDA-11.8-green.svg" />
  <img src="https://img.shields.io/badge/Detectron2-supported-orange.svg" />
</p>

## 🔍 Overview

**BiMamba-Seg** is an efficient framework for Open-Vocabulary Remote Sensing Image Semantic Segmentation (OVRSIS). It addresses insufficient vision-language interaction during cost aggregation and limited long-range spatial context modeling in high-resolution remote sensing imagery.

Built upon a cost-aggregation-based segmentation framework, BiMamba-Seg introduces two major components:

- **Stage-aware Bidirectional Cross-modal Modulation (SBCM)**: Dynamically adapts visual and textual guidance according to their respective roles in different aggregation stages.

- **Dual-axis Serpentine Mamba Aggregation (DSMA)**: Captures long-range spatial dependencies through recurrent state-space modeling along complementary horizontal and vertical directions.

<img width="3249" height="1750" alt="main_page-0001" src="https://github.com/user-attachments/assets/8777ea93-e2e8-4d69-bf24-0df756fd4507" />

## 🔑 Key Components

### 1. Stage-aware Bidirectional Cross-modal Modulation (SBCM)

SBCM consists of two complementary modules:

- **Text-guided Semantic Frequency Modulation (TSFM)**: Injects text-conditioned mid- and high-frequency information into visual guidance before spatial aggregation, enhancing category-relevant structural and textural responses.

- **Visual-guided Text Relevance Modulation (VTRM)**: Exploits image-level visual context to adapt category-wise text guidance before class aggregation, reducing interference from scene-irrelevant categories.

### 2. Dual-axis Serpentine Mamba Aggregation (DSMA)

DSMA performs efficient long-range spatial modeling through horizontal and vertical serpentine scanning. It employs chunk-wise traversal and recurrent state propagation across scanning segments and neighboring spatial lines, facilitating contextual information exchange without explicit pairwise spatial attention.

##  🏁 Get Started

### 🛠️ Requirements

- Python ≥ 3.8 (Python 3.10 recommended)

- PyTorch ≥ 1.13 and a compatible torchvision

- CUDA ≥ 11.7 (for GPU training)

- Detectron2

The following configuration is used in the installation example:

- Python 3.10

- PyTorch 2.4.0

- torchvision 0.19.0

- CUDA 11.8

### 📦 Installation

**1. Create a Conda environment**

```
conda create -n bimamba-seg python=3.10 -y
conda activate bimamba-seg
```
**2. Install PyTorch**
```
python -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu118
```
**3. Install project dependencies**
```
pip install -r requirements.txt
```
**4. Install Detectron2**
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
> [!IMPORTANT]
> Make sure that the CUDA environment and PyTorch installation are compatible when compiling Detectron2.

### 🗂️ Dataset Preparation

**1. Download Datasets**

Download the datasets from the following Hugging Face repository:
 https://huggingface.co/datasets/kkk2026/OVRSISBenchtest

Extract all downloaded datasets into a common parent directory.

**2. Configure Dataset Paths**

Navigate to:
```
code_final/cat_seg/data/datasets/
```
Modify the dataset registration files whose names start with `register`.

At the end of each registration file, locate the root variable and replace it with the actual dataset directory.

**For example:**

```
root = "/path/to/your/datasets"
```
> Make sure that the specified paths correctly correspond to your extracted dataset folders.

**3. Configure Evaluation Paths**

Open `eval.sh` and replace the existing JSON path with the correct path to your dataset configuration file.

**4. Configure Category JSON Files**

Open `configs/vitb_384_DLRSD.yaml`, Update the following parameters to point to the correct DLRSD.json file:
```
TRAIN_CLASS_JSON: "/path/to/DLRSD.json"
TEST_CLASS_JSON: "/path/to/DLRSD.json"
```
> [!IMPORTANT]
> Ensure that both paths point to the corresponding category JSON file in your dataset setup.

### 🚀 Training and Evaluation

The main execution script is:
```
sh run.sh $config $gpus $output $opts
```
⚙️ The arguments are:

| Argument | Description                                |
|:---------|:-------------------------------------------|
| `config` | Configuration file                         |
| `gpus`   | Number of GPUs                             |
| `output` | Output directory for experimental results  |
| `opts`   | Additional configuration options           |


**For example, to use the DLRSD configuration with four GPUs:**
```
sh run.sh configs/vitb_384_DLRSD.yaml 4 ./output ""
```
> [!IMPORTANT]
> Before evaluation, make sure that the JSON paths in eval.sh have been correctly configured.

### 🧠 Pretrained Models

Pretrained model weights and corresponding configuration files will be publicly released upon publication of this paper.

## 🙏 Acknowledgement
This project builds upon excellent prior work:

- [CAT-Seg](https://github.com/KU-CVLAB/CAT-Seg) - Cost aggregation framework
- [Detectron2](https://github.com/facebookresearch/detectron2) - Detection framework
- [CLIP](https://github.com/openai/CLIP) - Vision-language pretraining
- [Pi-Seg](https://github.com/LiBingyu01/Pi-Seg) - Open-vocabulary RS baseline

We sincerely thank the authors of previous open-vocabulary segmentation methods for their valuable research and publicly available implementations.

## 📚 Citation

If you find this work useful in your research, please consider citing our paper:

`BiMamba-Seg: Bidirectional Cross-modal Modulation and Dual-Axis Mamba for Open-Vocabulary Remote Sensing Segmentation.`

**Full bibliographic information will be added upon publication.**
=======
## Requirements

- Linux or macOS with Python ≥ 3.8
- PyTorch ≥ 1.13 and matching torchvision
- CUDA 11.7 or higher (for GPU training)
- Detectron2
## Installation

```
# Create conda environment
conda create -n bimamba-seg python=3.10
conda activate bimamba-seg

# Install PyTorch (adjust CUDA version as needed)
python -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Install Detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## Dataset Preparation

1. Download all datasets from https://huggingface.co/datasets/kkk2026/OVRSISBenchtest , and extract them to the same folder.
2. Modify the files starting with 'register' under `code_final/cat_seg/data/datasets`, and change the `root` variable at the end of the file to the folder where the dataset is located.
3. Modify eval.sh and replace it with the correct JSON path.
4. Modify the `TRAIN_CLASS_JSON` and `TEST_CLASS_JSON` in `configs/vitb_384_DLRSD.yaml` so that they point to the correct `DLRSD.json`.

## Train & Eval

```
sh run.sh $config $gpus $output $opts
```
>>>>>>> 6ffdf2a (initial files)
