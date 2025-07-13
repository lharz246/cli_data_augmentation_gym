
# Incremental Learning with Targeted Forgetting via Data Augmentation

## 🧠 Project Overview

This project provides a modular framework for **incremental learning** in binary classification tasks. The main goal is to enable a model to sequentially learn new labels while **retaining prior knowledge**, specifically preserving the recognition of **class 0**.

At the same time, the system enables controlled **induction of catastrophic forgetting** using established **data augmentation techniques**. The core application context is network intrusion detection using datasets like **CIC-IDS 2017** and **ToN-IoT**.

---

## 🔧 Installation & Setup

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Download and Preprocess Dataset

Supported datasets:

- [CIC-IDS 2017](https://www.unb.ca/cic/datasets/ids-2017.html)
- [ToN-IoT](https://research.unsw.edu.au/projects/toniot-datasets)

Run preprocessing:
```bash
python preprocess.py --cic_dir path/to/cic/data --output_dir data/
```

This creates `.pt` files for training, validation, and testing.

---

## 🚀 Training Workflow

Train the model using:

```bash
python trainv2.py --config config.yaml
```

### Incremental Training Logic:

The model learns incrementally in stages:

- **Step 1**: Train with label `0` (class 0) vs. label `1` (class 1)
- **Step 2**: Train with label `0` (class 0) vs. label `2`
- **Step 3**: Train with label `0` (class 0) vs. label `3`
- ...

At each stage, the goal is to:
- Preserve the model’s ability to recognize **class 0**
- Integrate the new **class 1 label**
- Prevent **catastrophic forgetting** of previous class 1 labels

---

## 🧪 Inducing Forgetting via Data Augmentation

When `use_augmentation: True` is set in `config.yaml`, the system applies augmentation methods to **intentionally shift** previous class 1 labels closer to class 0 — provoking **forgetting**.

### Supported Augmentation Techniques:

- `LabelNoise`
- `MixUp`
- `FeatureCorruption`
- `FeatureSwap`
- `DataPoisoner`

Each augmenter is configurable individually via `augmenter_configs` in the config file.

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `trainv2.py` | Main training entry point |
| `incremental_trainer.py` | Implements incremental learning trainer |
| `base_model.py` | Base model class with optional EWC regularization |
| `augmenters.py` | All augmentation strategies implemented here |
| `preprocess.py` | Data loading and preprocessing for CIC/ToN datasets |
| `stats.py` | Dataset statistics and feature analysis |
| `config.yaml` | Configuration for training, model, and augmentations |

---

## 📊 Output & Logging

- Training metrics and plots are stored in `results/output/`.
- If `wandb` is enabled, experiments are also logged to Weights & Biases.
- Augmentation statistics and dataset insights are saved under `data/.../statistics`.

---

## 🧩 Extensibility

The system is designed to be modular and easy to extend:

- **Add new models**: Define in `base_model.py` and list in `config.yaml`
- **Add new augmenters**: Implement in `augmenters.py`
- **Add new loss types**: Extend loss handling in the training loop and config

---

## 🔬 Research Objective

> Investigate how established data augmentation techniques can be used to **intentionally induce catastrophic forgetting** in incremental learning.

This setting allows controlled experiments that simulate memory degradation under adversarial conditions.

---

## 📎 Sample Configuration (`config.yaml`)

```yaml
# Model parameters
model:
  - name: resmlp
    embed_dim: 4096
    depth: 8
    width_schedule: bottleneck
    bottleneck_factor: 0.4
    dropout: 0.2
    residual_strength: 0.8
    class_0_preservation: True


epochs: 14
batch_size: 2048
lr: 1e-4
optimizer: AdamW
weight_decay: 0.00
max_norm : 0.9
balance_classes: True
true_incremental: True
trainer_type: incremental
use_mixed_precision: True
# project
use_wandb: false
wandb_project: 
wandb_entity: 
wandb_run_name: 
output_dir: results/output
data_path: data/cic
scheduler: cosine_restart
scheduler_params:
  - T_0: 1
    eta_min: 7.106840358531577e-8
stats_path: data/cic/statistics/dataset_statistics.json
losses:
  - name: focal
    weight: 10
    gamma: 7.5
    alpha: 0.5
    reduction: mean
  - name: ewc
    weight: 1500

use_augmentation: False
target_class: 0
target_labels: [2,3,4,5,6]
augment_factor: 0.25

augmenter_configs:
  - name: FeatureCorruption
    corruption_strength: 3.0
    probability: 1.0
  - name: LabelNoise
    flip_probability: 1.0
    probability: 1.0
  - name: MixUp
    alpha: 0.01
    probability: 1.0
  - name: FeatureSwap
    swap_ratio: 1.0
    probability: 1.0
  - name: DataPoisoner
    epsilon: 2.0
    probability: 1.0
```

---

## 📜 License

This project was developed as part of an academic research course. Not intended for commercial use without permission.
