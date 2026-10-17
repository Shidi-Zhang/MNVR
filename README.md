# MNVR

This repository provides the MNVR V2 implementation for medication recommendation. Follow the instructions below to prepare the environment, train the model, and evaluate a saved checkpoint.

# Overview

MNVR combines longitudinal clinical records with molecular representations to predict a set of medications for each patient visit. The implementation consists of two main components:

- **Feature extraction:** diagnosis and procedure embeddings, attention pooling, recurrent visit encoding, a medication co-occurrence graph, and molecular and substructure graph encoders.
- **Medication prediction:** clinical and molecular feature fusion, direct diagnosis/procedure-to-medication scores, and a learned gate that incorporates prescriptions from previous visits.

The current visit's medication labels are used as training targets, not as prediction inputs. The medication co-occurrence graph is constructed from training patients only.

# Requirements

The following versions were used in the verified CPU environment. They describe a tested configuration, not the only supported versions.

| Package | Tested version |
|---------|----------------|
| Python | 3.12.5 |
| PyTorch | 2.4.1+cpu |
| NumPy | 1.26.4 |
| SciPy | 1.13.1 |
| scikit-learn | 1.5.1 |
| pandas | 2.2.2 |
| dill | 0.3.8 |
| PyTorch Geometric | 2.6.1 |
| OGB | 1.3.6 |
| RDKit | 2024.3.5 |

Install the CPU or CUDA build of PyTorch appropriate for your environment separately, then install the remaining dependencies from the project root:

```bash
python -m pip install -r requirements-fixed.txt
```

An existing GPU experiment used PyTorch 2.6.0+cu124 and NumPy 2.2.6. If that environment already works, reinstalling dependencies is unnecessary. The pinned requirements above reproduce the tested CPU dependency set.

Check whether the active Python environment can access CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

# Dataset

Place the preprocessed data in the `data/` directory. The default model requires the following files:

| File | Description |
|------|-------------|
| `records_final.pkl` | Patient visit sequences containing diagnoses, procedures, and medication labels |
| `voc_final.pkl` | Diagnosis, procedure, and medication vocabularies |
| `ddi_A_final.pkl` | Drug–drug interaction adjacency matrix |
| `ddi_mask_H.pkl` | Medication-to-substructure mapping |
| `idx2SMILES.pkl` | Molecular SMILES associated with medication codes |
| `substructure_smiles.pkl` | Substructure SMILES |

The supplied records contain 6,350 patients and 15,032 visits. The existing patient order is preserved, with 4,233 patients for training, 1,059 for validation, and 1,058 for testing. The medication vocabulary contains 131 labels.

The EHR co-occurrence graph is rebuilt from the training split; the existing `ehr_adj_final_newone.pkl` and `atc2rxnorm.pkl` are not loaded by this training pipeline. Use `--data-dir` to specify another preprocessed data directory. Dataset access and preprocessing provenance should be documented separately when reporting experiments; this repository does not download raw clinical records.

# Documentation

```text
MNVR-main/
├── README.md
├── requirements-fixed.txt
├── src/
│   ├── train_MNVR.py          # Main entry point
│   ├── train_fixed.py         # Training, validation, and test evaluation
│   ├── mnvr_plus.py           # MNVR V2 model
│   ├── mnvr_components.py     # Shared model layers and graph encoding
│   ├── SetTransformer.py      # Substructure attention
│   ├── data_utils.py          # Data preparation and evaluation metrics
│   └── objectives.py          # Loss functions and patient weighting
├── gnn/
│   ├── __init__.py
│   ├── GNNs.py                # Graph-level molecular encoders
│   ├── GNNConv.py             # Graph convolution modules
│   └── utils.py               # SMILES-to-graph conversion
├── data/                     # Preprocessed data
├── tests/                    # Model and training workflow checks
└── saved1/                   # Generated checkpoints and experiment logs
```

# How to Run MNVR

## Prerequisites

1. Activate a Python environment with the required dependencies.
2. Place the required preprocessed files in `data/`.
3. Run the following commands from the **project root directory**.

## Train the Model

Train with default settings. CUDA is selected automatically when available; otherwise, the model runs on CPU:

```bash
python src/train_MNVR.py
```

Explicitly select GPU or CPU:

```bash
# GPU training
python src/train_MNVR.py --device cuda

# CPU training
python src/train_MNVR.py --device cpu
```

To specify a new output directory:

```bash
python src/train_MNVR.py --device cuda --run-dir saved1/mnvr_v2_run1
```

The output directory must be empty or not yet exist. Alternatively, run from `src/`:

```bash
cd src
python train_MNVR.py --device cuda
```

When running from `src/`, paths supplied on the command line are relative to that directory, for example `../saved1/mnvr_v2_run1/best.pt`. The default data directory is resolved from the project location.

## Default Training Settings

| Setting | Default |
|---------|---------|
| Architecture | V2 |
| Random seed | 3407 |
| Hidden dimension | 64 |
| Optimizer | AdamW |
| Learning rate | 0.00015 |
| Weight decay | 0.00001 |
| Maximum epochs | 100 |
| Early-stopping patience | 15 epochs |
| Patient / graph dropout | 0.3 / 0.3 |
| Graph encoder layers | 4 |
| Loss | BCE + 0.15 × soft Jaccard loss |
| Training weighting | Equal total weight per patient |
| Prediction threshold | 0.5 |

Molecular features, visit history, direct medication scores, and the history-copy gate are enabled by default. The DDI penalty and DDI memory subtraction are disabled by default. Learning rate reduction is based on validation performance. V2 is the only supported architecture; `--architecture v2` remains accepted for compatibility.

## Quick CPU Check

Check the training pipeline and active-branch gradients before a full experiment:

```bash
python src/train_MNVR.py --device cpu --smoke-test --check-gradients
```

This checks two patients per split and at most three training steps. Its scores are diagnostic only. Remove `--smoke-test` for a full training run.

## Validate a Saved Model

Replace `YOUR_RUN` with the actual experiment directory:

```bash
python src/train_MNVR.py --device cuda --validate --resume-path saved1/YOUR_RUN/best.pt
```

The checkpoint's saved threshold is used. Optional validation-only threshold selection is available through `--tune-threshold`; report those results separately from `JA@0.5`.

## Evaluate on the Test Set

After selecting the model and prediction threshold using validation data, evaluate the independent test split:

```bash
python src/train_MNVR.py --device cuda --eval --resume-path saved1/YOUR_RUN/best.pt
```

Training does not automatically evaluate the test set. Test evaluation uses the threshold saved in the checkpoint and does not allow threshold tuning. Existing V2 checkpoints remain compatible with this implementation.

# Outputs and Evaluation

Unless `--run-dir` is supplied, each training run creates a timestamped directory under `saved1/`.

| File | Contents |
|------|----------|
| `config.json` | Training settings, model configuration, and data fingerprints |
| `history.json` | Per-epoch losses and validation metrics |
| `best.pt` | Best validation-selected model weights and evaluation metadata |
| `best_validation.json` | Metrics for the selected checkpoint |
| `validation_predictions.npz` | Validation labels, probabilities, and patient boundaries |

Explicit validation additionally writes `validation_metrics.json`; optional threshold calibration produces `calibrated.pt`. Independent test evaluation writes `test_metrics.json` alongside the supplied checkpoint.

Jaccard, PRAUC, precision, recall, and F1 are averaged across visits within each patient and then across patients. DDI is computed as the global fraction of recommended medication pairs with known interactions. Average medication count is reported per visit. `PRAUC` uses average precision from scikit-learn.

The checkpoint is selected by validation Jaccard. Report the data split, prediction threshold, seed, and DDI alongside Jaccard when comparing experiments. Validation results should not be presented as independent test results.

# Tests

Run the model and workflow tests from the project root:

```bash
python -m unittest discover -s tests -v
```

Tests cover model gradients, exclusion of current medication targets from prediction inputs, molecular feature caching, patient-level metrics, and checkpoint validation/test workflows. To see all available options:

```bash
python src/train_MNVR.py --help
```
