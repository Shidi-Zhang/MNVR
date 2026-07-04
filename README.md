# MNVR

For replication of the drug recommendation outcomes reported in this paper, please refer to the guidelines below.

# Overview

We have modularized and encapsulated the code to enhance its readability and maintainability. In summary, the MNVR can be split into two core components: the feature extractor and the prediction head. The feature extractor primarily captures multi-dimensional clinical feature representations of patients, while the prediction head computes the classification probability for each medical condition label.


# Requirements
Ensure your local environment meets the following requirements before installation:

| Package          | Version Requirement |
|------------------|---------------------|
| PyTorch          | 1.9 ≤ version ≤ 1.12.1 |
| NumPy            | == 1.15.1           |
| Python           | ≥ 3.8               |
| scikit-learn     | ≥ 0.24.2            |

> ⚠️ Note: PyTorch version is constrained to 1.9 ~ 1.12.1 (inclusive) for compatibility with core modules. We recommend installing PyTorch with CUDA support if GPU acceleration is available (see [PyTorch Official Installation Guide](https://pytorch.org/get-started/previous-versions/) for version-specific commands).


# Dataset
We provide the dataset in the `data/` folder. 
The experimental data in this project is derived from the **MIMIC-III** (Medical Information Mart for Intensive Care III) and **MIMIC-IV** datasets — two large-scale public clinical databases containing de-identified intensive care unit (ICU) patient records. For complete raw data (including full clinical notes, lab results, and prescription records), please refer to the official MIMIC repositories:
- MIMIC-III: [PhysioNet/MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/1.4/)
- MIMIC-IV: [PhysioNet/MIMIC-IV Clinical Database](https://physionet.org/content/mimiciv/2.2/)

# Documentation
```
--src
  │--README.md
  │--train_MMVR.py
  │--util.py
  │--models.py
  │--layers.py
  │--attention.py
  │--SetTransformer.py
  
--data
  │--atc2rxnorm.pkl
  │--ddi_A_final.pkl
  │--ddi_mask_H.pkl
  │--ehr_adj_final_newone.pkl
  │--records_final.pkl
  │--substructure_smiles.pkl
  │--voc_final.pkl

--gnn
  │--GNNConv.py
  │--GNNs.py
  │--_init_.py
  │--utils.py
```
# How to MNVR
## Train the MNVR Model
### Prerequisites
Before training, make sure the following conditions are met:
1. All required dependencies are installed (see the [Prerequisites](#prerequisites) section for the full dependency list).
2. Data is correctly placed in the `./data` directory as described above.

### Training Commands
Open the terminal, navigate to the **project root directory**, and execute the following commands:
```bash
# Step 1: Enter the source code directory where train_MNVR.py is located
cd src

# Step 2: Start training with default parameters
python train_MNVR.py
```
# TODO
To make the experiments more efficient, we developed some experimental scripts, which will be released along with the paper later.
