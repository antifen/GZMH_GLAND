# GZMH_GLAND

## Project Overview
This repository contains the supporting code for the **GZMH_GLAND** dataset, providing scripts for dataset preprocessing, classification and segmentation task testing.

---

## Repository Structure
```
GZMH_GLAND/
├── classification/          # Test scripts for dataset classification tasks
├── segmentation/           # Test scripts for dataset segmentation tasks
├── patch_extractor.py      # Script for dataset patching/preprocessing
└── README.md                # Project documentation
```

---

## Code Function Overview
| File/Directory | Description |
| :--- | :--- |
| `patch_extractor.py` | Dataset preprocessing script, used for patch extraction or dataset splitting of the GZMH_GLAND dataset, generating standard formatted data for subsequent tasks |
| `classification/` | Contains test scripts for classification tasks on the GZMH_GLAND dataset, to verify the dataset's usability in classification scenarios |
| `segmentation/` | Contains test scripts for segmentation tasks on the GZMH_GLAND dataset, to verify the dataset's usability in segmentation scenarios |

---

## Environment Setup
All code in this project is developed in Python, with Python 3.8+ recommended. The core dependencies are listed below (additional packages may be required based on actual script usage):
```bash
# Basic data processing dependencies
pip install numpy pandas opencv-python pillow

# Deep learning dependencies (if using PyTorch)
pip install torch torchvision

# Data processing and evaluation tools
pip install scikit-learn
```

---

## Quick Start
1.  Configure the Python environment and install the required dependencies listed above.
2.  Run `patch_extractor.py` to complete dataset preprocessing.
3.  Execute test scripts in the `classification/` or `segmentation/` directory according to your task requirements.

---
