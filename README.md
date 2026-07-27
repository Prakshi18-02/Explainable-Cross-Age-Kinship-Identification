# Explainable Cross-Age Parent–Child Kinship Identification

## Overview

This repository contains the official implementation of our framework for explainable cross-age parent–child kinship identification using deep learning. The framework performs kinship verification followed by relationship classification and provides visual explanations through Grad-CAM and hereditary attention maps.

The proposed framework consists of:

- Binary Fusion based kinship verification
- Hereditary ResNet18 for relationship classification
- Grad-CAM visualization
- Hereditary attention maps
- Automatic report generation

The framework identifies four parent–child relationship categories:

- Father–Son
- Father–Daughter
- Mother–Son
- Mother–Daughter


## Repository Structure

```
CrossAgeKinship_Final_Repo/
│
├── checkpoints/          # Trained model checkpoints
├── data/                 # Train, validation and test CSV files
├── docs/                 # Architecture and documentation
├── results/
│   ├── confusion_matrix/
│   ├── explainability/
│   ├── graphs/
│   └── metrics/
├── sample_inputs/        # Example images for testing (optional)
├── src/
│   ├── deploy/           # Deployment pipeline
│   └── Official training, evaluation and model scripts
├── .gitignore
├── requirements.txt
└── README.md
```

## Dataset

This project uses the **Families in the Wild (FIW)** dataset for parent–child kinship identification.

Only the CSV files used for training and evaluation are included in this repository.

The original FIW facial images are **not distributed** due to dataset licensing restrictions. Users should obtain the dataset separately and place the images in the appropriate directory before running the code.

## Installation

Clone the repository:

```bash
git clone <repository_url>
cd CrossAgeKinship_Final_Repo
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Model Checkpoints

The repository contains the frozen model checkpoints used to generate the results reported in the paper.

Relationship Classification Models:

- Siamese ResNet18
- Siamese SE-ResNet18
- Siamese CBAM-ResNet18
- Hereditary ResNet18

Binary Verification Models:

- Binary Fusion Model
- Triplet Model
- Fusion Triplet Model

These checkpoints should not be retrained if the objective is to reproduce the reported experimental results.

## Evaluation

The official evaluation scripts used in the paper are:

Relationship Classification

```bash
python src/evaluate_relationship_models_fixed.py
```

Binary Verification

```bash
python src/evaluate_binary_models_metrics.py
```

Performance Graphs

```bash
python src/generate_hereditary_graphs.py
```

These scripts reproduce the quantitative results reported in the paper using the frozen checkpoints provided in the repository.

## Deployment

The deployment pipeline performs the following steps:

1. Face detection and alignment
2. Facial region extraction
3. Binary kinship verification using the Binary Fusion model
4. Parent–child relationship classification using Hereditary ResNet18
5. Grad-CAM visualization
6. Hereditary attention map generation
7. Automatic report generation

The deployment modules are located in:

```text
src/deploy/
```

## Results

The proposed framework was evaluated on the Families in the Wild (FIW) dataset.

### Relationship Classification

| Model | Accuracy (%) |
|-------|-------------:|
| Siamese ResNet18 | 84.42 |
| Siamese SE-ResNet18 | 79.53 |
| Siamese CBAM-ResNet18 | 80.70 |
| **Hereditary ResNet18 (Proposed)** | **90.70** |

### Binary Verification

| Model | Accuracy (%) |
|-------|-------------:|
| Triplet Model | 66.13 |
| Fusion Triplet Model | 67.51 |
| **Binary Fusion Model (Proposed)** | **68.97** |

Additional evaluation results, graphs, confusion matrices, and explainability visualizations are available in the `results/` directory.

## Repository Highlights

This repository includes:

- Official implementation of the proposed framework
- Training scripts
- Evaluation scripts
- Deployment pipeline
- Frozen model checkpoints
- Dataset split CSV files
- Performance graphs
- Confusion matrix
- Explainability results
- Architecture diagram

The repository is organized to reproduce the experimental results reported in the paper using the provided checkpoints.

## Acknowledgement

This work uses the Families in the Wild (FIW) dataset. The original dataset is not redistributed in this repository and should be obtained from the official FIW source in accordance with its licensing terms.
## License

This repository is intended for academic and research purposes.