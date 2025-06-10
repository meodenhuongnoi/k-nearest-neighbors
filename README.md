# K-Nearest Neighbors Classifier

A production-ready KNN classifier that can train on any CSV dataset with automatic preprocessing and evaluation.

## Features

- **Automatic preprocessing**: Handles both numeric and categorical features
- **Interactive mode**: Guides you through target column selection
- **Model persistence**: Saves trained models with metadata
- **Comprehensive evaluation**: Accuracy, classification report, and confusion matrix
- **Flexible parameters**: Customizable k-value, weights, and test size

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage (Interactive Mode)
```bash
python knn-production.py data/IRIS.csv
```

### Specify Target Column
```bash
python knn-production.py data/IRIS.csv -t species
```

### Custom Parameters
```bash
python knn-production.py data/IRIS.csv -k 7 -w uniform -s 0.3
```

### All Options
```bash
python knn-production.py data.csv -t target_col -k 5 -w distance -s 0.2 -r 42 -o my_model.pkl
```

## Parameters

- `-t, --target`: Target column name (interactive if not specified)
- `-k, --neighbors`: Number of neighbors (default: 5)
- `-w, --weights`: Weight function - 'uniform' or 'distance' (default: distance)
- `-s, --test-size`: Test set fraction (default: 0.2)
- `-r, --random-state`: Random seed for reproducibility (default: 42)
- `-o, --output`: Output model path (default: models/knn_model.pkl)

## Output

The training pipeline generates:
- Trained model file (`.pkl`)
- Metadata file (`_metadata.json`)
- Performance metrics and evaluation results

## Example Dataset

The project includes the classic Iris dataset (`data/IRIS.csv`) for testing and demonstration. 