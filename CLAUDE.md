# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

用繁體中文回答與解釋

This is a quantum machine learning project focused on drug classification using patient health data. The project combines traditional machine learning with quantum computing approaches, implementing both classical ML methods and quantum SVM algorithms for predicting suitable drug types based on patient characteristics.

## Dataset

- **File**: `drug200.csv`
- **Size**: 200 patient records
- **Features**: Age, Sex, BP (Blood Pressure), Cholesterol, Na_to_K (Sodium to Potassium ratio)
- **Target**: Drug type (DrugY, drugX, drugA, drugB, drugC)

## Development Environment Setup

### Dependencies Installation
1. **Basic setup**: Run `bash install_quantum.sh` to install all quantum computing and ML dependencies
2. **Requirements**: All dependencies are listed in `requirements.txt` including:
   - qiskit>=1.0.0, qiskit-aer>=0.14.0, qiskit-machine-learning>=0.8.0
   - Standard ML libraries: pandas, numpy, matplotlib, seaborn, scikit-learn
   - imbalanced-learn for SMOTE oversampling

### Running the Code
- **Classical ML workflow**: Open `drug-classification-w-various-ml-models.ipynb` in Jupyter
- **Quantum SVM implementations**: Run Python scripts directly:
  - `python true_quantum_svm.py` - Pure quantum implementation
  - `python mike_Quantum_SVM.py` - IBM quantum hardware approach
  - `python quantum_svm_fixed.py` - Fixed quantum implementation

## Project Architecture

### 1. Classical Machine Learning Pipeline (`drug-classification-w-various-ml-models.ipynb`)
- Data exploration and preprocessing with feature binning
- Six algorithms comparison: Logistic Regression, KNN, SVM, Naive Bayes, Decision Tree, Random Forest
- SMOTE oversampling for class imbalance
- Results achieve 85% accuracy with Logistic Regression

### 2. Quantum Machine Learning Implementations
- **`true_quantum_svm.py`**: Pure quantum approach using:
  - ZZFeatureMap for quantum feature encoding
  - FidelityQuantumKernel for quantum kernels
  - Variational Quantum Classifier (VQC)
  - AerSimulator for quantum circuit simulation
- **`mike_Quantum_SVM.py`**: IBM quantum hardware implementation with real QPU backends
- **`quantum_svm_fixed.py`**: Optimized quantum SVM with error handling

### 3. Data Processing Pipeline
Both classical and quantum approaches follow consistent preprocessing:
- Age binning into 7 categories (<20s, 20s, 30s, 40s, 50s, 60s, >60s)
- Na_to_K ratio binning into 4 ranges (<10, 10-20, 20-30, >30)
- One-hot encoding for categorical variables
- 70/30 train/test split with random_state=0

## Key Technical Implementation Details

### Quantum Circuit Design
- Feature dimension: 4-18 qubits depending on encoding method
- Quantum feature maps: ZZFeatureMap with entanglement
- Optimization: COBYLA, SPSA optimizers for variational circuits
- Measurement shots: 1024-8192 for quantum state sampling

### Model Performance Tracking
- Classical ML baseline: 85% accuracy (Logistic Regression)
- Quantum implementations target matching or exceeding classical performance
- Results visualization saved as PNG files in project root

## Common Development Tasks

### Testing Quantum Environment
```bash
python -c "from qiskit import QuantumCircuit; from qiskit_machine_learning.algorithms import QSVC; print('✅ Quantum environment ready')"
```

### Running Full Comparison
1. Execute notebook for classical ML baseline
2. Run `python true_quantum_svm.py` for quantum comparison
3. Compare accuracy results and visualizations

### Adding New Quantum Algorithms
- Extend `TrueQuantumSVM` class in `true_quantum_svm.py`
- Follow existing pattern: feature_map → quantum_kernel → classifier
- Maintain consistent preprocessing pipeline for fair comparison