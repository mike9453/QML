# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Quantum Machine Learning (QML) research project focused on drug classification using quantum computing techniques. The project implements quantum Support Vector Machines (QSVM) and Variational Quantum Classifiers (VQC) using Qiskit to classify drugs based on patient characteristics.

## Dataset

- **File**: `drug200.csv`
- **Features**: Age, Sex, Blood Pressure (BP), Cholesterol, Na_to_K ratio
- **Target**: Drug type (DrugY, drugC, drugX, drugA, drugB)
- **Size**: 200 samples with 5 features

## Key Dependencies and Setup

### Python Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Core Dependencies (requirements.txt)
- **Quantum Computing**: qiskit>=1.0.0, qiskit-aer>=0.14.0, qiskit-machine-learning>=0.8.0, qiskit-algorithms>=0.3.0
- **ML/Data Science**: scikit-learn>=1.0.0, pandas>=1.3.0, numpy>=1.21.0, imbalanced-learn>=0.8.0
- **Visualization**: matplotlib>=3.4.0, seaborn>=0.11.0

## Main Implementation Files

### 1. Quantum SVM Implementations
- **`quantum_svm_fixed.py`**: Complete quantum SVM implementation with `TrueQuantumSVM` class
- **`mike_Quantum_SVM.py`**: Enhanced quantum SVM with 5-dimensional feature space (includes Na_to_K)
- **`mike_SVM_withBin.py`**: Traditional SVM with data binning preprocessing
- **`mike_SVM_withoutBin.py`**: Comparison between traditional SVM and quantum SVM

### 2. Core Architecture

#### QuantumSVM/TrueQuantumSVM Class Structure
```python
class QuantumSVM:
    def __init__(self, feature_dimension=5, reps=3, shots=1024)
    def _setup_quantum_circuits()      # ZZFeatureMap + RealAmplitudes
    def _setup_quantum_kernel()        # FidelityQuantumKernel
    def _setup_quantum_classifier()    # QSVC + VQC options
    def fit(X, y)                      # Quantum training with PCA/padding
    def predict(X)                     # Quantum prediction
```

#### Key Quantum Components
- **Feature Mapping**: `ZZFeatureMap` with full entanglement for quantum state encoding
- **Variational Circuit**: `RealAmplitudes` ansatz for VQC
- **Quantum Kernel**: `FidelityQuantumKernel` for QSVM
- **Optimizers**: `COBYLA` for variational quantum algorithms
- **Backend**: `AerSimulator` with configurable shots (default: 1024)

## Common Development Commands

### Running Quantum SVM Models
```bash
# Run main quantum SVM (5-feature version)
python mike_Quantum_SVM.py

# Run complete quantum SVM with visualization
python quantum_svm_fixed.py

# Compare traditional vs quantum SVM
python mike_SVM_withoutBin.py

# Run traditional SVM with binning
python mike_SVM_withBin.py
```

### Data Processing Pipeline
1. **Load**: CSV data with pandas
2. **Encode**: LabelEncoder for categorical features (Sex, BP, Cholesterol)
3. **Scale**: StandardScaler for numerical features
4. **Resample**: SMOTE for class imbalance (when used)
5. **Split**: train_test_split with stratification
6. **Quantum Mapping**: Feature dimension matching via PCA or padding

### Model Training Workflow
1. **Initialize**: Quantum circuits with specified qubits/reps/shots
2. **Fit**: Feature scaling + label encoding + quantum training
3. **Predict**: Transform test data + quantum inference
4. **Evaluate**: Classification metrics + confusion matrix
5. **Visualize**: Generate quantum results plots (saved as PNG)

## Output Files
- **Results**: `quantum_results_fixed.csv` - prediction results
- **Visualizations**: `quantum_svm_results_fixed.png` - comprehensive quantum analysis plots
- **Distribution**: `drug_distribution.png` - class distribution after SMOTE

## Quantum Computing Specifics

### Circuit Architecture
- **Qubits**: 4-5 qubits depending on feature dimensions
- **Depth**: 3-layer circuits (configurable via `reps` parameter)
- **Entanglement**: Full connectivity between qubits
- **Measurements**: 1024 shots per quantum execution

### Quantum Advantage Features
- Quantum parallelism for high-dimensional feature mapping
- Quantum entanglement for complex feature relationships
- Quantum interference for optimized decision boundaries
- Exponential Hilbert space scaling

## Development Notes

- All quantum models use Qiskit's latest API (1.0+)
- Feature dimensions must match quantum circuit qubits (handled automatically via PCA/padding)
- Quantum training can take several minutes - progress indicators included
- Results include both accuracy metrics and quantum circuit information
- Visualization functions generate comprehensive analysis plots

## File Structure
```
QML/
├── drug200.csv                    # Dataset
├── requirements.txt               # Dependencies
├── quantum_svm_fixed.py          # Main quantum SVM implementation
├── mike_Quantum_SVM.py           # Enhanced quantum SVM (5D)
├── mike_SVM_withBin.py           # Traditional SVM with binning
├── mike_SVM_withoutBin.py        # SVM comparison script
├── reference/                     # Reference materials and outputs
│   ├── *.png                     # Visualization outputs
│   ├── *.csv                     # Results files
│   └── *.ipynb                   # Jupyter notebooks
└── venv/                         # Python virtual environment
```