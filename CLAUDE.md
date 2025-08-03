# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Quantum Machine Learning (QML) research project focused on drug classification using quantum computing techniques. The project implements quantum Support Vector Machines (QSVM) and Variational Quantum Classifiers (VQC) using Qiskit to classify drugs based on patient characteristics.

## Dataset

- **File**: `drug200.csv`
- **Features**: Age, Sex, Blood Pressure (BP), Cholesterol, Na_to_K ratio
- **Target**: Drug type (DrugY, drugC, drugX, drugA, drugB)
- **Size**: 200 samples with 5 features

## Core Dependencies

### Quantum Computing Stack
- **qiskit**: 1.4.3 (core quantum framework)
- **qiskit-aer**: 0.17.1 (quantum simulators)
- **qiskit-algorithms**: 0.3.1 (quantum algorithms)
- **qiskit-machine-learning**: 0.7.2 (quantum ML algorithms)

### Machine Learning & Data Science
- **scikit-learn**: 1.6.1 (classical ML algorithms)
- **pandas**: 2.3.1, **numpy**: 1.26.4 (data processing)
- **imbalanced-learn**: 0.13.0 (SMOTE oversampling)
- **matplotlib**: 3.10.3, **seaborn**: 0.13.2 (visualization)

## Main Implementation Files

### 1. Primary Quantum SVM Implementations
- **`mike_Quantum_SVM.py`**: Main 5-dimensional quantum SVM with ZZFeatureMap (includes Na_to_K feature)
- **`reference/quantum_svm_fixed.py`**: Complete quantum SVM with TrueQuantumSVM class (4-dimensional)
- **`reference/true_quantum_svm.py`**: Alternative quantum SVM implementation

### 2. Traditional SVM Comparisons
- **`mike_SVM_withoutBin.py`**: Enhanced SVM with 10 ML methods comparison (includes English visualization)
- **`mike_SVM_withoutBin_annotated.py`**: Fully annotated version with detailed code comments (recommended for learning)
- **`reference/mike_SVM_withBin.py`**: Traditional SVM with data binning preprocessing
- **`ml_comparison_english.py`**: Comprehensive ML methods comparison with English labels
- **`quick_english_ml.py`**: Quick 4-method comparison with English visualization

### 3. Core Quantum Architecture

#### QuantumSVM Class Structure (mike_Quantum_SVM.py)
```python
class QuantumSVM:
    def __init__(self, feature_dimension=5, reps=3, shots=4096)  # High-precision config
    def _setup_quantum_circuits()      # ZZFeatureMap with linear entanglement
    def _setup_quantum_kernel()        # FidelityQuantumKernel
    def _setup_quantum_classifier()    # QSVC implementation
    def fit(X, y, label_encoder)       # Academic-standard quantum training
    def predict(X)                     # Quantum prediction with dimension handling
```

#### Key Quantum Components
- **Feature Mapping**: `ZZFeatureMap` with academic standard configuration (≤4 qubits, 2 layers, linear entanglement)
- **Quantum Kernel**: `FidelityQuantumKernel` for quantum similarity computation
- **Backend**: `AerSimulator` with 4096 shots for high precision
- **Dimension Handling**: Automatic PCA reduction or zero-padding for feature dimension matching

## Common Development Commands

### Environment Setup
```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or 
venv\Scripts\activate     # Windows

# Install all dependencies
pip install -r requirements.txt
```

### Running Quantum Models
```bash
# Primary 5-dimensional quantum SVM (recommended)
python mike_Quantum_SVM.py

# 4-dimensional quantum SVM with comprehensive visualization
python reference/quantum_svm_fixed.py
```

### Running Traditional ML Comparisons
```bash
# Enhanced SVM with 10 ML methods comparison (English charts)
python mike_SVM_withoutBin.py

# Comprehensive ML methods comparison (8 methods, English labels)
python ml_comparison_english.py

# Quick 4-method comparison (fast execution, English charts)
python quick_english_ml.py

# Traditional SVM with data binning
python reference/mike_SVM_withBin.py
```

## Data Processing Pipeline

### Standard Workflow
1. **Load**: `pd.read_csv('drug200.csv')` - 200 samples, 5 features
2. **Encode**: `LabelEncoder` for categorical features (Sex, BP, Cholesterol)
3. **Scale**: `StandardScaler` for numerical normalization
4. **Split**: `train_test_split` with stratification (70/30 split)
5. **Dimension Matching**: Automatic PCA reduction (if >4 features) or zero-padding
6. **Quantum Training**: Feature mapping through ZZFeatureMap → QSVC training

### Class Distribution (Imbalanced)
- DrugY: Majority class
- drugC, drugX, drugA, drugB: Minority classes
- SMOTE oversampling available but not used in main implementations

## Quantum Circuit Specifications

### Academic Standard Configuration (Havlíček et al., Nature 2019)
- **Qubits**: Maximum 4 qubits (academic standard limit)
- **Layers**: 2 repetitions for ZZFeatureMap
- **Entanglement**: Linear connectivity pattern
- **Shots**: 4096 measurements for high precision
- **Circuit Depth**: Typically 10-15 gates after decomposition

### Feature Dimension Handling
- **5D → 4D**: Automatic PCA reduction with explained variance tracking
- **<4D**: Zero-padding to match quantum circuit requirements
- **Manual ZZ Implementation**: Fallback for Qiskit optimization issues

## Output Files and Visualization

### Generated Files
- **`quantum_results_fixed.csv`**: Detailed prediction results with correctness flags
- **`quantum_svm_results_fixed.png`**: 6-panel comprehensive quantum analysis
- **`drug_distribution.png`**: Class distribution visualization (when SMOTE used)
- **`ml_methods_comparison_english.png`**: Comprehensive ML methods comparison (English labels)
- **`quick_ml_comparison_english.png`**: Quick 4-method comparison visualization
- **`simple_ml_comparison_english.png`**: Simple 2-method accuracy comparison

### Visualization Components
1. Quantum circuit architecture parameters
2. Drug type distribution
3. Quantum feature space mapping (Age vs Na_to_K)
4. Confusion matrix with quantum predictions
5. Classification accuracy metrics
6. Quantum computing advantage analysis

## Performance Expectations

### Typical Results
- **Quantum SVM Accuracy**: 85-95% (dataset dependent)
- **Training Time**: 2-5 minutes on local simulator
- **Classical SVM Baseline**: ~93% accuracy for comparison
- **Quantum Advantage**: High-dimensional feature mapping, quantum parallelism

## Development Notes

### Qiskit Version Compatibility
- Uses Qiskit 1.4+ API with `Sampler` primitives
- Backward compatibility handled for deprecated `quantum_instance`
- Circuit decomposition implemented for optimization bypassing

### Feature Engineering Strategy
- Categorical encoding preserves quantum circuit compatibility
- Standard scaling ensures quantum gate parameter stability
- Na_to_K ratio treated as key continuous feature for quantum encoding

### Circuit Optimization
- Manual ZZ feature map implementation for academic accuracy
- Progress indicators for long quantum training processes
- Quantum circuit depth optimization for simulator efficiency

## File Structure
```
QML/
├── drug200.csv                           # Primary dataset
├── requirements.txt                      # Complete dependency list
├── mike_Quantum_SVM.py                  # Main 5D quantum SVM implementation  
├── mike_SVM_withoutBin.py               # Enhanced SVM with 10 methods comparison
├── mike_SVM_withoutBin_annotated.py     # Fully annotated version (learning guide)
├── ml_comparison_english.py             # Comprehensive ML comparison (English)
├── quick_english_ml.py                  # Quick 4-method comparison
├── CODE_STRUCTURE_GUIDE.md              # Detailed code explanation guide
├── reference/                            # Additional implementations and outputs
│   ├── quantum_svm_fixed.py            # 4D quantum SVM with TrueQuantumSVM
│   ├── mike_SVM_withBin.py             # Traditional SVM with preprocessing
│   ├── true_quantum_svm.py             # Alternative quantum implementation
│   ├── *.png                           # Generated visualizations
│   ├── *.csv                           # Results and analysis files
│   └── *.ipynb                         # Jupyter research notebooks
├── test.py                              # Testing utilities
├── mike.py                              # Additional utilities
└── venv/                                # Python virtual environment
```