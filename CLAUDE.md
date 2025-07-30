# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

用繁體中文回答與解釋

##

This is a machine learning project focused on drug classification using patient health data. The project is implemented as a Jupyter notebook that explores various ML algorithms for predicting suitable drug types based on patient characteristics.

## Dataset

- **File**: `drug200.csv`
- **Size**: 200 patient records
- **Features**: Age, Sex, BP (Blood Pressure), Cholesterol, Na_to_K (Sodium to Potassium ratio)
- **Target**: Drug type (DrugY, drugX, drugA, drugB, drugC)

## Development Environment

This is a Jupyter notebook-based project. To work with the code:

1. **Running the notebook**: Use Jupyter Lab or Jupyter Notebook to open `drug-classification-w-various-ml-models.ipynb`
2. **Dependencies**: The notebook uses standard data science libraries:
   - pandas, numpy (data manipulation)
   - matplotlib, seaborn (visualization)
   - scikit-learn (machine learning models)
   - imblearn (SMOTE for handling class imbalance)

## Machine Learning Pipeline

The notebook follows a structured ML workflow:

1. **Data Loading & Exploration**: Basic statistics and data quality checks
2. **EDA**: Comprehensive visualization of data distributions and relationships
3. **Data Preprocessing**:
   - Feature binning (Age into age groups, Na_to_K into ratio categories)
   - One-hot encoding for categorical variables
   - SMOTE oversampling to handle class imbalance
4. **Model Training**: Six different algorithms are compared:
   - Logistic Regression
   - K-Nearest Neighbors
   - Support Vector Machine (SVM)
   - Naive Bayes (Categorical & Gaussian)
   - Decision Tree
   - Random Forest
5. **Model Evaluation**: Accuracy scores, classification reports, and confusion matrices
6. **Output Generation**: Predictions exported to CSV format

## Key Architecture Patterns

- **Data-centric approach**: Heavy focus on data exploration and preprocessing
- **Model comparison framework**: Systematic evaluation of multiple algorithms
- **Reproducible pipeline**: Clear separation of data prep, training, and evaluation phases
- **Visualization-heavy**: Extensive use of plots for data understanding

## Working with This Project

When modifying or extending this project:

- The main workflow is in the single notebook file
- Data preprocessing steps are critical for model performance
- SMOTE balancing is applied to training data only
- Model hyperparameter tuning is done through grid search for some algorithms
- Output format transformation is handled in the final sections for practical use