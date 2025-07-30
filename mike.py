import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os   
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

df_drug = pd.read_csv('drug200.csv')

#Data binning
# Binning the 'Age' column into categories
bin_age = [0, 19, 29, 39, 49, 59, 69, 80]
category_age = ['<20s', '20s', '30s', '40s', '50s', '60s', '>60s']
df_drug['Age_binned'] = pd.cut(df_drug['Age'], bins=bin_age, labels=category_age)
df_drug = df_drug.drop(['Age'], axis = 1)

# Binning the 'Na_to_K' column into categories
bin_NatoK = [0, 9, 19, 29, 50]
category_NatoK = ['<10', '10-20', '20-30', '>30']
df_drug['Na_to_K_binned'] = pd.cut(df_drug['Na_to_K'], bins=bin_NatoK, labels=category_NatoK)
df_drug = df_drug.drop(['Na_to_K'], axis = 1)

# splitting the dataset
X = df_drug.drop(["Drug"], axis=1)
y = df_drug["Drug"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Feature Engineering
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

print(X_train.head())

#SMOTE Technique

