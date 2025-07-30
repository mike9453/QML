import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os   

df_drug = pd.read_csv('drug200.csv')
print(df_drug.describe())

skewAge = df_drug.Age.skew(axis = 0, skipna = True)
print('Age skewness: ', skewAge)

skewNatoK = df_drug.Na_to_K.skew(axis = 0, skipna = True)
print('Na to K skewness: ', skewNatoK)

