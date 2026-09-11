import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.read_csv('student-lifestyle-and-stress-dataset.csv')

# Separate numeric and categorical columns
numeric_cols = df.select_dtypes(include='number').columns
categorical_cols = df.select_dtypes(exclude='number').columns

# Impute numeric columns with median
if len(numeric_cols) > 0:
    num_imputer = SimpleImputer(strategy='mean')
    df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

# Impute categorical columns with most frequent value
if len(categorical_cols) > 0:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

df.to_csv('student-lifestyle-and-stress-dataset-imputed.csv', index=False)