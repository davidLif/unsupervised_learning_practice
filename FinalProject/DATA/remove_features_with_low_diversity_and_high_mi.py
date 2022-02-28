import numpy as np
import pandas as pd

data_df = pd.read_csv("FinalProject/DATA/USCensus1990.data.txt")
data_df = data_df.drop(columns=['caseid'])

# find features that the majority of the samples has the same values (over 98% has value 0)
unmeaningful_features = []
for column in data_df.columns:
    unique_values_counts = data_df[column].value_counts()
    max_perc = unique_values_counts.max() / unique_values_counts.sum()
    if max_perc > 0.98:
        print(column, unique_values_counts.argmax(), max_perc)
        unmeaningful_features.append(column)
data_df.drop(columns=unmeaningful_features, inplace=True)

# Create correlation matrix
corr_matrix = data_df.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation coefficient greater than 0.95
correlated_features=[]
for column in upper.columns:
  if any(upper[column] > 0.95):
    indx = upper.index[np.where(upper[column] > 0.95)]
    print(upper[column][indx])
    print(f"{column} is correlated to {indx}")
    correlated_features.append(column)

print(correlated_features)
data_df.drop(columns=correlated_features, inplace=True)

