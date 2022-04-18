# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

# Add the necessary imports for the starter code.
from ml.data import process_data

# Add code to load in the data.

# Optional enhancement, use K-fold cross validation instead of a train-test split.
data = pd.read_csv('../data/clean_census.csv')

train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# I commented below to test out Github Actions
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

print(X_train.shape)
# Proces the test data with the process_data function.

# Train and save a model.
