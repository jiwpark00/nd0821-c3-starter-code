# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from xgboost import XGBClassifier

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

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

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train XGBoost Model
model = train_model(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred.shape[0])

# Save y_pred to make sure that file was scored with the right dataset
pd.DataFrame(y_pred).to_csv('../data/y_pred_xgb.csv',index=False)