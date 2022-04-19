# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from xgboost import XGBClassifier
import joblib

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

# Add code to load in the data.

# Optional enhancement, use K-fold cross validation instead of a train-test split.
data = pd.read_csv('../data/clean_census.csv')

data.columns = [col.replace('-','_') if '-' else col in col for col in data.columns]

train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

# Needs to re-use it in the inference for encoder and lb
train.to_csv('../data/train.csv',index=False)

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# To do the slice, we need to reset the test index
test.reset_index(drop=True,inplace=True)

# Save an example subset here
test.iloc[0:1].to_csv('../data/test_input_example.csv',index=False)

# Process the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Save this to X_test first 100 rows for inference - used in predict_static example
pd.DataFrame(X_test).iloc[0:100].to_csv('../data/first_100_test_inputs.csv',index=False)

# Train XGBoost Model
model = train_model(X_train, y_train)

y_pred = inference(model,X_test)

# Save y_pred to make sure that file was scored with the right dataset
pd.DataFrame(y_pred).to_csv('../data/y_pred_xgb.csv',index=False)

# Total precision, recall, fbeta
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

# Model evaluation on the test dataset
slice_based_output = [('columnName_sliceName'), ("precisionVal, recallVal, fbetaVal")]

slice_based_output.append( (("Total_NoSlice"), precision, recall, fbeta) )

for col in cat_features:
	# define categories
	col_categ = np.unique(test[col].values)
	for categ in col_categ:
		categ_ind = test[test[col] == categ].index
		y_test_sub = y_test[categ_ind]
		y_pred_sub = y_pred[categ_ind]
		slice_based_output.append( ((col + "_" + categ), compute_model_metrics(y_test_sub, y_pred_sub)) )

pd.DataFrame(slice_based_output).to_csv('../data/slice_output.txt',index=False)

# Model dumps
joblib.dump(model, '../model/final_xgb.pkl')
joblib.dump(encoder, '../model/encoder.pkl')
joblib.dump(lb, '../model/lb.pkl')