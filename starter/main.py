# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
from starter.ml.data import process_data

app = FastAPI()

class Value(BaseModel):
    value: dict

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

# Due to integration with pytest and local, if we are running on local
# This would execute to push "up" the directory one level
if '/starter' in os.getcwd():
	os.chdir('..')

# Imports the model
score_data = pd.read_csv('starter/data/first_100_test_inputs.csv')
train_data = pd.read_csv('starter/data/train.csv')

xgb = joblib.load("starter/model/final_xgb.pkl")

@app.get("/")
def welcome():
	return {"Hello World, ": "Welcome!"}

@app.get("/predict_static")
def return_prediction():
	'''
	Sample model output for testing
	'''

	print(score_data.iloc[0:1])
	# print(xgb.predict(score_data.iloc[0:1]))
	print(score_data.shape)
	return {'Hi': score_data.shape}

@app.post("/predict_dynamic")
def predict(body: Value):
	body_dict = body.dict()
	body_dict_vals = body_dict['value']

	body_dict_vals_fixed = {}

	for k,v in body_dict_vals.items():
		if '-' in k:
			body_dict_vals_fixed[k.replace('-','_')] = v
		else:
			body_dict_vals_fixed[k] = v

	# Due to index issue
	fixed_body = {k:[v] for k,v in body_dict_vals_fixed.items()}  # WORKAROUND
	fixed_body_df = pd.DataFrame(fixed_body)

	# This is inefficient but lets us to re-use encoder and lb
	X_train, y_train, encoder, lb = process_data(
    train_data, categorical_features=cat_features, label="salary", training=True
)

	fixed_body_processed, y_test, encoder, lb = process_data(
    fixed_body_df, categorical_features=cat_features, label=None, training=False, encoder=encoder, lb=lb
)

	fixed_body_scores = xgb.predict(fixed_body_processed)
	
	final_score = str(fixed_body_scores[0])

	return {"Prediction is ": final_score}