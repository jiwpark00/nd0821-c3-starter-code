# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()

class Value(BaseModel):
    age: int = Field(..., example=45)
    workclass: str = Field(..., example=" Private")
    fnlgt: int = Field(..., example=12345)
    education: str = Field(..., example=" HS-grad")
    education_num: int = Field(alias="education-num",example=9)
    marital_status: str = Field(alias="marital-status",example=" Married-civ-spouse")
    occupation: str = Field(..., example=" Prof-specialty")
    relationship: str = Field(..., example=" Husband")
    race: str = Field(..., example=" White")
    sex: str = Field(..., example=" Male")
    capital_gain: int = Field(alias="capital-gain",example=1234)
    capital_loss: int = Field(alias="capital-loss",example=567)
    hours_per_week: int = Field(alias="hours-per-week",example=45)
    native_country: str = Field(alias="native-country",example=" United States")


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
	if '/app' not in os.getcwd(): 
		os.chdir('..')
	else: # This is updated to allow for Heroku
		os.chdir('..')

# from starter.ml.data import process_data
# Imports the model
score_data = pd.read_csv('starter/data/first_100_test_inputs.csv')
train_data = pd.read_csv('starter/data/train.csv')

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

	# Due to index issue
	fixed_body = {k:[v] for k,v in body_dict.items()}  # WORKAROUND
	fixed_body_df = pd.DataFrame(fixed_body)

	# This is inefficient but lets us to re-use encoder and lb
# 	X_train, y_train, encoder, lb = process_data(
#     train_data, categorical_features=cat_features, label="salary", training=True
# )
	xgb = joblib.load("starter/model/final_xgb.pkl")
	encoder = joblib.load("starter/model/encoder.pkl")
	lb = joblib.load("starter/model/lb.pkl")

	fixed_body_processed, y_test, encoder, lb = process_data(
    fixed_body_df, categorical_features=cat_features, label=None, training=False, encoder=encoder, lb=lb
)

	fixed_body_scores = xgb.predict(fixed_body_processed)
	
	final_score = str(fixed_body_scores[0])

	return {"Prediction is ": final_score}