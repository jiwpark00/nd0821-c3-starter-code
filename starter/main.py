# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# Imports the model
score_data = pd.read_csv('data/first_100_test_inputs.csv')

xgb = joblib.load("model/final_xgb.pkl")

@app.get("/")
def welcome():
	return {"Hello World, ": "Welcome!"}

@app.get("/predict")
def return_prediction():
	print(xgb.predict(score_data.iloc[0:1]))
	return {'Hi': score_data.shape}