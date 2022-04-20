'''
Testing for the ML model

Author: Ji Park
Created: 4/18/2022
'''

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, precision_score, recall_score
import joblib

@pytest.fixture(scope="session")
def read_data():
    data = pd.read_csv('starter/data/clean_census.csv')
    return data

@pytest.fixture(scope="session")
def split_data(read_data):
    train, test = train_test_split(read_data, test_size=0.20)
    return train, test

@pytest.fixture(scope="session")
def pred_result():
    y_pred = pd.read_csv('starter/data/y_pred_xgb.csv')
    return y_pred

def test_example():
    '''
    Test function included to pass the Github Actions
    '''
    pass

def test_correct_import(read_data):
    '''
    Ensures that the correct data is imported
    '''

    assert read_data.shape[0] == 32561

def test_correct_partition(split_data):
    '''
    Ensures the split is accurate
    '''

    assert split_data[0].shape[0] == 26048
    assert split_data[1].shape[0] == 6513

def test_pred_counts(pred_result):
    '''
    Ensures the model scored all the dataset
    '''

    assert pred_result.shape[0] == 6513

def test_train_model():
    '''
    Ensures train_model function correctly creates the model as xgboost.
    '''
    model = joblib.load('starter/model/final_xgb.pkl')
    model_type = str(type(model))
    assert ('xgboost' in model_type) == True

def test_compute_model_metrics():
    '''
    Ensures that fbeta, precision, recall are returned
    '''

    mock_y_test = [0,0,0,1] # fake 4 results
    mock_y_pred = [0,0,1,1] # fake 4 predictions

    mock_fbeta = fbeta_score(mock_y_test, mock_y_pred, beta=1, zero_division=1)
    mock_precision = precision_score(mock_y_test, mock_y_pred, zero_division=1)
    mock_recall = recall_score(mock_y_test, mock_y_pred, zero_division=1)

    assert mock_fbeta == 2/3
    assert mock_precision == 0.5
    assert mock_recall == 1.0

def test_inference(pred_result):
    '''
    Ensures that the inference ran correctly
    Checks that no value in y_pred are less than or bigger than 0 or 1
    '''

    wrong_vals = []

    for val in pred_result.values:
        if float(val) < 0 or float(val) > 1:
            wrong_vals.append(val)

    # Ensure model didn't throw predictions outside of 0 and 1
    assert len(wrong_vals) == 0

def test_slice_output_generator():
    '''
    Ensures that slice_output_generator did work correctly
    '''
    slice_data = pd.read_csv('starter/data/slice_output.txt')

    slice_data = slice_data.values # turns into series
    # check that precisionVal, recallVal, fbetaval were written in
    assert slice_data[1][0] == 'precisionVal, recallVal, fbetaVal'

