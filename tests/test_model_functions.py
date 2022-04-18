'''
Testing for the ML model

Author: Ji Park
Created: 4/18/2022
'''

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

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