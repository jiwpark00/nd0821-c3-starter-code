'''
Testing for the ML model

Author: Ji Park
Created: 4/18/2022
'''

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def read_data():
    data = pd.read_csv('../starter/data/clean_census.csv')
    return data


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
