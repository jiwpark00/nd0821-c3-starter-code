'''
Testing for the API requests

Author: Ji Park
Created: 4/19/2022
'''

from fastapi.testclient import TestClient
from main import app
import json

client = TestClient(app)


def test_sample():
    pass


def test_get_home():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"Hello World, ": "Welcome!"}


def test_post_one():
    data = {
        "value": {
            "age": 31,
            "workclass": " Private",
            "fnlgt": 132996,
            "education": " HS-grad",
            "education-num": 9,
            "marital-status": " Married-civ-spouse",
            "occupation": " Prof-specialty",
            "relationship": " Husband",
            "race": " White",
            "sex": " Male",
            "capital-gain": 5178,
            "capital-loss": 0,
            "hours-per-week": 45,
            "native-country": " United-States"
        }
    }
    response = client.post(
        "/predict_dynamic",
        data=json.dumps(data),
        headers={
            "Content-Type": "application/json"},
    )
    assert response.status_code == 200


def test_post_two():
    # Another dataset - this time includes 0 as this could be a problem
    data = {
        "value": {
            "age": 0,
            "workclass": " Private",
            "fnlgt": 0,
            "education": " HS-grad",
            "education-num": 0,
            "marital-status": " Married-civ-spouse",
            "occupation": " Prof-specialty",
            "relationship": " Husband",
            "race": " White",
            "sex": " Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 0,
            "native-country": " United-States"
        }
    }
    response = client.post(
        "/predict_dynamic",
        data=json.dumps(data),
        headers={
            "Content-Type": "application/json"},
    )
    assert response.status_code == 200
