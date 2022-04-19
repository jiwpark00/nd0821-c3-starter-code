'''
Testing for the API requests

Author: Ji Park
Created: 4/19/2022
'''

from fastapi.testclient import TestClient

def test_sample():
	pass

# from starter.main import app

# client = TestClient(app)

# def test_get_home():
# 	r = client.get("/")
# 	assert r.status_code == 200
# 	assert r.json() == {"Hello World, ": "Welcome!"}