import requests
import json

response = requests.get('https://salary-predict-xgb.herokuapp.com/')

print('Here is the GET request from the home page')
print(response.text)

data = {
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
post_response = requests.post(
        "http://127.0.0.1:8000/predict_dynamic",
        data=json.dumps(data),
        headers={
            "Content-Type": "application/json"},
    )
print('Here is the POST request')
print(post_response.text)