# This will send a GET and POST request to the app on Heroku
# to check it is working. Note the app is at
# https://robsmith155-salary-prediction.herokuapp.com/

import json

import requests

example = {
    "age": 39,
    "workclass": "State-gov",
    "fnlwgt": 10,
    "education": "Bachelors",
    "education-num": 10,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

r = requests.post(
    url="https://robsmith155-salary-prediction.herokuapp.com/predict",
    data=json.dumps(example),
)

print(r.status_code)
print(r.json())

r = requests.get(url="https://robsmith155-salary-prediction.herokuapp.com/")

print(r.status_code)
print(r.json())
