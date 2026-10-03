import requests
import json

url = 'http://127.0.0.1:5000/predict'
data = {
    "gender": "M",
    "NationalITy": "KW",
    "PlaceofBirth": "KuwaIT",
    "StageID": "lowerlevel",
    "GradeID": "G-04",
    "SectionID": "A",
    "Topic": "IT",
    "Semester": "F",
    "Relation": "Father",
    "raisedhands": 10,
    "VisITedResources": 5,
    "AnnouncementsView": 2,
    "Discussion": 10,
    "ParentAnsweringSurvey": "No",
    "ParentschoolSatisfaction": "Bad",
    "StudentAbsenceDays": "Above-7"
}

response = requests.post(url, json=data)
print("Status Code:", response.status_code)
print("Response Body:", json.dumps(response.json(), indent=2))
