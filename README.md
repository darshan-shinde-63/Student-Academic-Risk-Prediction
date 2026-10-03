# Student Dropout Risk Prediction

A Flask web application that estimates a student's dropout risk from academic engagement, attendance, and profile information. It combines a scikit-learn preprocessing and classification pipeline with a browser-based dashboard.

> This project is an educational prototype. Predictions are estimates from a small dataset and should not be used as the sole basis for decisions about students.

## Features

- Student profile and engagement input form.
- Dropout probability and Low / Medium / High risk category.
- REST API accepting one student or a list of students.
- Pre-trained model pipeline in `dropout_model.pkl`.
- Training script that compares classifiers, tunes the best candidate by dropout recall, and saves the selected pipeline.

## Project files

| File or folder | Purpose |
| --- | --- |
| `app.py` | Flask server, dashboard route, and `/predict` API. |
| `frontend/` | HTML, CSS, and JavaScript for the dashboard. |
| `train.py` | Loads data, trains/evaluates models, runs hyperparameter search, and saves the model. |
| `dropout_model.pkl` | Pre-trained model pipeline used by the application. |
| `xAPI-Edu-Data.csv` | Student performance dataset used for training and evaluation. |
| `requirements.txt` | Python package dependencies. |
| `run.bat` | Windows launcher; creates the virtual environment if missing, installs dependencies, starts the server, and opens the browser. |
| `test_api.py` | Example script for sending a prediction request to a running server. |

## Requirements

- Windows
- Python installed and available through the Python launcher (`py`)
- Internet access for the first dependency installation

The dashboard loads Tailwind CSS and Chart.js from public CDNs, so those visual libraries also require internet access in the browser.

## Run the project

### Windows (recommended)

Double-click `run.bat` in the project folder. It will:

1. Change to the project directory.
2. Create `venv` if it does not already exist.
3. Install packages from `requirements.txt`.
4. Start the Flask application and open the dashboard.

The dashboard is available at **http://127.0.0.1:5000**. Keep the command window open while using it; closing that window stops the server.

### Manual setup

From a terminal opened in the project folder:

```bat
py -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
python app.py
```

Then browse to http://127.0.0.1:5000.

## Using the dashboard

1. Open the local dashboard.
2. Fill in the student profile and engagement fields.
3. Click **Analyze Risk**.
4. The dashboard displays the predicted dropout probability and risk category.

The risk bands are based on the model's predicted probability for the dropout class:

- **Low:** probability below 0.30
- **Medium:** probability from 0.30 through 0.70
- **High:** probability above 0.70

The dashboard's category-distribution and feature-importance charts are currently mock/illustrative values; they are not generated from the individual prediction or computed model explanations.

## API

The application exposes `POST /predict`. Send a JSON object for one student or a JSON array of student objects. Use the dataset's column names and category values. The features are:

`gender`, `NationalITy`, `PlaceofBirth`, `StageID`, `GradeID`, `SectionID`, `Topic`, `Semester`, `Relation`, `raisedhands`, `VisITedResources`, `AnnouncementsView`, `Discussion`, `ParentAnsweringSurvey`, `ParentschoolSatisfaction`, and `StudentAbsenceDays`.

Example request:

```bat
curl -X POST http://127.0.0.1:5000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"gender\":\"M\",\"NationalITy\":\"KW\",\"PlaceofBirth\":\"KuwaIT\",\"StageID\":\"lowerlevel\",\"GradeID\":\"G-04\",\"SectionID\":\"A\",\"Topic\":\"IT\",\"Semester\":\"F\",\"Relation\":\"Father\",\"raisedhands\":10,\"VisITedResources\":5,\"AnnouncementsView\":2,\"Discussion\":10,\"ParentAnsweringSurvey\":\"No\",\"ParentschoolSatisfaction\":\"Bad\",\"StudentAbsenceDays\":\"Above-7\"}"
```

Example response:

```json
{
  "probability": 0.42,
  "risk": "Medium"
}
```

For a JSON array, the API returns an array of result objects in the same order. The `probability` is between 0 and 1. Unknown categorical values are accepted by the fitted one-hot encoder; provide all expected feature fields for reliable results.

To run the included example client while the server is running:

```bat
venv\Scripts\python.exe test_api.py
```

## Model workflow

`train.py` performs the following steps:

1. Downloads `xAPI-Edu-Data.csv` from its configured source if the data file is absent.
2. Loads the dataset and removes duplicate rows.
3. Converts the `Class` target to a binary label: `L` (low academic performance in the source data) becomes `1` (treated by this project as dropout); the other classes become `0`.
4. Splits features into numeric and categorical columns. Numeric values are standardized and categorical values are one-hot encoded, with unseen categories ignored.
5. Creates a stratified 80/20 train/test split using `random_state=42`.
6. Trains Logistic Regression, Random Forest, and XGBoost candidates, and compares their recall on the test split.
7. Tunes the highest-recall candidate using a 5-fold `GridSearchCV` scored by recall.
8. Evaluates the tuned pipeline on the held-out test set and saves it as `dropout_model.pkl`.

To retrain the model, activate the project's virtual environment and run:

```bat
venv\Scripts\python.exe train.py
```

Training can take longer than starting the web app because it includes a hyperparameter grid search.

## Accuracy and evaluation

The checked-in `train.py` reports recall, F1-score, ROC-AUC, confusion matrices, and the final test recall; it does not print accuracy. Accuracy was calculated separately using the saved `dropout_model.pkl` on the same stratified 20% held-out split (`random_state=42`) from the deduplicated local CSV:

| Metric | Held-out result |
| --- | ---: |
| Accuracy | **94.79%** (91/96 correct) |
| Recall for dropout class | **88.00%** |
| F1-score for dropout class | **89.80%** |
| ROC-AUC | **97.58%** |
| Held-out samples | 96 |

These are results on this dataset and split, not a guarantee of real-world performance. The dataset is small (478 rows after duplicate removal), and accuracy alone can hide missed dropout cases; recall, precision, class balance, and independent validation should also be considered.

## Author
Darshan Shinde
