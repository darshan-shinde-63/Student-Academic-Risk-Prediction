import os
import requests
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, f1_score, roc_auc_score, confusion_matrix

DATA_URL = "https://raw.githubusercontent.com/basilatawneh/Students-Academic-Performance-Dataset-xAPI-Edu-Data-/master/xAPI-Edu-Data.csv"
DATA_FILE = "xAPI-Edu-Data.csv"

def download_data():
    if not os.path.exists(DATA_FILE):
        print(f"Downloading dataset from {DATA_URL}...")
        response = requests.get(DATA_URL)
        response.raise_for_status()
        with open(DATA_FILE, 'wb') as f:
            f.write(response.content)
        print("Download complete.")
    else:
        print("Dataset already exists locally.")

def main():
    download_data()

    # 1. Load Data
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    
    # EDA Summaries
    print("\n--- Exploratory Data Analysis ---")
    print(f"Shape: {df.shape}")
    print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0])
    
    # 2. Preprocessing
    print("\nDropping duplicates...")
    initial_shape = df.shape
    df = df.drop_duplicates()
    print(f"Dropped {initial_shape[0] - df.shape[0]} duplicates.")
    
    print("\nClass distribution before conversion:")
    print(df['Class'].value_counts())

    # Convert Target: L -> 1 (Dropout), others -> 0
    df['Dropout'] = (df['Class'] == 'L').astype(int)
    print("\nTarget distribution (1=Dropout, 0=Not):")
    print(df['Dropout'].value_counts())

    # Prepare features and target
    X = df.drop(['Class', 'Dropout'], axis=1)
    y = df['Dropout']

    # Identify categorical vs numerical features
    num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    print("\nNumerical features:", num_features)
    print("Categorical features:", cat_features)

    # 3. Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])

    # 4. Model Training & Evaluation Setup
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    best_recall = 0
    best_model_name = ""

    print("\n--- Model Evaluation ---")
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        # For ROC-AUC, we need probabilities of the positive class
        if hasattr(pipeline, "predict_proba"):
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
        else:
            auc = np.nan
            
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\n{name}:")
        print(f"  Recall:  {recall:.4f}")
        print(f"  F1-score:{f1:.4f}")
        print(f"  ROC-AUC: {auc:.4f}")
        print(f"  Confusion Matrix:\n{cm}")
        
        if recall > best_recall:
            best_recall = recall
            best_model_name = name

    print(f"\nBest model based on Recall: {best_model_name}")

    # 5. Optimization (GridSearchCV on best model)
    print(f"\n--- Optimizing {best_model_name} ---")
    
    # Define param grids
    param_grids = {
        'Logistic Regression': {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__solver': ['liblinear', 'lbfgs']
        },
        'Random Forest': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 5, 10],
            'classifier__min_samples_split': [2, 5]
        },
        'XGBoost': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.1, 0.2]
        }
    }

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor), 
        ('classifier', models[best_model_name])
    ])
    
    param_grid = param_grids.get(best_model_name, {})
    
    grid_search = GridSearchCV(
        pipeline, param_grid, scoring='recall', cv=5, n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation Recall: {grid_search.best_score_:.4f}")
    
    best_pipeline = grid_search.best_estimator_
    
    # Final eval on test set
    final_pred = best_pipeline.predict(X_test)
    final_recall = recall_score(y_test, final_pred)
    print(f"\nFinal Test Recall with optimized model: {final_recall:.4f}")

    # 6. Save Model
    model_filename = 'dropout_model.pkl'
    joblib.dump(best_pipeline, model_filename)
    print(f"\nModel saved to {model_filename}")

if __name__ == "__main__":
    main()
