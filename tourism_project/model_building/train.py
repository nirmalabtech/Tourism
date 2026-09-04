import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


XTRAIN_PATH = "Xtrain.csv"
XTEST_PATH = "Xtest.csv"
YTRAIN_PATH = "ytrain.csv"
YTEST_PATH = "ytest.csv"
MODEL_PATH = "tourism_project/deployment/best_tourism_model.joblib"
TARGET_NAME = "ProdTaken"
CLASSIFICATION_THRESHOLD = 0.5

# Load the train/test splits created by prep.py.
X_train = pd.read_csv(XTRAIN_PATH)
X_test = pd.read_csv(XTEST_PATH)
y_train = pd.read_csv(YTRAIN_PATH).squeeze("columns")
y_test = pd.read_csv(YTEST_PATH).squeeze("columns")

numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=["number"]).columns.tolist()

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)
categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)

model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=1,
)

model_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", model)]
)

parameter_grid = {
    "classifier__n_estimators": [100, 150],
    "classifier__max_depth": [3, 5],
    "classifier__learning_rate": [0.05, 0.1],
    "classifier__subsample": [0.8, 1.0],
}

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("tourism-package-prediction")

with mlflow.start_run() as run:
    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=parameter_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        refit=True,
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    test_probabilities = best_model.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= CLASSIFICATION_THRESHOLD).astype(int)

    metrics = {
        "cv_roc_auc": grid_search.best_score_,
        "test_accuracy": accuracy_score(y_test, test_predictions),
        "test_precision": precision_score(y_test, test_predictions, zero_division=0),
        "test_recall": recall_score(y_test, test_predictions, zero_division=0),
        "test_f1": f1_score(y_test, test_predictions, zero_division=0),
        "test_roc_auc": roc_auc_score(y_test, test_probabilities),
    }

    mlflow.log_params({key: str(value) for key, value in grid_search.best_params_.items()})
    mlflow.log_param("classification_threshold", CLASSIFICATION_THRESHOLD)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(best_model, "model")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    print("Best parameters:", grid_search.best_params_)
    print("Evaluation metrics:", metrics)
    print("MLflow run ID:", run.info.run_id)
    print(f"Model saved to {MODEL_PATH}")
