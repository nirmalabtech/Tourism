import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

XTRAIN_PATH = "Xtrain.csv"
XTEST_PATH = "Xtest.csv"
YTRAIN_PATH = "ytrain.csv"
YTEST_PATH = "ytest.csv"
MODEL_PATH = "tourism_project/deployment/best_tourism_model.joblib"
CV_RESULTS_PATH = "cv_results.csv"
TARGET_NAME = "ProdTaken"
CLASSIFICATION_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# 1. Load the train/test splits produced by prep.py (the workflow artifact).
# ---------------------------------------------------------------------------
X_train = pd.read_csv(XTRAIN_PATH)
X_test = pd.read_csv(XTEST_PATH)
y_train = pd.read_csv(YTRAIN_PATH).squeeze("columns")
y_test = pd.read_csv(YTEST_PATH).squeeze("columns")

print(f"Train shape: {X_train.shape}   Test shape: {X_test.shape}")

numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=["number"]).columns.tolist()
print(f"Numeric features    ({len(numeric_features)}): {numeric_features}")
print(f"Categorical features({len(categorical_features)}): {categorical_features}")

# ---------------------------------------------------------------------------
# 2. Preprocessing, fitted inside the pipeline so no test data leaks into it.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 3. Model and hyperparameter grid.
# ---------------------------------------------------------------------------
model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=1,
)

model_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

# Only about 18% of customers buy the package. scale_pos_weight lets the search
# decide how much to up-weight that minority class, which matters here because
# the business cost of missing a buyer is higher than a wasted sales call.
parameter_grid = {
    "classifier__n_estimators": [100, 150],
    "classifier__max_depth": [3, 5],
    "classifier__learning_rate": [0.05, 0.1],
    "classifier__subsample": [0.8, 1.0],
    "classifier__scale_pos_weight": [1, 4],
}

# ---------------------------------------------------------------------------
# 4. Experiment tracking with MLflow.
# ---------------------------------------------------------------------------
# A SQLite backend is used instead of the legacy "file:./mlruns" store: newer
# MLflow releases refuse the file store outright, and a single .db file is much
# easier to upload as a workflow artifact so the runs survive the CI runner.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("tourism-package-prediction")
print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")

with mlflow.start_run(run_name="xgboost_grid_search") as parent_run:
    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=parameter_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)

    # --- Log EVERY tuned combination, not just the winner -------------------
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results.to_csv(CV_RESULTS_PATH, index=False)

    for index, row in cv_results.iterrows():
        with mlflow.start_run(nested=True, run_name=f"candidate_{index:02d}"):
            mlflow.log_params(row["params"])
            mlflow.log_metric("mean_cv_roc_auc", row["mean_test_score"])
            mlflow.log_metric("std_cv_roc_auc", row["std_test_score"])
            mlflow.log_metric("mean_train_roc_auc", row["mean_train_score"])
            mlflow.log_metric("rank_cv_roc_auc", row["rank_test_score"])

    print(f"Logged {len(cv_results)} tuned parameter combinations to MLflow.")

    # --- Evaluate the best model on the held-out test set -------------------
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

    # --- Log the winning configuration on the parent run --------------------
    mlflow.log_params({key: str(value) for key, value in grid_search.best_params_.items()})
    mlflow.log_param("param_grid", str(parameter_grid))
    mlflow.log_param("n_candidates", len(cv_results))
    mlflow.log_param("classification_threshold", CLASSIFICATION_THRESHOLD)
    mlflow.log_metrics(metrics)
    mlflow.log_artifact(CV_RESULTS_PATH)

    # cloudpickle is requested explicitly because recent MLflow versions default
    # to a serializer that rejects XGBoost objects as untrusted types.
    try:
        mlflow.sklearn.log_model(
            best_model, name="model", serialization_format="cloudpickle"
        )  # MLflow >= 3.x
    except TypeError:
        mlflow.sklearn.log_model(
            best_model, artifact_path="model", serialization_format="cloudpickle"
        )  # MLflow 2.x

    # --- Save the best model so the workflow can commit it to the repo ------
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    print("\nBest parameters:")
    for key, value in grid_search.best_params_.items():
        print(f"  {key}: {value}")

    print("\nEvaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\nClassification report (test set):")
    print(classification_report(y_test, test_predictions, digits=3, zero_division=0))

    print("Confusion matrix (rows = actual, cols = predicted):")
    print(confusion_matrix(y_test, test_predictions))

    print(f"\nMLflow run ID: {parent_run.info.run_id}")
    print(f"Model saved to {MODEL_PATH}")
