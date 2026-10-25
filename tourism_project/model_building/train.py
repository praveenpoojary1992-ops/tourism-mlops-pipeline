"""Model building: tune XGBoost, track every trial with MLflow, evaluate and save the best model."""
import os
import urllib.request
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_FILE = "best_tourism_model_v1.joblib"
MODEL_PATH = Path(__file__).resolve().parents[1] / "deployment" / MODEL_FILE
THRESHOLD = 0.45  # classification threshold (the Streamlit app uses the same value)

numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting", "NumberOfFollowups",
    "PreferredPropertyStar", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]
categorical_features = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched", "MaritalStatus", "Designation",
]


def configure_mlflow():
    """Use the MLflow server started by the workflow; fall back to a local ./mlruns store."""
    uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    if uri.startswith("http"):
        try:
            urllib.request.urlopen(uri, timeout=5)
        except Exception:
            print(f"MLflow server not reachable at {uri} - using local ./mlruns instead")
            uri = "file:./mlruns"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("tourism-package-prediction")
    print("MLflow tracking URI:", mlflow.get_tracking_uri())


def evaluate(model, X, y):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    return pred, {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }


def main():
    # 1. Load the train/test splits produced by the data-prep job
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
    ytest = pd.read_csv("ytest.csv").squeeze("columns")
    print("Loaded splits:", Xtrain.shape, Xtest.shape)

    # 2. Define the model: preprocessing + XGBoost (class weight handles the imbalance)
    class_weight = (ytrain == 0).sum() / (ytrain == 1).sum()
    preprocessor = make_column_transformer(
        (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric_features),
        (make_pipeline(SimpleImputer(strategy="most_frequent"),
                       OneHotEncoder(handle_unknown="ignore")), categorical_features),
    )
    xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42, n_jobs=-1)
    pipeline = make_pipeline(preprocessor, xgb_model)

    # 3. Hyperparameter grid and tuning
    param_grid = {
        "xgbclassifier__n_estimators": [75, 125],
        "xgbclassifier__max_depth": [3, 5],
        "xgbclassifier__learning_rate": [0.05, 0.1],
        "xgbclassifier__colsample_bytree": [0.6, 0.8],
        "xgbclassifier__reg_lambda": [1, 5],
    }
    configure_mlflow()
    with mlflow.start_run(run_name="xgboost_grid_search"):
        grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="f1", n_jobs=-1)
        grid.fit(Xtrain, ytrain)

        # 4. Log every tuned parameter combination as a nested run
        results = grid.cv_results_
        for i, params in enumerate(results["params"]):
            with mlflow.start_run(run_name=f"trial_{i:02d}", nested=True):
                mlflow.log_params({k.replace("xgbclassifier__", ""): v for k, v in params.items()})
                mlflow.log_metric("cv_mean_f1", results["mean_test_score"][i])
                mlflow.log_metric("cv_std_f1", results["std_test_score"][i])
        print(f"Logged {len(results['params'])} tuning trials to MLflow")

        # 5. Evaluate the best model on train and test data
        best_model = grid.best_estimator_
        mlflow.log_params({k.replace("xgbclassifier__", "best_"): v
                           for k, v in grid.best_params_.items()})
        mlflow.log_param("threshold", THRESHOLD)
        mlflow.log_metric("best_cv_f1", grid.best_score_)

        train_pred, train_metrics = evaluate(best_model, Xtrain, ytrain)
        test_pred, test_metrics = evaluate(best_model, Xtest, ytest)
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        print("\nBest parameters:", grid.best_params_)
        print(f"Best CV F1: {grid.best_score_:.4f}")
        print(f"\nClassification report - TRAIN (threshold={THRESHOLD})")
        print(classification_report(ytrain, train_pred))
        print(f"Classification report - TEST (threshold={THRESHOLD})")
        print(classification_report(ytest, test_pred))
        print("Test metrics:", {k: round(v, 4) for k, v in test_metrics.items()})

        # 6. Save the best model so the workflow can commit it to the repository
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)
        mlflow.log_artifact(str(MODEL_PATH))
        print(f"\nBest model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
