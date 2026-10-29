# Binary classification: ProdTaken (0 = no purchase, 1 = purchase).
import pandas as pd
import xgboost as xgb
import joblib
import mlflow
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Use a local MLflow store by default; set MLFLOW_TRACKING_URI to use a remote server.
import os
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
mlflow.set_experiment("mlops-training-experiment")

# CSV artifacts are downloaded by the preceding GitHub Actions job.
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze().astype(int)
ytest = pd.read_csv("ytest.csv").squeeze().astype(int)

numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome"
]
categorical_features = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation"
]

# Keep raw string categories in prep.py and at inference; encode inside pipeline.
num_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
cat_pipe = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))
preprocessor = make_column_transformer(
    (num_pipe, numeric_features),
    (cat_pipe, categorical_features)
)

counts = ytrain.value_counts()
scale_pos_weight = float(counts.get(0, 0) / counts.get(1, 1))
if counts.get(1, 0) == 0 or counts.get(0, 0) == 0:
    raise ValueError("Both classes 0 and 1 are required in the training split")

xgb_model = xgb.XGBClassifier(
    objective="binary:logistic", eval_metric="logloss",
    scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=1
)
model_pipeline = make_pipeline(preprocessor, xgb_model)
param_grid = {
    "xgbclassifier__n_estimators": [50, 100],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.05, 0.1],
}

with mlflow.start_run(run_name="tourism-xgb-classifier"):
    grid_search = GridSearchCV(
        model_pipeline, param_grid, cv=3, n_jobs=-1,
        scoring="f1", error_score="raise"
    )
    grid_search.fit(Xtrain, ytrain)
    results = grid_search.cv_results_
    for i, param_set in enumerate(results["params"]):
        with mlflow.start_run(run_name=f"combination-{i+1}", nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_cv_f1", float(results["mean_test_score"][i]))
            mlflow.log_metric("std_cv_f1", float(results["std_test_score"][i]))

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_f1", float(grid_search.best_score_))
    mlflow.log_param("scale_pos_weight", scale_pos_weight)
    best_model = grid_search.best_estimator_
    print("Best parameters:", grid_search.best_params_)

    # Match the decision threshold in Streamlit.
    threshold = 0.45
    mlflow.log_param("classification_threshold", threshold)
    for name, features, labels in [
        ("train", Xtrain, ytrain), ("test", Xtest, ytest)
    ]:
        proba = best_model.predict_proba(features)[:, list(best_model.classes_).index(1)]
        pred = (proba >= threshold).astype(int)
        mlflow.log_metrics({
            f"{name}_accuracy": accuracy_score(labels, pred),
            f"{name}_precision": precision_score(labels, pred, zero_division=0),
            f"{name}_recall": recall_score(labels, pred, zero_division=0),
            f"{name}_f1": f1_score(labels, pred, zero_division=0),
        })
        print(f"\n{name.title()} classification report:\n", classification_report(labels, pred, zero_division=0))
        print(f"{name.title()} confusion matrix:\n", confusion_matrix(labels, pred))

    # Resolve relative to the script, not the GitHub Actions working directory.
    model_path = Path(__file__).resolve().parents[1] / "deployment" / "tourism_package_prediction.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    assert model_path.is_file() and model_path.stat().st_size > 0
    mlflow.log_artifact(str(model_path), artifact_path="model")
    print(f"Model saved successfully to {model_path}")
