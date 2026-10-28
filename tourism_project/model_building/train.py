"""Train binary tourism purchase classifier and persist full pipeline."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import mlflow
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_DIR = ROOT / "deployment"
DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)
model_path = DEPLOYMENT_DIR / "tourism_package_prediction.joblib"

Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").iloc[:, 0].astype(int)
ytest = pd.read_csv("ytest.csv").iloc[:, 0].astype(int)

numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome"
]
categorical_features = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation"
]

# Both training and Streamlit use identical categorical representations.
for data in [Xtrain, Xtest]:
    for col in categorical_features:
        data[col] = data[col].astype("object").where(data[col].notna(), np.nan)
        data[col] = data[col].map(lambda x: x.strip() if isinstance(x, str) else x)
    data["Gender"] = data["Gender"].replace({"Fe Male": "Female"})
    data["MaritalStatus"] = data["MaritalStatus"].replace({"Unmarried": "Single"})
    for col in numeric_features:
        data[col] = pd.to_numeric(data[col], errors="coerce")

preprocessor = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numeric_features),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_features),
])

counts = ytrain.value_counts()
weight = float(counts.get(0, 0) / counts.get(1, 1))
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        random_state=42, n_jobs=1, scale_pos_weight=weight
    )),
])
param_grid = {
    "classifier__n_estimators": [50, 100],
    "classifier__max_depth": [3, 5],
    "classifier__learning_rate": [0.05, 0.1],
}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Local MLflow tracking works in CI without a separate server.
mlflow.set_tracking_uri((ROOT / "mlruns").resolve().as_uri())
mlflow.set_experiment("tourism-package-classification")
with mlflow.start_run(run_name="xgboost-gridsearch"):
    grid = GridSearchCV(
        model_pipeline, param_grid, cv=cv, scoring="f1",
        n_jobs=-1, error_score="raise"
    )
    grid.fit(Xtrain, ytrain)
    best_model = grid.best_estimator_
    for i, params in enumerate(grid.cv_results_["params"]):
        with mlflow.start_run(run_name=f"candidate-{i+1}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("mean_cv_f1", float(grid.cv_results_["mean_test_score"][i]))
            mlflow.log_metric("std_cv_f1", float(grid.cv_results_["std_test_score"][i]))
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("best_cv_f1", float(grid.best_score_))

    threshold = 0.45
    for name, X, y in [("train", Xtrain, ytrain), ("test", Xtest, ytest)]:
        prob = best_model.predict_proba(X)[:, 1]
        pred = (prob >= threshold).astype(int)
        mlflow.log_metric(f"{name}_accuracy", accuracy_score(y, pred))
        mlflow.log_metric(f"{name}_f1", f1_score(y, pred, zero_division=0))
        print(f"{name} report:\n{classification_report(y, pred, zero_division=0)}")

    # IMPORTANT: joblib.dump needs the parent directory to exist.
    joblib.dump(best_model, model_path)
    assert model_path.is_file() and model_path.stat().st_size > 0
    mlflow.log_artifact(str(model_path), artifact_path="model")
    print(f"Saved model: {model_path} ({model_path.stat().st_size:,} bytes)")
