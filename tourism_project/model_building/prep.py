"""Inspect and split data. Learned preprocessing happens in train.py."""
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "tourism.csv")
df = df.drop(columns=["CustomerID", "Unnamed: 0"], errors="ignore")
print("Original rows:", len(df))
print("Exact duplicate records (review before removing):", df.duplicated().sum())
print("Missing values:\n", df.isna().sum())

# Normalize known spelling variations; leave raw categorical strings for
# one-hot encoding in the trained model. No label encoding here.
for col in ["Gender", "MaritalStatus"]:
    df[col] = df[col].astype("string").str.strip()
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

# Report anomalies; don't remove unusual but potentially valid customers.
for col in ["Age", "DurationOfPitch", "MonthlyIncome", "NumberOfTrips"]:
    v = pd.to_numeric(df[col], errors="coerce")
    print(f"{col}: negative values = {(v < 0).sum()}")

X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"].astype(int)
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
for name, obj in [("Xtrain", Xtrain), ("Xtest", Xtest),
                  ("ytrain", ytrain), ("ytest", ytest)]:
    obj.to_csv(f"{name}.csv", index=False)
print("Train/test CSV artifacts created.")
