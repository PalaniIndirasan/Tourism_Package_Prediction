# Import Required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split

# Load Tourism Dataset
df = pd.read_csv("tourism_project/data/tourism.csv")
print("Dataset loaded successfully.")
print("Original Dataset Shape:", df.shape)

# Identify Duplicate Customer Records
if df["CustomerID"].duplicated().any():
    raise ValueError("Duplicate CustomerID values detected.")

# Remove Unnecessary Identifier Columns
df = df.drop(
    columns=["CustomerID", "Unnamed: 0"],
    errors="ignore"
)

# Analyze Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Standardize Categorical Variables
categorical_columns = df.select_dtypes(
    include=["object", "string"]
).columns

for col in categorical_columns:
    df[col] = df[col].str.strip()

# Standardize Gender
df["Gender"] = df["Gender"].replace({
    "Fe Male": "Female"
})

# Standardize Marital Status
df["MaritalStatus"] = df["MaritalStatus"].replace({
    "Unmarried": "Single"
})

# Validate Numerical Data Types
numeric_features = [
    "Age", "CityTier", "DurationOfPitch",
    "NumberOfPersonVisiting", "NumberOfFollowups",
    "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting",
    "MonthlyIncome"
]

for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Validate Target Variable
if df["ProdTaken"].isnull().any():
    raise ValueError("Target variable contains missing values.")

if not df["ProdTaken"].isin([0, 1]).all():
    raise ValueError("ProdTaken must contain only 0 and 1.")

# Separate Features and Target Variable
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# Split Dataset into Training and Testing Sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Save Training and Testing Datasets
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Verify Data Preparation
print("\nData preparation completed successfully.")
print("Training Shape:", Xtrain.shape)
print("Testing Shape:", Xtest.shape)
print("Training Target Distribution:")
print(ytrain.value_counts(normalize=True).round(3))
print("Testing Target Distribution:")
print(ytest.value_counts(normalize=True).round(3))
