# Import Required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split

# Load Tourism Dataset and Inspect Data Structure
df = pd.read_csv("tourism_project/data/tourism.csv", on_bad_lines="skip")
print("Dataset loaded successfully.")
print("\nDataset Shape:", df.shape)
print("\nDataset Information:")
df.info()

# Remove Unnecessary Identifier Column
df = df.drop(columns=["CustomerID"])
print("\nOriginal Dataset Shape:", df.shape)

# Identify and Remove Duplicate Records
print("\n--- Duplicate Records ---")
duplicate_count = df.duplicated().sum()
print("Number of duplicate records:", duplicate_count)
if duplicate_count > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print("Duplicate records removed.")
else:
    print("No duplicate records found.")

# Analyze Missing and Null Values
print("\n--- Missing Value Analysis ---")
missing_values = pd.DataFrame({
    "Missing_Count": df.isnull().sum(),
    "Missing_Percentage": (df.isnull().mean() * 100).round(2)
})
print(missing_values)

# Clean and Standardize Categorical Variables
print("\n--- Categorical Data Cleaning ---")
categorical_columns = df.select_dtypes(include=["object", "string"]).columns
# Remove leading and trailing spaces
for col in categorical_columns:
    df[col] = df[col].str.strip()
# Standardize Gender values
df["Gender"] = df["Gender"].replace({
    "Fe Male": "Female",
    "Female": "Female"
})
# Standardize MaritalStatus values
df["MaritalStatus"] = df["MaritalStatus"].replace({
    "Unmarried": "Single",
    "Single": "Single"
})
print("\nGender Distribution:")
print(df["Gender"].value_counts(dropna=False))
print("\nMarital Status Distribution:")
print(df["MaritalStatus"].value_counts(dropna=False))

# Define Numerical and Categorical Features
print("\n--- Data Type Validation ---")
numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome"
]
categorical_features = [
    "TypeofContact", "Occupation", "Gender",
    "ProductPitched", "MaritalStatus", "Designation"
]

# Validate and Convert Numerical Data Types
for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")
print(df.dtypes)

# Detect Anomalies and Invalid Numerical Values
print("\n--- Anomaly Detection ---")
anomaly_rules = {
    "Age": df["Age"].notna() & (
        (df["Age"] < 18) | (df["Age"] > 100)
    ),
    "DurationOfPitch": df["DurationOfPitch"].notna() & (
        df["DurationOfPitch"] < 0
    ),
    "MonthlyIncome": df["MonthlyIncome"].notna() & (
        df["MonthlyIncome"] < 0
    ),
    "NumberOfTrips": df["NumberOfTrips"].notna() & (
        df["NumberOfTrips"] < 0
    ),
    "NumberOfFollowups": df["NumberOfFollowups"].notna() & (
        df["NumberOfFollowups"] < 0
    ),
    "NumberOfPersonVisiting": df["NumberOfPersonVisiting"].notna() & (
        df["NumberOfPersonVisiting"] < 1
    ),
    "NumberOfChildrenVisiting": df["NumberOfChildrenVisiting"].notna() & (
        df["NumberOfChildrenVisiting"] < 0
    ),
    "PreferredPropertyStar": df["PreferredPropertyStar"].notna() & (
        ~df["PreferredPropertyStar"].between(1, 5)
    )
}
for column, mask in anomaly_rules.items():
    print(f"{column}: {mask.sum()} suspicious values")

# Identify Potential Outliers Using the IQR Method
print("\n--- IQR Outlier Analysis ---")
continuous_features = [
    "Age", "DurationOfPitch", "NumberOfFollowups",
    "NumberOfTrips", "MonthlyIncome"
]
for col in continuous_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[
        (df[col] < lower_bound) |
        (df[col] > upper_bound)
    ]
    print(f"{col}: {len(outliers)} potential outliers")

# Display Final Data Quality Summary
print("\n--- Final Data Quality Summary ---")
print("Dataset Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicate Records:")
print(df.duplicated().sum())
print("\nNumerical Summary:")
print(df.describe().T)
print("\nCategorical Summary:")
print(df[categorical_features].describe().T)

# Separate Predictor Features and Target Variable
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# Split Dataset into Training (80%) and Testing (20%) Sets
# Stratification preserves the target class distribution
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Export Training and Testing Datasets as CSV Files
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)
print("Data prepared: train/test splits written.")
