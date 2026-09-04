import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"

# Every column the downstream pipeline depends on. CustomerID is deliberately
# excluded: it is dropped during preparation and is not required here.
EXPECTED_COLUMNS = [
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
]

data = pd.read_csv(DATA_PATH)

# Validation: fail loudly if the schema does not match what the pipeline expects.
missing_columns = [column for column in EXPECTED_COLUMNS if column not in data.columns]
if missing_columns:
    raise ValueError(f"Missing expected columns: {missing_columns}")

if data.empty:
    raise ValueError(f"{DATA_PATH} contains no rows.")

# Summary of the registered dataset.
print("Dataset registered successfully.")
print(f"Source      : {DATA_PATH}")
print(f"Rows        : {data.shape[0]}")
print(f"Columns     : {data.shape[1]}")
print(f"Duplicates  : {data.duplicated().sum()}")

print("\nColumn data types:")
print(data.dtypes.to_string())

print("\nMissing values per column:")
print(data.isnull().sum().to_string())

print("\nTarget distribution (ProdTaken):")
print(data["ProdTaken"].value_counts().to_string())
print(data["ProdTaken"].value_counts(normalize=True).round(4).to_string())
