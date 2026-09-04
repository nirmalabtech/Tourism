import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"
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
missing_columns = [column for column in EXPECTED_COLUMNS if column not in data.columns]

if missing_columns:
    raise ValueError(f"Missing expected columns: {missing_columns}")

print("Dataset registered successfully.")
print(f"Rows: {data.shape[0]}")
print(f"Columns: {data.shape[1]}")
print("Target distribution:")
print(data["ProdTaken"].value_counts())
