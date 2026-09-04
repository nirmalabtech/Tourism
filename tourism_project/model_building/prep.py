import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
TARGET = "ProdTaken"

# ---------------------------------------------------------------------------
# 1. Load the dataset directly from the repository data folder.
# ---------------------------------------------------------------------------
tourism_data = pd.read_csv(DATA_PATH)
print(f"Raw dataset shape: {tourism_data.shape}")

# ---------------------------------------------------------------------------
# 2. Data cleaning
# ---------------------------------------------------------------------------

# 2a. Drop the exported CSV index and the customer identifier. Neither carries
#     predictive signal; CustomerID is a unique key that would only add noise.
tourism_data = tourism_data.drop(columns=["Unnamed: 0", "CustomerID"], errors="ignore")

if TARGET not in tourism_data.columns:
    raise ValueError(f"Target column '{TARGET}' was not found in {DATA_PATH}")

# 2b. Remove exact duplicate records.
duplicate_count = tourism_data.duplicated().sum()
tourism_data = tourism_data.drop_duplicates()
print(f"Duplicate rows removed: {duplicate_count}")

# 2c. Fix inconsistent category labels captured during data entry.
#     "Fe Male" is a typo for "Female"; without this fix the encoder would learn
#     a third gender category that the Streamlit app can never produce.
if "Gender" in tourism_data.columns:
    tourism_data["Gender"] = tourism_data["Gender"].replace({"Fe Male": "Female"})

#     "Unmarried" and "Single" describe the same status; merging them keeps the
#     category set consistent between training and the deployed app.
if "MaritalStatus" in tourism_data.columns:
    tourism_data["MaritalStatus"] = tourism_data["MaritalStatus"].replace(
        {"Unmarried": "Single"}
    )

# 2d. Report the remaining missing values. They are deliberately NOT imputed
#     here: imputation is fitted inside the modelling pipeline in train.py so
#     that the imputer only ever learns from the training fold. Imputing before
#     the split would leak test information into the training data.
print("\nMissing values per column:")
print(tourism_data.isnull().sum())

print("\nCleaned dataset shape:", tourism_data.shape)
print("\nTarget distribution:")
print(tourism_data[TARGET].value_counts(normalize=True).round(4))

# ---------------------------------------------------------------------------
# 3. Split into train and test sets and save them locally.
# ---------------------------------------------------------------------------
features = tourism_data.drop(columns=[TARGET])
target = tourism_data[TARGET]

# Stratify so the ~18% positive rate is preserved in both splits.
features_train, features_test, target_train, target_test = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=42,
    stratify=target,
)

# These four files are uploaded by the workflow as the "data-splits" artifact
# and downloaded again by the model-training job.
features_train.to_csv("Xtrain.csv", index=False)
features_test.to_csv("Xtest.csv", index=False)
target_train.to_csv("ytrain.csv", index=False)
target_test.to_csv("ytest.csv", index=False)

print("\nData preparation completed successfully.")
print(f"Training rows: {len(features_train)}")
print(f"Testing rows : {len(features_test)}")
print(f"Features     : {features_train.shape[1]}")
