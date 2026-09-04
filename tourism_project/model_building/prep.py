import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
TARGET = "ProdTaken"

# Load the dataset from the repository data folder.
tourism_data = pd.read_csv(DATA_PATH)

# Remove the CSV index and customer identifier; neither is a predictive feature.
tourism_data = tourism_data.drop(columns=["Unnamed: 0", "CustomerID"], errors="ignore")

if TARGET not in tourism_data.columns:
    raise ValueError(f"Target column '{TARGET}' was not found in {DATA_PATH}")

features = tourism_data.drop(columns=[TARGET])
target = tourism_data[TARGET]

# Preserve the target distribution in both splits.
features_train, features_test, target_train, target_test = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=42,
    stratify=target,
)

# These files are uploaded by the workflow as the data-splits artifact.
features_train.to_csv("Xtrain.csv", index=False)
features_test.to_csv("Xtest.csv", index=False)
target_train.to_csv("ytrain.csv", index=False)
target_test.to_csv("ytest.csv", index=False)

print("Data preparation completed successfully.")
print(f"Training rows: {len(features_train)}")
print(f"Testing rows: {len(features_test)}")
