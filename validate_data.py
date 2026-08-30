import os
import pandas as pd

# Path to the registered dataset
DATA_PATH =  "data/SuperKart.csv"

# Expected columns from the SuperKart data dictionary
EXPECTED_COLUMNS = [
    "Product_Id",
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_Type",
    "Product_MRP",
    "Store_Id",
    "Store_Establishment_Year",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Store_Sales_Total"
]

# Check if dataset exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

# Load dataset
df = pd.read_csv(DATA_PATH)

# Check expected columns
missing_columns = [
    col for col in EXPECTED_COLUMNS
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing expected columns: {missing_columns}"
    )

# Validation successful
print("Dataset validation successful!")

# Dataset summary
print("\nDataset Summary")
print("=" * 40)
print("Number of rows   :", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nColumns:")
for col in df.columns:
    print("-", col)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

print("\nFirst 5 Rows:")
print(df.head())
