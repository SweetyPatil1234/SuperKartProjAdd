import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "data/SuperKart.csv"
OUTPUT_DIR = "data/processed"
TRAIN_PATH = os.path.join(OUTPUT_DIR, "train.csv")
TEST_PATH = os.path.join(OUTPUT_DIR, "test.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print("Dataset loaded successfully.")
print("Original shape:", df.shape)

df = df.drop(columns=["Product_Id"])
print("Shape after removing unnecessary columns:", df.shape)

df = df.drop_duplicates()
print("Shape after removing duplicates:", df.shape)

df['Product_Weight'].fillna(df['Product_Weight'].median(), inplace=True)
df['Store_Size'].fillna(df['Store_Size'].mode()[0], inplace=True)

print("Missing values after cleaning:")
print(df.isnull().sum())

train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)

train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)

print("\nData preparation completed successfully.")
print("Training set shape:", train_df.shape)
print("Testing set shape :", test_df.shape)
print("\nTraining data saved to:", TRAIN_PATH)
print("Testing data saved to :", TEST_PATH)
