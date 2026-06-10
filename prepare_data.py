import pandas as pd
import json
import os

# Load dataset
df = pd.read_csv("data/train-chennai-sale.csv")

# Identify categorical and numerical features
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
num_cols = df.select_dtypes(include=['number']).columns.tolist()

# Choose target and locality columns
target = 'SALES_PRICE'
locality_col = 'AREA'

# Handle missing values
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

# Save defaults
numeric_medians = df[num_cols].median().to_dict()
categorical_modes = df[cat_cols].mode().iloc[0].to_dict()

defaults = {
    "numeric_medians": numeric_medians,
    "categorical_modes": categorical_modes
}

os.makedirs("app", exist_ok=True)
with open("app/defaults.json", "w") as f:
    json.dump(defaults, f, indent=2)

# Save locality statistics
locality_col = "AREA"
if locality_col in df.columns:
    locality_stats = (
        df.groupby(locality_col)[target]
        .agg(["median", "min", "max"])
        .reset_index()
        .sort_values("median", ascending=False)
    )
    locality_stats.to_csv("app/locality_stats.csv", index=False)
    print("✅ Locality stats saved to app/locality_stats.csv")
else:
    print(f"⚠️ {locality_col} column not found. Skipping locality stats.")
