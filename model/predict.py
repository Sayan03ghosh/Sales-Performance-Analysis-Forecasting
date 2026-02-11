# Generate Predictions for All Months
import pandas as pd
import joblib
import os
import numpy as np

# Load data using same logic as training
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')

candidates = ['processed_sales.csv', 'processed_sales_data.csv', 'ml_monthly_sales.csv', 'superstore_sales.csv']
data_path = None

for fn in candidates:
    p = os.path.join(data_dir, fn)
    if os.path.exists(p):
        data_path = p
        print(f'Found: {fn}')
        break

if data_path is None:
    raise FileNotFoundError(f'No data file found. Looked for: {candidates}')

# Load data with encoding fallback
encodings = ['utf-8', 'cp1252', 'latin-1']
for enc in encodings:
    try:
        data = pd.read_csv(data_path, encoding=enc)
        print(f'Loaded with encoding {enc}')
        break
    except:
        continue

# Prepare features (match training schema)
features_path = os.path.join(script_dir, "feature_cols.pkl")
if os.path.exists(features_path):
    feature_cols = joblib.load(features_path)
else:
    feature_cols = ["Year", "Month", "Quarter", "Prev_month_sales"]

missing_cols = [col for col in feature_cols if col not in data.columns]
for col in missing_cols:
    data[col] = 0
X = data[feature_cols].fillna(0)

# Load trained model
model_path = os.path.join(script_dir, 'sales_model.pkl')
model = joblib.load(model_path)

# Generate predictions
data['predicted_sales'] = model.predict(X)

# Save predictions for Power BI
output_path = os.path.join(data_dir, 'sales_predictions.csv')
data.to_csv(output_path, index=False)

print("Predictions generated successfully!")
print(f"Saved to: {output_path}")
