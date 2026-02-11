# Load ML Dataset
import os
import pandas as pd
import joblib

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "data", "ml_monthly_sales.csv")
data = pd.read_csv(data_path)

feature_cols = ["Year", "Month", "Quarter", "Prev_month_sales"]
X = data[feature_cols]
y = data["Sales"]

# Train/Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Train the model   Model 1: Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

lr = LinearRegression()
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

lr_mae = mean_absolute_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)
print("Linear Regression MAE:", lr_mae)
print("Linear Regression R2:", lr_r2)

# Model 2: Random Forest
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)
print("Random Forest MAE:", rf_mae)
print("Random Forest R2:", rf_r2)

# Select Best Model (lower MAE wins; tie-breaker: higher R2)
if (rf_mae < lr_mae) or (rf_mae == lr_mae and rf_r2 > lr_r2):
    best_model = rf
    best_name = "Random Forest"
    best_mae = rf_mae
    best_r2 = rf_r2
else:
    best_model = lr
    best_name = "Linear Regression"
    best_mae = lr_mae
    best_r2 = lr_r2

print(f"Best Model: {best_name} (MAE={best_mae}, R2={best_r2})")
# Save the model and feature list
model_path = os.path.join(script_dir, "sales_model.pkl")
features_path = os.path.join(script_dir, "feature_cols.pkl")
joblib.dump(best_model, model_path)
joblib.dump(feature_cols, features_path)

