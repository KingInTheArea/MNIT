# verify_accuracy.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# =======================================================
# 🛠️ HELPER FUNCTION: CLEAN AND SAFE LOADING
# =======================================================
def load_and_clean_csv(file_path, expected_columns):
    # Use header=1 to completely skip the generic row 1 labels and use row 2 as headers
    df = pd.read_csv(file_path, header=1, encoding='latin1')
    
    # Standardize names to clean up whitespace
    df.columns = [str(col).strip() for col in df.columns]
    
    # If the column names still mismatch, apply the safe positional mapping array
    if not any(feat in df.columns for feat in expected_columns):
        df.columns = expected_columns[:len(df.columns)]
        
    return df

# =======================================================
# 📐 PHASE 1: GEOMETRY MODEL VERIFICATION
# =======================================================
clean_cols_1 = [
    'iteration', 'generation', 'category',
    'total_energy', 'discomfort_hours', 'cooling_energy',
    'window_to_wall', 'orientation', 'facade_type', 'shading_type', 'window_open_pct',
    'unnamed'
]

df1 = load_and_clean_csv('Pareto.csv', clean_cols_1)

feature_cols_1 = ['window_to_wall', 'orientation', 'facade_type', 'shading_type', 'window_open_pct']
X1 = df1[feature_cols_1].copy()
X1[['facade_type', 'shading_type']] = OrdinalEncoder().fit_transform(X1[['facade_type', 'shading_type']])
y1 = df1['total_energy']

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)
rf1 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf1.fit(X_train1, y_train1)

print("--- PHASE 1: GEOMETRY MODEL VERIFICATION ---")
print(f"Accuracy Score (R²): {r2_score(y_test1, rf1.predict(X_test1))*100:.2f}%")
print(f"Average Prediction Deviation (MAE): {mean_absolute_error(y_test1, rf1.predict(X_test1)):.2f} kWh")


# =======================================================
# 🧱 PHASE 2: MATERIALS MODEL VERIFICATION
# =======================================================
clean_cols_2 = [
    'iteration', 'generation', 'category',
    'total_energy', 'discomfort_hours', 'cooling_energy',
    'external_wall', 'flat_roof', 'glazing_type', 'partition_wall',
    'unnamed'
]

df2 = load_and_clean_csv('2nd Optimization Results.csv', clean_cols_2)

feature_cols_2 = ['external_wall', 'flat_roof', 'glazing_type', 'partition_wall']
X2 = df2[feature_cols_2].copy()
X2[feature_cols_2] = OrdinalEncoder().fit_transform(X2[feature_cols_2])
y2 = df2['total_energy']

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)
rf2 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf2.fit(X_train2, y_train2)

print("\n--- PHASE 2: MATERIALS MODEL VERIFICATION ---")
print(f"Accuracy Score (R²): {r2_score(y_test2, rf2.predict(X_test2))*100:.2f}%")
print(f"Average Prediction Deviation (MAE): {mean_absolute_error(y_test2, rf2.predict(X_test2)):.2f} kWh")
