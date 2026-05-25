# verify_accuracy.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# Verify Layout Model
df1 = pd.read_csv('Pareto.csv', header=[0, 1], encoding='latin1')
df1.columns = ['iteration','generation','category','total_energy','discomfort_hours','cooling_energy','window_to_wall','orientation','facade_type','shading_type','window_open_pct','unnamed'][:len(df1.columns)]

X1 = df1[['window_to_wall', 'orientation', 'facade_type', 'shading_type', 'window_open_pct']].copy()
X1[['facade_type', 'shading_type']] = OrdinalEncoder().fit_transform(X1[['facade_type', 'shading_type']])
y1 = df1['total_energy']

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)
rf1 = RandomForestRegressor(n_estimators=100, random_state=42)
rf1.fit(X_train1, y_train1)

print("--- PHASE 1: GEOMETRY MODEL VERIFICATION ---")
print(f"Accuracy Score (R²): {r2_score(y_test1, rf1.predict(X_test1))*100:.2f}%")
print(f"Average Prediction Deviation (MAE): {mean_absolute_error(y_test1, rf1.predict(X_test1)):.2f} kWh")

# Verify Materials Model
df2 = pd.read_csv('2nd Optimization Results.csv', header=[0, 1], encoding='latin1')
df2.columns = ['iteration','generation','category','total_energy','discomfort_hours','cooling_energy','external_wall','flat_roof','glazing_type','partition_wall','unnamed'][:len(df2.columns)]

X2 = df2[['external_wall', 'flat_roof', 'glazing_type', 'partition_wall']].copy()
X2[['external_wall', 'flat_roof', 'glazing_type', 'partition_wall']] = OrdinalEncoder().fit_transform(X2[['external_wall', 'flat_roof', 'glazing_type', 'partition_wall']])
y2 = df2['total_energy']

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)
rf2 = RandomForestRegressor(n_estimators=100, random_state=42)
rf2.fit(X_train2, y_train2)

print("\n--- PHASE 2: MATERIALS MODEL VERIFICATION ---")
print(f"Accuracy Score (R²): {r2_score(y_test2, rf2.predict(X_test2))*100:.2f}%")
print(f"Average Prediction Deviation (MAE): {mean_absolute_error(y_test2, rf2.predict(X_test2)):.2f} kWh")