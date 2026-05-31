import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder

# ==========================================
# 🛠️ HELPER FUNCTION: SKIP UNWANTED ROW 1
# ==========================================
def load_and_clean_csv(file_path, expected_columns):
    # header=1 tells pandas to completely ignore row 1 and use row 2 as column names
    df = pd.read_csv(file_path, header=1, encoding='latin1')
    
    # Standardize names to clean up whitespace or weird characters
    df.columns = [str(col).strip() for col in df.columns]
    
    # If the column names still mismatch, slice and force assign your specific names safely
    if not any(feat in df.columns for feat in expected_columns):
        print(f"⚠️ Column text mismatch in {file_path}. Applying safe forced mapping...")
        df.columns = expected_columns[:len(df.columns)]
        
    return df

# ==========================================
# 🛠️ PROCESS DATASET 1: LAYOUT & ORIENTATION
# ==========================================
print("⚡ Loading Dataset 1 (Layout & Orientation)...")

clean_cols_1 = [
    'iteration', 'generation', 'category',
    'total_energy', 'discomfort_hours', 'cooling_energy',
    'window_to_wall', 'orientation', 'facade_type', 'shading_type', 'window_open_pct',
    'unnamed'
]

# Pass the clean columns we expect down into the loader
df1 = load_and_clean_csv('Pareto.csv', clean_cols_1)
print("Verified columns in Dataset 1:", df1.columns.tolist())

feature_cols_1 = ['window_to_wall', 'orientation', 'facade_type', 'shading_type', 'window_open_pct']
X1 = df1[feature_cols_1].copy()

encoder_1 = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X1[['facade_type', 'shading_type']] = encoder_1.fit_transform(X1[['facade_type', 'shading_type']])

with open('encoder_layout.pkl', 'wb') as f:
    pickle.dump(encoder_1, f)

targets_1 = {'total_energy': 'total_energy', 'cooling_energy': 'cooling_energy', 'discomfort_hours': 'discomfort_hours'}

for name, col in targets_1.items():
    print(f"🤖 Training Layout Model for: {name}...")
    y1 = df1[col].copy()
    X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    # Copy and paste this right under model.fit(X_train, y_train)
    importances = dict(zip(feature_cols_1, model.feature_importances_))
    print(f"📊 Feature Importances for {name}: {importances}")

    
    print(f"   ↳ Real Test R² Score: {model.score(X_test, y_test)*100:.2f}%")
    
    with open(f'model_layout_{name}.pkl', 'wb') as f:
        pickle.dump(model, f)

# ==========================================
# 🛠️ PROCESS DATASET 2: MATERIAL PROPERTIES
# ==========================================
print("\n⚡ Loading Dataset 2 (Material & Thermal Envelope)...")

clean_cols_2 = [
    'iteration', 'generation', 'category',
    'total_energy', 'discomfort_hours', 'cooling_energy',
    'external_wall', 'flat_roof', 'glazing_type', 'partition_wall',
    'unnamed'
]

df2 = load_and_clean_csv('2nd Optimization Results.csv', clean_cols_2)
print("Verified columns in Dataset 2:", df2.columns.tolist())

feature_cols_2 = ['external_wall', 'flat_roof', 'glazing_type', 'partition_wall']
X2 = df2[feature_cols_2].copy()

encoder_2 = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X2[feature_cols_2] = encoder_2.fit_transform(X2[feature_cols_2])

with open('encoder_materials.pkl', 'wb') as f:
    pickle.dump(encoder_2, f)

for name, col in targets_1.items():
    print(f"🤖 Training Materials Model for: {name}...")
    y2 = df2[col].copy()
    X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    print(f"   ↳ Real Test R² Score: {model.score(X_test, y_test)*100:.2f}%")
    
    with open(f'model_mat_{name}.pkl', 'wb') as f:
        pickle.dump(model, f)

print("\n✅ Success! All models and encoders have been successfully compiled.")
