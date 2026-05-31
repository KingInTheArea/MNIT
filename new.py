# Fixed new.py script
import pandas as pd

# Adding encoding='latin1' allows pandas to handle special characters like °
raw_df = pd.read_csv('Pareto.csv', header=1, encoding='latin1')
print("Raw CSV Row 2 Headers:")
print(raw_df.columns.tolist()[:11])
