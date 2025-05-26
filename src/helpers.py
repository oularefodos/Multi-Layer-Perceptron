import pandas as pd
import numpy as np

def process_dataset(pathname):
    try:
        df = pd.read_csv(pathname, skiprows=1, header=None)
        if df is None:
            exit(1)
        df = df.drop(df.columns[0], axis=1)  # Drop ID column
        X = df.iloc[:, 1:].values            # Features (30 columns)
        y = df.iloc[:, 0]                    # Diagnosis column (e.g., 'M' or 'B')
        y_hot = np.stack(y.map({'M': np.array([1, 0]), 'B': np.array([0, 1])}).values)
        return X, y_hot
    except Exception as e:
        print(f"Error reading dataset: {e}")
        exit(1)
