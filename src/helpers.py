import pandas as pd

def load_dataset(pathname):
    try:
        df = pd.read_csv(pathname, skiprows=1)
        if df is None:
            exit(1)
        df = df.drop(df.columns[0], axis=1)
        X, y = df.iloc[:, 1:], df.iloc[:, 0]
        return X, y
    except Exception as e:
        print(f"Error reading dataset: {e}")
        exit(1)
