import pandas as pd
import argparse
import random

# 1) Define the full list of column names:
columns = [
    "id",
    "diagnosis",
    # -- the “mean” features --
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
    # -- the “standard error” features --
    "radius error", "texture error", "perimeter error", "area error", "smoothness error",
    "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error",
    # -- the “worst” features --
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", help="dataset_path needs be a string")
    args = parser.parse_args()
    
    try:
        df = pd.read_csv(args.dataset_path, header=None, names=columns)
        if df is None:
            exit(1)
    except Exception as e:
        print(f"Error reading dataset: {e}")
        exit(1)
    
    # divide dataset based on label
    df_M = df[df['diagnosis'] == 'M']
    df_B = df[df['diagnosis'] == 'B']
    
    shuffled_df_M = df_M.sample(frac=1).reset_index(drop=True)
    shuffled_df_B = df_B.sample(frac=1).reset_index(drop=True)
    
    split_ratio = 0.8  # We are using an 80-20 split here
    split_index_M = int(split_ratio * len(df_M))
    split_index_B = int(split_ratio * len(df_B))
    
    train_sample_M = shuffled_df_M[:split_index_M]
    valid_sample_M = shuffled_df_M[split_index_M:]
    
    train_sample_B = shuffled_df_B[:split_index_B]
    valid_sample_B = shuffled_df_B[split_index_B:]
    
    train_df = pd.concat([train_sample_M, train_sample_B], axis=0).reset_index(drop=True)
    valid_df = pd.concat([valid_sample_M, valid_sample_B], axis=0).reset_index(drop=True)
    
    train_df.to_csv('data/train.csv', sep=',', index=False, encoding='utf-8');
    valid_df.to_csv('data/valid.csv', sep=',', index=False, encoding='utf-8');
    
    print("Train class distribution:")
    print(train_df['diagnosis'].value_counts())
    print("\nvalid class distribution:")
    print(valid_df['diagnosis'].value_counts())
    