import argparse
import numpy as np
from model import MLP
from helpers import process_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Predict a Multilayer Perceptron")

    parser.add_argument('--data', type=str, required=True,
                        help="the path of your dataset")
    
    parser.add_argument('--model', type=str, required=True,
                        help="the path of your model")

    return parser.parse_args()


# Example use
if __name__ == "__main__":
    args = parse_args()
    
    model = args.model
    data = args.data
    
    X, y = process_dataset(data)
    
    breast_cancer_model = MLP.load_model(model);
    
    prediction = breast_cancer_model.predict(X);
    
    print(prediction);
    
    