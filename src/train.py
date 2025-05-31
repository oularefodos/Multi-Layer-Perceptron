import argparse
import numpy as np
from model import MLP
from helpers import process_dataset, plot_training_history
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Multilayer Perceptron")

    parser.add_argument('--layers', type=int, nargs='+', required=True,
                        help="List of hidden layer sizes. Example: --layer 24 24")
    
    parser.add_argument('--activations', type=str, nargs='+', required=True,
                        help="List of hidden layer sizes. Example: --activations sigmoid sigmoid")

    # Optional: Add other parameters like epochs, lr, etc.
    parser.add_argument('--epochs', type=int, default=50)

    parser.add_argument('--learning_rate', type=float, default=0.01)

    parser.add_argument('--batch_size', type=int, default=8)

    return parser.parse_args()

def get_layers_size(features_size, layers, output_size):
    layers_size = [features_size]
    for layer in layers:
        layers_size.append(layer)
        print(layer);
    layers_size.append(output_size)
    return layers_size;

# Example use
if __name__ == "__main__":
    args = parse_args()
    
    layers_size = get_layers_size(30, args.layers, 2);
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    activations = args.activations
    
    if len(activations) != len(args.layers):
        print("Error: The number of activation functions must match the number of layers specified.")
        exit(1)

    print(layers_size)
    X, y = process_dataset('data/train.csv')
    X_valid, y_valid = process_dataset('data/valid.csv')
    
    model = MLP(layers_size=layers_size, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size);
    model.train(X, y, X_valid, y_valid);
    plot_training_history(model);
    
    