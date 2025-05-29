import argparse
import numpy as np
from Model import MLP
from helpers import process_dataset
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Multilayer Perceptron")

    parser.add_argument('--layers', type=int, nargs='+', required=True,
                        help="List of hidden layer sizes. Example: --layer 24 24")

    # Optional: Add other parameters like epochs, lr, etc.
    parser.add_argument('--epochs', type=int, default=50)

    parser.add_argument('--learning_rate', type=float, default=0.01)

    parser.add_argument('--batch_size', type=float, default=8)

    return parser.parse_args()

# Example use
if __name__ == "__main__":
    args = parse_args()

    X, y = process_dataset('data/train.csv')
    
    model = MLP();
    model.train(X, y);

    loss_history = model.loss_history
    acurracy_history = model.accuracy_history
    
    x_loss = list(range(len(loss_history)))
    x_accur = list(range(len(acurracy_history)))
    
    plt.figure()
    plt.plot(x_loss, loss_history)
    plt.legend()
    
    plt.figure()
    plt.plot(x_accur, acurracy_history)
    plt.legend()
    
    plt.show()
    