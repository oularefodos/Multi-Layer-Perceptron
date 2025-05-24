import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Multilayer Perceptron")

    parser.add_argument('--layer', type=int, nargs='+', required=True,
                        help="List of hidden layer sizes. Example: --layer 24 24")

    # Optional: Add other parameters like epochs, lr, etc.
    parser.add_argument('--epochs', type=int, default=50)

    parser.add_argument('--learning_rate', type=float, default=0.01)

    parser.add_argument('--batch_size', type=float, default=8)

    return parser.parse_args()

# Example use
if __name__ == "__main__":
    args = parse_args()
    print("Layer sizes:", args.layer)
    print("learning_rate:", args.learning_rate)
    print("epoch:", args.epochs)