import pandas as pd
import numpy as np
import numpy
import matplotlib.pyplot as plt


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


def plot_training_history(model):
    # Extract training and validation histories from the model
    loss_history_train = model.loss_history_train
    accuracy_history_train = model.accuracy_history_train

    loss_history_valid = model.loss_history_valid
    accuracy_history_valid = model.accuracy_history_valid

    # Generate x-axis values (epochs)
    x_train_loss = list(range(len(loss_history_train)))
    x_train_accur = list(range(len(accuracy_history_train)))

    x_valid_loss = list(range(len(loss_history_valid)))
    x_valid_accur = list(range(len(accuracy_history_valid)))

    # Plot Loss
    plt.figure()
    plt.plot(x_train_loss, loss_history_train, label='Train Loss')
    plt.plot(x_valid_loss, loss_history_valid, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot Accuracy
    plt.figure()
    plt.plot(x_train_accur, accuracy_history_train, label='Train Accuracy')
    plt.plot(x_valid_accur, accuracy_history_valid, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.show()

