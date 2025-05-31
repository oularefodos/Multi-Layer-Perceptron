import numpy as np
import pickle
import os

def softmax(Z):
    exp_x = np.exp(Z - np.max(Z))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def ReLU(Z):
    return np.maximum(0, Z)

def apply_activation(name, Z):
    if name == "softmax":
        return softmax(Z)
    elif name == "sigmoid":
        return sigmoid(Z)
    elif name == "relu":
        return ReLU(Z)
    else:
        raise ValueError(f"There is no such activation function: '{name}'")

class MLP():
    def __init__(self, layers_size=[30, 24, 24, 2], epochs=800, activations=["sigmoid", "sigmoid"], learning_rate=0.01, batch_size=4):
        self.layers_size = layers_size;
        self.epochs = epochs;
        self.learning_rate = learning_rate;
        self.activations = activations;
        self.batch_size = batch_size;
        self.loss_history_train = []
        self.accuracy_history_train = []
        self.loss_history_valid = []
        self.accuracy_history_valid = []
        self.weights = []
        self.bias = []

    def xavier_init(self, n_in, n_out):
        limit = np.sqrt(6 / (n_in + n_out))
        return np.random.uniform(-limit, limit, (n_in, n_out))
    
    def bias_initializer(self, n_output):
        return np.zeros((1, n_output))
    
    def back_propagation(self, activations, pre_activations, Y):
        grads_w = [None] * len(self.weights)
        grads_b = [None] * len(self.bias)
        m = Y.shape[0]
        # output layer gradient
        A_final = activations[-1]
        delta = (A_final - Y) 
        # gradients for last layer
        grads_w[-1] = np.dot(activations[-2].T, delta) / m
        grads_b[-1] = np.sum(delta, axis=0, keepdims=True) / m

        # backprop through hidden layers
        for l in range(len(self.layers_size)-2, 0, -1):
            Z = pre_activations[l-1]
            dA_prev = np.dot(delta, self.weights[l].T)
            dZ = dA_prev * (activations[l] * (1 - activations[l]))  # sigmoid derivative
            grads_w[l-1] = np.dot(activations[l-1].T, dZ) / m
            grads_b[l-1] = np.sum(dZ, axis=0, keepdims=True) / m
            delta = dZ

        # update parameters
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads_w[i]
            self.bias[i] -= self.learning_rate * grads_b[i]

    def initiaze_parms(self):
        for i in range(len(self.layers_size) - 1):
            w = self.xavier_init(self.layers_size[i], self.layers_size[i+1])
            b = self.bias_initializer(self.layers_size[i+1])
            self.weights.append(w)
            self.bias.append(b)
    
    def compute_cross_entropy_loss(self, y_true, y_pred):
        epsilon = 1e-8
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.sum(y_true * np.log(y_pred), axis=1)
        return np.mean(loss)

    def normalize(self, X: np.ndarray):
        self.mean_, self.std_ = np.mean(X, axis=0), np.std(X, axis=0)
        return (X - self.mean_) / self.std_
    
    def forward(self, X):
        A = X
        activations = [A]
        pre_activations = []
        for i in range(len(self.weights)):
            Z = np.dot(A, self.weights[i]) + self.bias[i]
            pre_activations.append(Z)
            if i < len(self.weights) - 1:
                A = apply_activation(self.activations[i], Z)
            else:
                A = apply_activation("softmax", Z)
            activations.append(A)
        return activations, pre_activations

    def run_epoch(self, n_sample, X, Y, isTraining):
        correct = 0
        total = 0
        epoch_loss = 0
        for start in range(0, n_sample, self.batch_size):
            
            end = start + self.batch_size
            X_batched = X[start:end]
            Y_batched = Y[start:end]
            activations, pre_activations = self.forward(X_batched)
            
            if isTraining == True:
                self.back_propagation(activations, pre_activations, Y_batched);
            
            y_pred = np.argmax(activations[-1], axis=1)
            y_true = np.argmax(Y_batched, axis=1)
            correct += np.sum(y_pred == y_true);
            
            total += len(Y_batched)
            epoch_loss += self.compute_cross_entropy_loss(Y_batched, activations[-1])
        average_loss = epoch_loss / n_sample;
        accuracy = correct / total;
        
        return average_loss, accuracy;
        
    def predict(self, X):
        X = self.normalize(X);
        activations, _ = self.forward(X);
        return np.argmax(activations[-1], axis=1);        

    def train(self, X, Y, X_valid, Y_valid):
        self.initiaze_parms();
        X = self.normalize(X);
        X_valid = self.normalize(X_valid)
        n_sample_train = X.shape[0]
        n_sample_valid = X_valid.shape[0]
        for epoch in range(self.epochs):
            train_average_loss, train_accuracy = self.run_epoch(n_sample_train, X, Y, True)
            self.accuracy_history_train.append(train_accuracy)
            self.loss_history_train.append(train_average_loss)

            valid_average_loss, valid_accuracy = self.run_epoch(n_sample_valid, X_valid, Y_valid, False)
            self.accuracy_history_valid.append(valid_accuracy)
            self.loss_history_valid.append(valid_average_loss)

            if epoch % 10 == 0:
                print(
                    f"Epoch: {epoch}, "
                    f"Train Loss: {train_average_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                    f"Validation Loss: {valid_average_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}"
                )
    def save(self, filepath):
        # Ensure parent directory exists:
        directory = os.path.dirname(filepath)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved to: {filepath}")
    
    @staticmethod
    def load_model(filepath):
        with open(filepath, 'rb') as f:
            loaded_model = pickle.load(f)
        print(f"Model loaded from: {filepath}")
        return loaded_model

