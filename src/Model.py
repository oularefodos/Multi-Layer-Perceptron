import numpy as np

class MLP():
    def __init__(self, layers_size=[30, 24, 24, 2], epochs=1, learning_rate=0.01, batch_size=4):
        self.layers_size = layers_size;
        self.epochs = epochs;
        self.learning_rate = learning_rate;
        self.batch_size = batch_size;

        self.weights = []
        self.bias = []
    
    def ReLU(self, Z):
        return np.maximum(0, Z)

    def softmax(self, Z):
        exp_x = np.exp(Z - np.max(Z))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def xavier_init(self, n_in, n_out):
        limit = np.sqrt(6 / (n_in + n_out))
        return np.random.uniform(-limit, limit, (n_in, n_out))
    
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def bias_initializer(self, n_output):
        return np.zeros((1, n_output))

    def initiaze_parms(self):
        for i in range(len(self.layers_size) - 1):
            w = self.xavier_init(self.layers_size[i], self.layers_size[i+1])
            b = self.bias_initializer(self.layers_size[i+1])
            self.weights.append(w)
            self.bias.append(b)
    
    def categorical_cross_entropy(y_true, y_pred):
        epsilon = 1e-8
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.sum(y_true * np.log(y_pred), axis=1)
        return np.mean(loss)
    
    def forward(self, X):
        A = X
        activations = [A]
        pre_activations = []
        for i in range(len(self.weights)):
            Z = np.dot(A, self.weights[i]) + self.bias[i]
            pre_activations.append(Z)
            if i < len(self.weights) - 1:
                A = self.sigmoid(Z)
            else:
                A = self.softmax(Z)
            activations.append(A)
        return activations, pre_activations

    def train(self, X, Y):
        self.initiaze_parms();
        n_sample = X.shape[0]
        for _ in range(self.epochs):
            for start in range(0, n_sample, self.batch_size):
                
                end = start + self.batch_size
                X_batched = X[start:end]
                Y_batched = Y[start:end]

                activations, pre_activations = self.forward(X_batched)
                print(activations[-1])
                break;
