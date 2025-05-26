import numpy as np

class MLP():
    def __init__(self, layers_size=[30, 24, 24, 2], epochs=84, learning_rate=0.01, batch_size=8):
        self.layers_size = layers_size;
        self.epochs = epochs;
        self.learning_rate = learning_rate;
        self.batch_size = batch_size;

        self.weights = []
        self.bias = []
    
    def ReLU(self, x):
        return np.maximum(0, x)

    def solfmax(self, x):
        return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def bias_initializer(self, n_output):
        return np.zeros((n_output, 1))

    def initiaze_parms(self):
        for i in range(len(self.layers_size) - 1):
            w = self.weight_initializer(self.layers_size[i], self.layers_size[i+1])
            b = self.bias_initializer(self.layers_size[i+1])
            self.weights.append(w)
            self.bias.append(b)
    
    def farward(self, X, y):
        input_arr = X.T
        z = None
        for i in range(len(self.weights)):
            result = np.sum(self.weights[i].dot(input_arr)) + self.bias[i]
            z = self.sigmoid(result)
            input_arr = z
        print(z)

    def train(self, X, y):
        self.initiaze_parms();

        for _ in range(self.epochs):
            y = self.farward(X, y)
