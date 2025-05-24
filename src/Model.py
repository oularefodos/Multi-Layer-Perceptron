import numpy as np

class MLP():
    def __init__(layers_size=[30, 24, 24, 2], epochs=84, learning_rate=0.01, batch_size=8):
        self.layers_size = layers_size;
        self.epochs = epochs;
        self.learning_rate = learning_rate;
        self.batch_size = batch_size;

        self.Weights = []
        self.bias = []
    
    def weight_initializer(n_input, n_output):
        return np.random.randn(n_output, n_input) * np.sqrt(2 / n_input)
    
    def bias_initializer(n_output):
        return np.zeros((n_output, 1))

    def initiaze_parms():
        for index in range(len(self.layers_size)):
            w = he_initializer(layer_sizes[i], layer_sizes[i+1])
            b = bias_initializer(layer_sizes[i+1])
            weights.append(w)
            biases.append(b)

