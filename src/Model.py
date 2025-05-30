import numpy as np

class MLP():
    def __init__(self, layers_size=[30, 24, 24, 2], epochs=800, learning_rate=0.01, batch_size=4):
        self.layers_size = layers_size;
        self.epochs = epochs;
        self.learning_rate = learning_rate;
        self.batch_size = batch_size;
        self.loss_history_train = []
        self.accuracy_history_train = []
        self.loss_history_valid = []
        self.accuracy_history_valid = []
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
                A = self.sigmoid(Z)
            else:
                A = self.softmax(Z)
            activations.append(A)
        return activations, pre_activations
    
    def predict(self, X):
        X = self.normalize(X);
        activations, _ = self.forward(X);
        return np.argmax(activations[-1], axis=1);

    def validate(self, X, Y):
        n_sample = X.shape[0]
        correct = 0
        total = 0
        epoch_loss = 0;
        for start in range(0, n_sample, self.batch_size):
                
            end = start + self.batch_size
            X_batched = X[start:end]
            Y_batched = Y[start:end]
            
            activations, _ = self.forward(X_batched)

            y_pred = np.argmax(activations[-1], axis=1)
            y_true = np.argmax(Y_batched, axis=1)
            correct += np.sum(y_pred == y_true); 
            total += len(Y_batched)
    
            epoch_loss += self.compute_cross_entropy_loss(Y_batched, activations[-1])
        average_loss = epoch_loss / n_sample
        self.loss_history_valid.append(average_loss)

        accuracy = correct / total;
        self.accuracy_history_valid.append(accuracy);
        

    def train(self, X, Y, X_valid, Y_valid):
        self.initiaze_parms();
        X = self.normalize(X);
        X_valid = self.normalize(X_valid)
        n_sample = X.shape[0]
        for epoch in range(self.epochs):
            correct = 0
            total = 0
            epoch_loss = 0
            for start in range(0, n_sample, self.batch_size):
                
                end = start + self.batch_size
                X_batched = X[start:end]
                Y_batched = Y[start:end]

                activations, pre_activations = self.forward(X_batched)
                
                self.back_propagation(activations, pre_activations, Y_batched);
                
                y_pred = np.argmax(activations[-1], axis=1)
                y_true = np.argmax(Y_batched, axis=1)
                correct += np.sum(y_pred == y_true);
                
                total += len(Y_batched)
                epoch_loss += self.compute_cross_entropy_loss(Y_batched, activations[-1])
            average_loss = epoch_loss / n_sample
            self.loss_history_train.append(average_loss)
            
            accuracy = correct / total;
            self.accuracy_history_train.append(accuracy);
            
            self.validate(X_valid, Y_valid);

            if (epoch % 100  == 0):
                print(f"Epoch: {epoch}, loss: {average_loss}, accuracy: {accuracy}")