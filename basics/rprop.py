import numpy as np
import pdb

def generate_data(n_features, n_values):
    random_generator = np.random.default_rng()
    features = random_generator.random((n_features, n_values))
    targets = random_generator.random((n_features))
    return features, targets

class RNN:
    def __init__(self):
        self.global_weight = [1, 1] # [Input, Recurrent Weight]
        self.local_weight = [0.001, 0.001]
        self.W_sign = [0, 0]

        self.eta_p = 1.2
        self.eta_n = 0.5

    def state_handler(self, input_x, previous_state):
        return input_x * self.global_weight[0] + previous_state * self.global_weight[1]

    def forward_propagation(self, X):
      S = np.zeros((X.shape[0], X.shape[1]+1))  
      for k in range(0, X.shape[1]):
          next_state = self.state_handler(X[:,k], S[:,k]) 
          S[:,k+1] = next_state 
      return S

    def backward_propagation(self, X, S, grad_out):
        grad_over_time = np.zeros((X.shape[0], X.shape[1]+1))
        grad_over_time[:,-1] = grad_out 
        wx_grad = 0
        wy_grad = 0
        for k in range(X.shape[1], 0, -1):
            wx_grad += np.sum(grad_over_time[:, k] * X[:, k-1])
            wy_grad += np.sum(grad_over_time[:, k] * S[:, k-1])
            grad_over_time[:, k-1] = grad_over_time[:, k] * self.global_weight[1]
        return (wx_grad, wy_grad), grad_over_time

    def update_rprop(self, X, Y, W_prev_sign, local_weight):
        S = self.forward_propagation(X)
        grad_out = 2 * (S[:, -1] - Y) / 500 
        W_grads, _ = self.backward_propagation(X, S, grad_out)
        self.W_sign = np.sign(W_grads)

        for i, _ in enumerate(self.global_weight):
            if self.W_sign[i] == W_prev_sign[i]: 
                local_weight[i] *= self.eta_p
            else:                               
                local_weight[i] *= self.eta_n
        self.local_weight = local_weight

    def train(self, X, Y, training_epochs):
        for epochs in range(training_epochs):
            self.update_rprop(X, Y, self.W_sign, self.local_weight)

            for i, _ in enumerate(self.global_weight):
                self.global_weight[i] -= self.W_sign[i] * self.local_weight[i]

if __name__ == "__main__":
    trainX, trainY = generate_data(500, 4)
    testX, testY = generate_data(5, 4)
    
    rnn = RNN()
    rnn.train(trainX, trainY, 200)
    
    print(f"Targets are: {testY}")
    y = rnn.forward_propagation(testX)[:, -1]
    print(f"Predicted are: {y}")






