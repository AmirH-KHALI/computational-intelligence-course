import numpy as np
import matplotlib.pyplot as plt

class Q1_B_C_Kohonen:
    
    def __init__(self, n, learning_rate=0.1, epochs=100, sigma=4, d_sigma=1, d_learning_rate=1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.rand(n**2, 3)
        self.n = n
        self.sigma = sigma
        self.d_sigma = d_sigma
        self.d_learning_rate = d_learning_rate

    def get_dists_2(self, x, y):
        i, j = np.indices((self.n, self.n))
        return (i - x) ** 2 + (j - y) ** 2

    def train(self, training_inputs):
        for _ in range(self.epochs):
            for training_input in training_inputs:
                winner_w = np.argmin(np.sum(np.square(np.subtract(self.weights, training_input)), axis=1))
                
                ww_x = winner_w // self.n
                ww_y = winner_w %  self.n

                dist_2 = self.get_dists_2(ww_x, ww_y).reshape(-1,1)

                self.weights += np.multiply(training_input - self.weights, np.exp((-dist_2) / (2 * (self.sigma ** 2)))) * self.learning_rate
            self.learning_rate *= self.d_learning_rate
            self.sigma *= self.d_sigma

# 1_B
kohonen = Q1_B_C_Kohonen(40, learning_rate=1, sigma=6, d_learning_rate=.9)
training_inputs = np.random.rand(1600, 3)

kohonen.train(training_inputs)

plt.imshow(kohonen.weights.reshape(40, 40, 3))
plt.show()
    

# 1_C
kohonen = Q1_B_C_Kohonen(40, learning_rate=1, sigma=10, d_learning_rate=.9, d_sigma=.99)
training_inputs = np.random.rand(1600, 3)

kohonen.train(training_inputs)

plt.imshow(kohonen.weights.reshape(40,40,3))
plt.show()


