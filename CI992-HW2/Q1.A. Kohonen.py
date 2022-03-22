import numpy as np
import matplotlib.pyplot as plt

class Q1_A_Kohonen:
    
    def __init__(self, n, learning_rate=0.1, epochs=100, sigma=4):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.rand(n**2, 3)
        self.n = n
        self.sigma = sigma

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
            

kohonen = Q1_A_Kohonen(40)
training_inputs = np.random.rand(1600, 3)

kohonen.train(training_inputs)

plt.imshow(kohonen.weights.reshape(40, 40, 3))
plt.show()
    


