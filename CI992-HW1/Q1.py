# Q1_graded
# Do not change the above line.

# This cell is for your imports.

import numpy as np
import matplotlib.pyplot as plt

# Q1_graded
# Do not change the above line.

class Q1_Perceptron:
    def __init__(self, n, learning_rate=0.1, threshold=100):
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.weights = np.zeros(n + 1)
        self.n = n
           
    def get_error(self, inputs, label):
        return label - np.dot(inputs, self.weights)

    def train(self, training_inputs, labels):
        for iteration in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                
                error = self.get_error(inputs, label)
                self.weights += self.learning_rate * error * inputs

                self.learning_rate = self.learning_rate *0.999

# Q1_graded
# Do not change the above line.

# Generate data
training_inputs = [np.array([1, -1, -1]), 
                   np.array([1, -1, 1]), 
                   np.array([1, 1, -1]), 
                   np.array([1, 1, 1])]
labels = [1, -1, -1, -1]

# Perceptron
nor_perceptron = Q1_Perceptron(n=2, threshold=5000)
nor_perceptron.train(training_inputs, labels)

print("weights: ", nor_perceptron.weights)

