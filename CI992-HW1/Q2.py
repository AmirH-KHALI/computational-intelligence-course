# Q2_graded
# Do not change the above line.

# This cell is for your imports.

import numpy as np
import matplotlib.pyplot as plt

# Q2_graded
# Do not change the above line.

class Q2_Perceptron:
    def __init__(self, n, learning_rate=0.1, threshold=100):
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.weights = np.zeros(n + 1)
        self.n = n
        self.cost = []
           
    def get_error(self, label, y):
        return label - y

    def train(self, training_inputs, labels):
        for iteration in range(self.threshold):
            loss = []
            for inputs, label in zip(training_inputs, labels):
                
                y = np.dot(inputs, self.weights)
                error = self.get_error(label, y)
                loss.append((1 / 2) * (error**2))
                self.weights += self.learning_rate * error * inputs
                

                self.learning_rate = self.learning_rate *0.999

            self.cost.append(np.mean(loss, axis=0, keepdims=True))


# Q2_graded
# Do not change the above line.

# Read
X = []
Y = []
labels = []
data_file = open('/content/data.txt', 'r')
for line in data_file.readlines():
    splited = line.split(',')
    X.append(float(splited[0]))
    Y.append(float(splited[1]))
    if float(splited[2]) == 0: labels.append(-1)
    else: labels.append(1)

# Normalize
X = X - np.mean(X, axis=0, keepdims=True)
X = X / np.std(X, axis=0, keepdims=True)
Y = Y - np.mean(Y, axis=0, keepdims=True)
Y = Y / np.std(Y, axis=0, keepdims=True)

# Generate training inputs
training_inputs = []
for x, y in zip(X, Y):
    training_inputs.append(np.array([1, x, y]))

# Train
perceptron = Q2_Perceptron(n=2)
perceptron.train(training_inputs, labels)
print('Weights: ', perceptron.weights, '\n')

# Plot perceptron
for x, y, label in zip(X, Y, labels):
    if label == 1: plt.scatter(x, y, marker='.', color = '#ff0000')
    else: plt.scatter(x, y, marker='*', color = '#0000ff')

x = np.linspace(-2, 2, 400)
weights = perceptron.weights
y = -1 * ((weights[1] / weights[2]) * x + (weights[0] / weights[2]))
plt.plot(x, y, color = '#00ff00')
plt.show()

# Plot Mistakes
print()
plt.plot(perceptron.cost)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.show()

