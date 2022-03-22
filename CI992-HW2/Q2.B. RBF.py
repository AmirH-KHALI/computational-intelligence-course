import numpy as np
import matplotlib.pyplot as plt

class RBF:
    def __init__(self):
        self.hidden_shape = 8
        self.centers = None
        self.weights = None

    def calculate_interpolation_matrix(self, x):
        i, j = np.indices((len(x), self.hidden_shape))
        return np.exp(-(self.centers[j] - x[i])**2)

    def fit(self, x, y):
        random_args = np.random.choice(len(x), self.hidden_shape)
        self.centers = x[random_args]
        matrix = self.calculate_interpolation_matrix(x)
        self.weights = np.dot(np.linalg.pinv(matrix), y)

    def predict(self, x):
        matrix = self.calculate_interpolation_matrix(x)
        predictions = np.dot(matrix, self.weights)
        return predictions

# Generate training data
num_samples = 200
x_train = (np.random.rand(num_samples) * 6) - 3
x_train.reshape(num_samples, 1)
y_train = np.sin(x_train)

# Train
rbf_model = RBF()
rbf_model.fit(x_train, y_train)

# Generate test data
num_tests = 200
x_test = (np.random.rand(num_tests) * 6) - 3
x_test.reshape(num_tests, 1)
y_test = np.sin(x_test)

# Predict
predicted_y = rbf_model.predict(x_test)

# Sort based on x
idx = np.argsort(x_test)
x_test = np.array(x_test)[idx]
y_test = np.array(y_test)[idx]
predicted_y = np.array(predicted_y)[idx]

# Plot 
plt.plot(x_test, predicted_y, label='rbf')
plt.plot(x_test, y_test, label='sin(x)')
plt.legend()
plt.show()

