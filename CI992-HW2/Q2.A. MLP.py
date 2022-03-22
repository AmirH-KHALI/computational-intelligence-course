#write your code here

import keras
from keras.layers import *
from keras.optimizers import *
from keras.models import Sequential

import numpy as np
import matplotlib.pyplot as plt

# Generate training data
num_samples = 300
x_train = (np.random.rand(num_samples) * 6) - 3
x_train.reshape(num_samples, 1)
y_train = np.sin(x_train)

# Create the model
model = Sequential()
model.add(Dense(350, input_shape=(1,)))
model.add(Activation('sigmoid'))
model.add(Dense(50))
model.add(Activation('sigmoid'))
model.add(Dense(1))

# Train
model.compile(loss='mean_squared_error', optimizer='adam')
train_history = model.fit(x_train, y_train, epochs=100, batch_size=4)

# Generate test data
num_tests = 200
x_test = (np.random.rand(num_tests) * 6) - 3
x_test.reshape(num_tests, 1)
y_test = np.sin(x_test)

# Predict
predicted_y = model.predict(x_test)

# Sort based on x
idx = np.argsort(x_test)
x_test = np.array(x_test)[idx]
y_test = np.array(y_test)[idx]
predicted_y = np.array(predicted_y)[idx]

# Plot 
plt.plot(x_test, predicted_y, label='Predicted')
plt.plot(x_test, y_test, label='sin(x)')
plt.legend()
plt.show()

