# Q4_graded
# Do not change the above line.

# This cell is for your imports.

from keras.layers import *
from keras.optimizers import *
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# Q4_graded
# Do not change the above line.

# Load
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Normalize
X_train = X_train.reshape(X_train.shape[0], 784)
X_train = X_train.astype('float32')
X_train /= 255
Y_train = to_categorical(Y_train, 10)

# Create the model
model = Sequential()
model.add(Dense(350, input_shape=(784,), activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Train
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
train_history = model.fit(X_train, Y_train, epochs=30, batch_size=250)

# Plot loss
loss = train_history.history['loss']
plt.plot(loss)
plt.xlabel('Iteration')
plt.ylabel('loss')
plt.show()

# Plot accuracy
acc = train_history.history['acc']
plt.plot(acc)
plt.xlabel('Iteration')
plt.ylabel('accuracy')
plt.show()

# Test the model after training
X_test = X_test.reshape(X_test.shape[0], 784)
X_test = X_test.astype('float32')
X_test /= 255
Y_test = to_categorical(Y_test, 10)

test_results = model.evaluate(X_test, Y_test)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')

# Q4_graded
# Do not change the above line.

# This cell is for your codes.

