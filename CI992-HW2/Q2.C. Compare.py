#write your code here

# Generate test data
num_tests = 200
x_test = (np.random.rand(num_tests) * 8) - 4
x_test.reshape(num_tests, 1)
y_test = np.sin(x_test)

# Predict
predicted_y_rbf = rbf_model.predict(x_test)
predicted_y_mlp = model.predict(x_test)

# Sort based on x
idx = np.argsort(x_test)
x_test = np.array(x_test)[idx]
y_test = np.array(y_test)[idx]
predicted_y_rbf = np.array(predicted_y_rbf)[idx]
predicted_y_mlp = np.array(predicted_y_mlp)[idx]

# Plot 
plt.plot(x_test, predicted_y_rbf, label='rbf')
plt.plot(x_test, predicted_y_mlp, label='mlp')
plt.plot(x_test, y_test, label='sin(x)')
plt.legend()
plt.show()

