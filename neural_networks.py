import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


learnrate = 0.5
x = np.array([1, 2])
y = np.array(0.5)

# Initial weights
w = np.array([0.5, -0.5])

activation_function = np.dot(w, x)
# Calculate one gradient descent step for each weight
# TODO: Calculate output of neural network
nn_output = sigmoid(activation_function)

# TODO: Calculate error of neural network
error = y - nn_output

error_term = error * (nn_output * (1 - nn_output))

# TODO: Calculate change in weights
del_w = learnrate * error_term * x

print('Neural Network predicted output:')
print(nn_output)
print('Amount of Error (actual_output - predicted_output):')
print(error)
print('Change in Weights:')
print(del_w)
