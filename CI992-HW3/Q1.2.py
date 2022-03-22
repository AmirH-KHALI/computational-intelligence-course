# Q1.2_graded
# Do not change the above line.

import numpy as np
import matplotlib.pyplot as plt

# Q1.2_graded
# Do not change the above line.

class HopfieldNet:

    def __init__(self, n):
        self.n = n
        self.weights = np.zeros((n, n))

    def train(self, input):
        input = np.array([input])
        
        p = np.dot(input.T, input)

        np.fill_diagonal(p, 0)
        self.weights += p 

    def is_stable(self, input):
        input = np.array(input)
        dp = np.dot(self.weights, input)

        signs = np.sign(dp)

        return ((signs == input).all(), input, signs)


# Q1.2_graded
# Do not change the above line.

def print_result(result):
    if (result[0]):
        print(result[1], 'is stable.')
    else:
        print(result[1], 'is unstable. nearest data: ', result[2])

hn = HopfieldNet(6)

hn.train([ 1, 1, 1,-1,-1,-1])
hn.train([ 1,-1, 1,-1, 1,-1])

print_result(hn.is_stable([ 1, 1, 1,-1,-1,-1]))
print_result(hn.is_stable([-1, 1, 1,-1,-1,-1]))

