import numpy as np

class Layer():
    def __init__(self, n_x, n_y, eta=0.01):
        self.Nx = n_x  # The number of input neurons
        self.Ny = n_y  # The number of output neurons
        self.eta = eta
        self.W = np.random.randn(self.Ny, self.Nx)
        self.B = np.random.randn(self.Ny, 1)
        # self.W = self.random.normal(loc=0.0, scale=1.0, size=(self.Ny, self.Nx))
        # self.B = self.random.normal(loc=0.0, scale=1.0, size=(self.Ny,1))

    def _forward(self, X_in):
        self.X_in = X_in
        output = (np.dot(self.W, X_in))+ self.B
        return output

    def _backward(self, grad_EY):
        grad_EX = np.dot(self.W.T, grad_EY)
        delta_W = np.dot(grad_EY, (self.X_in).T)
        self.W -= self.eta * delta_W
        delta_B = grad_EY
        self.B -= self.eta * delta_B
        return grad_EX


class Activation():
    def __init__(self, a_function='relu'):
        self.a_function = a_function

    def _forward(self, X_in):
        self.X_in = X_in
        if (self.a_function == 'relu'):
            Y_out = X_in * (X_in > 0)
            return Y_out

        elif (self.a_function == 'sigmoid'):
            Y_out = 1/(1 + np.exp(-1*(X_in)))
            return Y_out

        elif (self.a_function == 'tanh'):
            Y_out = np.tanh(X_in)
            return Y_out

    def _backward(self, Y_in):
        if (self.a_function == 'relu'):
            f_prime = (self.X_in > 0)
            return np.multiply(Y_in,f_prime)

        elif (self.a_function == 'sigmoid'):
            sigma = 1/(1 + np.exp(-1*self.X_in))
            f_prime = sigma * (1-sigma)
            return np.multiply(Y_in,f_prime)

        elif (self.a_function == 'tanh'):
            f_prime = (1 - (np.tanh(self.X_in)**2))
            return np.multiply(Y_in,f_prime)
