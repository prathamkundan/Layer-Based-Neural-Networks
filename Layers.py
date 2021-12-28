import numpy as np
from numpy.random.mtrand import seed

class Layer():
    '''
    NN Layer:
    -----------------
        A Layer of the Neural Network

        Args:
        -----------------------
        n_x: int 
            Number of input neurons.
        n_y: int 
            Number of output neurons.
        eta: float (defaault 0.01)
            Learning rate
        seed: int  (default 69)
            Seed for generating random weights/ biases

        Parameters:
        -------------------------
        W: np.array (shape = (ny, nx))
            Weights initialised using the given seed value
        B: np.array (shape = (1, ny))
            Biases initialised using the given seed value
        N_x: int 
            Number of input neurons.
        N_y: int 
            Number of output neurons.
    '''
    def __init__(self, n_x, n_y, eta=0.01, seed = 69):
        self.Nx = n_x  # The number of input neurons
        self.Ny = n_y  # The number of output neurons
        self.eta = eta  # Learning Rate 
        self.random = np.random.RandomState(seed)
        # Initialising weights and biases 
        self.W = self.random.normal(loc=0.0, scale=1.0, size=(self.Ny, self.Nx))
        self.B = self.random.normal(loc=0.0, scale=1.0, size=(1, self.Ny))

    def _forward(self, X_in):
        # Forwarding the input
        self.X_in = X_in
        output = np.dot(X_in, self.W.T)
        output += self.B
        return output

    def _backward(self, grad_EY):
        # Back-propagating 
        grad_EX = np.dot(grad_EY, self.W)
        delta_W = np.dot(grad_EY.T, self.X_in)
        self.W -= self.eta * delta_W
        delta_B = grad_EY.sum(axis = 0)
        self.B -= self.eta * delta_B
        return grad_EX


class Activation():
    '''
    Activation Layer:
    ------------------
        Args:
        ------------------
        function: str (relu, sigmoid, tanh)
            The activation function
    '''
    def __init__(self, function='relu'):
        self.function = function

    def _forward(self, X_in):
        self.X_in = X_in
        if (self.function == 'relu'):
            Y_out = X_in * (X_in > 0)
            return Y_out

        elif (self.function == 'sigmoid'):
            Y_out = 1/(1 + np.exp(-1*(X_in)))
            return Y_out

        elif (self.function == 'tanh'):
            Y_out = np.tanh(X_in)
            return Y_out

    def _backward(self, Y_in):
        if (self.function == 'relu'):
            f_prime = (self.X_in > 0)*1
            return np.multiply(Y_in,f_prime)

        elif (self.function == 'sigmoid'):
            sigma = 1/(1 + np.exp(-1*self.X_in))
            f_prime = sigma * (1-sigma)
            return np.multiply(Y_in,f_prime)

        elif (self.function == 'tanh'):
            f_prime = (1 - (np.tanh(self.X_in)**2))
            return np.multiply(Y_in,f_prime)
