from typing import no_type_check_decorator
import numpy as np
from numpy.core.fromnumeric import shape
from scipy import signal
from numpy import random


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

    def __init__(self, n_x, n_y, eta=0.01, seed=69):
        self.Nx = n_x  # The number of input neurons
        self.Ny = n_y  # The number of output neurons
        self.eta = eta  # Learning Rate
        self.random = np.random.RandomState(seed)
        # Initialising weights and biases
        self.W = self.random.normal(
            loc=0.0, scale=1.0, size=(self.Ny, self.Nx))
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
        delta_B = grad_EY.sum(axis=0)
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
            return np.multiply(Y_in, f_prime)

        elif (self.function == 'sigmoid'):
            sigma = 1/(1 + np.exp(-1*self.X_in))
            f_prime = sigma * (1-sigma)
            return np.multiply(Y_in, f_prime)

        elif (self.function == 'tanh'):
            f_prime = (1 - (np.tanh(self.X_in)**2))
            return np.multiply(Y_in, f_prime)


class Convolutional():
    '''
    Convolution Layer:
    ------------------
        Args:
        ------------------
        input_shape: tuple
            Shape of input
        kernel_shape: tuple
            Shape of kernel
        n_kernel: int
            Depth of kernel/ number of kernels
        seed: int
            Seed to initialise Kernels and Biases
        learning_rate: float
            learning rate of layer 
    '''
    def __init__(self, input_shape, kernel_shape, n_kernels, seed=69, learning_rate = 0.01):
        self.input_depth, input_height, input_width = input_shape
        self.input_dim = input_shape[1:]
        self.kernel_shape = kernel_shape
        self.n_kernel = n_kernels
        self.learning_rate = learning_rate
        self.random = np.random.RandomState(seed)
        self.output_shape =  (n_kernels,) + (input_height - kernel_shape[0]+1, input_width - kernel_shape[1]+1)
        self.K = random.normal(loc=0.0, scale=1.0, size=(
            n_kernels, self.input_depth)+kernel_shape)
        self.B = random.normal(
            loc=0.0, scale=1.0, size=(self.output_shape))

    def _forward(self, X_in):
        self.X_in = X_in
        self.output = np.copy(self.B)
        
        for i in range(self.n_kernel):
            kernel = self.K[i]
            for x,k in zip(X_in,kernel):
                self.output[i] += signal.correlate2d(x,k, mode= 'valid')

        return (self.output)
        
    def _backward(self, grad_Y):
        delta_B = grad_Y
        self.B -= self.learning_rate * delta_B
        delta_K = np.zeros_like(self.K, dtype=np.float64)
        grad_EX = np.zeros_like(self.X_in, dtype=np.float64)
        for i in range(self.n_kernel):
            for j in range(self.input_depth):
                delta_K[i,j] += signal.correlate2d(self.X_in[j], grad_Y[i], mode= 'valid')
                grad_EX[j] += signal.convolve2d(grad_Y[i], self.K[i,j], mode = 'full')
        
        self.B -= self.learning_rate * delta_B
        self.K -= self.learning_rate * delta_K
        return grad_EX

class Reshape():
    def __init__(self, input_shape, output_shape) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape

    def _forward(self, X_in):
        return np.reshape(X_in, self.output_shape)

    def _backward(self, Y_in):
        return np.reshape(Y_in, self.input_shape)