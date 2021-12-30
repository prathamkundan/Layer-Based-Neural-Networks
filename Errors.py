from matplotlib.pyplot import axis
import numpy as np

class Error_functions():
    def mse(calc, actual):
        delta = calc-actual
        error_mat = np.square(delta)/(calc.shape[1])
        error = np.sum(error_mat, axis=0)
        return error

    def mse_prime(calc, actual):
        op_prime = 2*(calc-actual)/calc.shape[1]
        return op_prime

    def binary_entropy_loss(calc, actual):
        error_mat = -1/calc.shape[0]*((actual * np.log(calc)) + ((1-actual)*np.log(1-calc)))
        error = np.sum(error_mat, axis = 0)
        return error

    def bel_prime(calc,actual):
        return ((1-actual)/(1-calc) - actual/calc)/calc.shape[0]

    def __init__(self, function = 'mse'):
        if (function == 'mse'):
            self.get_error = lambda calc,actual: Error_functions.mse(calc, actual) 
            self.get_error_gradient =  lambda calc,actual: Error_functions.mse(calc, actual)
        if (function == 'bel'):
            self.get_error = lambda calc,actual: Error_functions.binary_entropy_loss(calc, actual) 
            self.get_error_gradient =  lambda calc,actual: Error_functions.bel_prime(calc, actual)
