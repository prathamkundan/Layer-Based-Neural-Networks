import numpy as np

def mse(calc, actual):
    delta = calc-actual
    error_mat = np.square(delta)/(calc.shape[0])
    error = np.sum(error_mat, axis=0)
    return error

def mse_prime(calc, actual):
    op_prime = 2*(calc-actual)/calc.size
    return op_prime