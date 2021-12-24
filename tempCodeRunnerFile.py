import numpy as np
from Layers import *
from post_cont import *
import matplotlib.pyplot as plt

X_train = np.reshape([[1, 1], [0, 1], [1, 0], [0, 0]], (4, 2, 1))
y_train = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = [
    Layer(n_x=2, n_y = 3, eta=0.2),
    Activation(a_function = 'tanh'),
    Layer(n_x = 3, n_y = 1, eta=0.1),
    Activation(a_function = 'tanh')
]