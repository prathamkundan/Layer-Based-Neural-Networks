# Testing Neural Network for xor dataset

from matplotlib import colors
import numpy as np
from numpy.random.mtrand import shuffle
from coNNstruct.Layers import *
from coNNstruct.Modelling import *
import matplotlib.pyplot as plt


X_train = np.array([[1, 1], [0, 1], [1, 0], [0, 0]])
y_train = np.array([[0], [1], [1], [0]])

network = [
    Layer(n_x=2, n_y = 3, eta=0.3, regulariser=Regularisation.l1(0.001)),
    Activation(function = 'sigmoid'),
    Layer(n_x = 3, n_y = 1, eta=0.3, regulariser=Regularisation.l1(0.001)),
    Activation(function = 'sigmoid')
]

err = fit(model = network, X_train= X_train, Y_train= y_train, epochs=10000, shuffle = False)
vals = []
points = []
for x in np.linspace(0, 1, 20):
    for y in np.linspace(0, 1, 20):
        z = predict(network, [[x, y]])
        points.append([x, y, z[0,0]])

points = np.array(points)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")
ax.scatter(X_train[:, 0], X_train[:, 1], y_train[:][:], color = 'r')
plt.show()