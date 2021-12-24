import numpy as np
from Layers import *
from Modelling import *
import matplotlib.pyplot as plt

X_train = np.reshape([[1, 1], [0, 1],[0.5,0.5], [1, 0], [0, 0]], (5, 2, 1))
y_train = np.reshape([[0], [1], [0], [1], [0]], (5, 1, 1))

network = [
    Layer(n_x=2, n_y = 3, eta=0.2),
    Activation(a_function = 'tanh'),
    Layer(n_x = 3, n_y = 1, eta=0.1),
    Activation(a_function = 'tanh')
]

fit(model = network, X_train= X_train, Y_train= y_train, epochs=1000)
vals = []
points = []
for x in np.linspace(0, 1, 20):
    for y in np.linspace(0, 1, 20):
        z = predict(network, [[x], [y]])
        points.append([x, y, z[0,0]])

points = np.array(points)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")
ax.scatter(X_train[:, 0], X_train[:, 1], y_train[:][:], color = 'r')
plt.show()