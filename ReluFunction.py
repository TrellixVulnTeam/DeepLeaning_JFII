import numpy as np
import matplotlib.pylab as plt


def relu_function(x):
    return np.maximum(0, x)


x = np.arange(-5, 5, 0.1)
y = relu_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
