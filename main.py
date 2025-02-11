import matplotlib.pyplot as plt
import nnfs
import nnfs.datasets
import numpy as np

from core.dense_layer import Layer_Dense

nnfs.init()

X, y = nnfs.datasets.spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 1)

dense1.forward(X)

print(dense1.output[:5])
