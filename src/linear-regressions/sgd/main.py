#!/usr/bin/env python3

from numpy import random, ravel
from matplotlib import pyplot
from sklearn.linear_model import SGDRegressor

# Create the dataset
random.seed(42)
X = 2 * random.rand(100, 1)
y = ravel(4 + 3 * X + random.randn(100, 1))

# Create the SGD regressor
reg = SGDRegressor()

# Fit the "model" w/ the dataset
reg.fit(X, y)

# Predict using the SGD regressor
pred = reg.predict(X)

pyplot.plot(X, y, "b.")
pyplot.plot(X, pred, "r")
pyplot.savefig("output.png")
