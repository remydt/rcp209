#!/usr/bin/env python3

import numpy

from matplotlib import pyplot
from sklearn import tree
from sklearn.datasets import load_iris

from utils import exportDecisionTree


def main():
    # 1. Minimal dataset

    # Create a tree regressor with a very simple dataset.
    regressor = createDecisionTreeRegressor(
        [[173, 75], [165, 55], [180, 93]],
        [0, 1, 0],
    )
    exportDecisionTree("regressor-simple.pdf", regressor)

    # Create a sample to use as an example
    sample = [[179, 80]]

    # Try to predict if the couple (179, 80) is a "man" or a "woman"
    print(f"Simple dataset - Values: {sample}, decision: {regressor.predict(sample)}")

    # 2. Sinusoidal signal

    # Generate the dataset
    data = numpy.sort(5 * numpy.random.rand(80, 1))

    target = numpy.sin(data).ravel()

    noise_range = 0
    max_depth = 4

    # Add some noise every X items
    target[::] += 3 * (0.5 - numpy.random.rand(80))

    # Create a tree regressor with this dataset
    regressor = createDecisionTreeRegressor(data, target, max_depth=max_depth)

    # Extract a sample from the dataset
    sample = numpy.arange(0.0, 5.0, 0.01)[:, numpy.newaxis]

    # Predict
    prediction = regressor.predict(sample)

    # Show the results
    pyplot.figure()

    # Add the raw data and targets
    pyplot.scatter(data, target, label=f"noise_range: {noise_range}")

    # Show the prediction
    pyplot.plot(
        sample, prediction, color="orange", linewidth=2, label=f"max_range: {max_depth}"
    )

    pyplot.xlabel("Data")
    pyplot.ylabel("Target")
    pyplot.legend()

    pyplot.savefig("regressor-sinusoid.png")


# Build a decision tree regressor from the training set (X, y)
# X: The training input samples of shape (n_samples, n_features).
# y: The target values (class labels) as integers or strings.
# max_depth: The maximum depth of the tree.
# min_samples_leaf: The minimum number of samples required to be at a leaf node.
def createDecisionTreeRegressor(data, target, max_depth=None):
    regressor = tree.DecisionTreeRegressor(
        max_depth=max_depth,
    )

    return regressor.fit(data, target)


if __name__ == "__main__":
    main()
