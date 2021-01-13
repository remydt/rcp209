#!/usr/bin/env python3

import numpy

from matplotlib import pyplot
from sklearn.model_selection import train_test_split


def main():
    # 1. Create a first dataset from a multivariate normal distribution. See:
    #    https://numpy.org/doc/stable/reference/random/generated/numpy.random.multivariate_normal.html
    data_class_1 = (
        numpy.random.default_rng()
        .multivariate_normal(
            [0, 0],  # The center of the multivariate normal distribution
            [[10, 5], [5, 4]],  # The covariance matrix
            2,  # The number of elements to create
        )
        .T
    )

    # Create a diagram for the first dataset
    pyplot.scatter(data_class_1[0, :], data_class_1[1, :])

    # 2. Create a second dataset. This one is a combination of 4 multivariate normal
    #    distributions with different centers.
    data_class_2 = create_data_class_2([[-10, 2], [-7, -2], [-2, -6], [5, -7]])
    print(data_class_2)

    # Create a diagram for the second dataset
    # pyplot.scatter(data_class_2[0, :], data_class_2[1, :])

    # # Contatenate the data classes
    # dataset = numpy.concatenate((data_class_1, data_class_2))

    # # Create labels to identify classes in the dataset (the first 100 are from the first class and the
    # # last 100 are from the second one)
    # labels = numpy.concatenate(
    #     (numpy.ones(100, dtype=int), numpy.zeros(100, dtype=int))
    # )

    # 3. Let's train

    # 4. Linear model

    # Export the diagram
    pyplot.savefig("data-classes.png")


def create_data_class_2(centers):
    data_class_2 = numpy.empty([0, 2])

    for center in centers:
        dataset = (
            numpy.random.default_rng()
            .multivariate_normal(
                center,
                [
                    [10, 5],
                    [5, 4],
                ],  # We are using a fixed value for the covariance matrix
                2,
            )
            .T
        )

        numpy.concatenate((data_class_2, dataset))

    return data_class_2


if __name__ == "__main__":
    main()
