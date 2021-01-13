#!/usr/bin/env python3

import numpy

from matplotlib import pyplot
from sklearn import svm, datasets, model_selection


# def classify(
#     data,
#     target,
#     C=0.1,
#     test_size=0.3,
# ):
#     X_train, _, y_train, _ = model_selection.train_test_split(
#         data, target, test_size=test_size
#     )

#     return svm.LinearSVC(C=0.1).fit(X_train, y_train).score(X_train, y_train)


def classify(
    C,
    data,
    target,
):
    X_train, _, y_train, _ = model_selection.train_test_split(
        data, target, test_size=0.8
    )

    return svm.LinearSVC(C=C).fit(X_train, y_train)


def draw(data, linear_svc, target, title="linear-svc-iris"):
    h = 0.02

    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1

    xx, yy = numpy.meshgrid(
        numpy.arange(x_min, x_max, h), numpy.arange(y_min, y_max, h)
    )

    Z = linear_svc.predict(numpy.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    pyplot.contourf(xx, yy, Z, cmap=pyplot.cm.coolwarm, alpha=0.8)

    pyplot.scatter(data[:, 0], data[:, 1], c=target, cmap=pyplot.cm.coolwarm)

    pyplot.xlabel("Sepal length")
    pyplot.ylabel("Sepal width")

    pyplot.xlim(xx.min(), xx.max())
    pyplot.ylim(yy.min(), yy.max())

    pyplot.xticks(())
    pyplot.yticks(())

    pyplot.savefig(f"{title}.png")


def main():
    # Load the Iris dataset
    iris_dataset = datasets.load_iris()

    # Keep only two attributes from the dataset
    data = iris_dataset.data[:, :2]
    target = iris_dataset.target

    # 1./2.
    # Save scores of SVM classifications
    # scores = []
    # for i in range(1, 20):
    #     for _ in range(10):
    #         score = classify(data, target, test_size=i / 20)
    #         scores.append(score)

    # scores.reverse()

    # pyplot.plot(scores)
    # pyplot.savefig("scores.png")

    # 3./4.
    # Try to change C
    for i in range(1, 16):
        linear_svm = classify(i / 10, data, target)

        draw(data, linear_svm, target, title=f"{i}-linear-svm")


if __name__ == "__main__":
    main()
