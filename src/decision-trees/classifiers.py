#!/usr/bin/env python3

from sklearn import tree
from sklearn.datasets import load_iris

from utils import exportDecisionTree


def main():
    # 1. Minimal dataset

    # Create a tree classifier with a very simple dataset. A classifier stores
    # the results into different classes. In this example, the data is an array
    # of shape (size, weight). And the class are "man" and "women".
    classifier = createDecisionTreeClassifier(
        [[173, 75], [165, 55], [180, 93]],
        ["man", "woman", "man"],
    )
    exportDecisionTree("simple.pdf", classifier, ["man", "woman"], ["size", "weight"])

    # Create a sample to use as an example
    sample = [[179, 80]]

    # Try to predict if the couple (179, 80) is a "man" or a "woman"
    print(
        f"Simple dataset - Values: {sample}, decision: {classifier.predict(sample)}, probabilities: {classifier.predict_proba(sample)}"
    )

    # 2. Iris dataset

    # Load the Iris dataset. This dataset is way more bigger than the previous
    # one.
    iris_dataset = load_iris()

    # Create a tree classifier with the Iris dataset
    classifier = createDecisionTreeClassifier(
        iris_dataset.data,
        iris_dataset.target,
    )
    exportDecisionTree(
        "iris.pdf", classifier, iris_dataset.target_names, iris_dataset.feature_names
    )

    # Use the first item of the dataset as an example for predictions
    sample = iris_dataset.data[[55]]

    print(
        f"Iris dataset - Target names: {iris_dataset.target_names}, sample target: {iris_dataset.target[[55]]}"
    )

    print(
        f"Iris dataset - Values: {sample}, decision: {classifier.predict(sample)}, probabilities: {classifier.predict_proba(sample)}"
    )

    # 3. Tweak the DecisionTreeClassifier

    # Create a tree classifier with the Iris dataset and custom max_depth parameter
    classifier = createDecisionTreeClassifier(
        iris_dataset.data,
        iris_dataset.target,
        max_depth=2,
    )
    exportDecisionTree(
        "iris-custom-max-depth.pdf",
        classifier,
        iris_dataset.target_names,
        iris_dataset.feature_names,
    )

    print(
        f"Iris dataset (max_depth) - Values: {sample}, decision: {classifier.predict(sample)}, probabilities: {classifier.predict_proba(sample)}"
    )

    # Create a tree classifier with the Iris dataset and custom min_samples_leaf parameter
    classifier = createDecisionTreeClassifier(
        iris_dataset.data,
        iris_dataset.target,
        min_samples_leaf=5,
    )
    exportDecisionTree(
        "iris-custom-min-samples-leaf.pdf",
        classifier,
        iris_dataset.target_names,
        iris_dataset.feature_names,
    )

    print(
        f"Iris dataset (min_samples_leaf=5) - Values: {sample}, decision: {classifier.predict(sample)}, probabilities: {classifier.predict_proba(sample)}"
    )


# Build a decision tree classifier from the training set (X, y)
# X: The training input samples of shape (n_samples, n_features).
# y: The target values (class labels) as integers or strings.
# max_depth: The maximum depth of the tree.
# min_samples_leaf: The minimum number of samples required to be at a leaf node.
def createDecisionTreeClassifier(data, target, max_depth=None, min_samples_leaf=1):
    classifier = tree.DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )

    return classifier.fit(data, target)


if __name__ == "__main__":
    main()
