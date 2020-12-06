#!/usr/bin/env python3

import pydotplus

from sklearn import tree

# Export the decision tree as a PDF file
def exportDecisionTree(
    output_filename, classifier, class_names=None, feature_names=None
):
    pydotplus.graph_from_dot_data(
        tree.export_graphviz(
            classifier,
            out_file=None,
            class_names=class_names,
            feature_names=feature_names,
            filled=True,
            special_characters=True,
        )
    ).write_pdf(output_filename)
