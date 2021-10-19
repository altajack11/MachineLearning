import numpy as np
import pandas as pd
import random as r
import math
import sys
# Imports all the collumn and attribute data from another file
from bank.formatted_data import bank_attribute_types, bank_columns, bank_attribute_values, bank_labels

# Declare global variables that get set for the data being processed.
columns = []
attribute_values = {}
attribute_types = {}
labels = []

trees = []

# region ID3


class Node:
    def __init__(self, branch_value=None, split_attribute=None, label=None):
        # The attribute this node splits on. i.e. "temperature".
        self.split_attribute = split_attribute
        # The attribute value taken to get to this node. i.e If the last split was on "temperature" this could be "hot", "mild", "cold".
        self.branch_value = branch_value
        # If the node is a leaf, this label represents the prediction.
        self.label = label
        # To determine numeric splits we must store the median that was used to split.
        self.median = 0
        self.children = []


# Calculates the entropy of a single subset
def calc_entropy(subset, data, column_name):
    subset_entropy = 0
    if subset.shape[0] == 0:
        return 0
    for label in labels:
        p = subset[subset['label'] == label].shape[0]
        if p == 0:
            continue
        subset_entropy -= (p / subset.shape[0]) * \
            math.log2(p / subset.shape[0])
    return subset_entropy * \
        (subset.shape[0] / data[column_name].shape[0])


# Calculates the entire information gain of splitting on a collumn name
def ig_gain(data, column_name):
    before_split_entropy = 0
    for p in data['label'].value_counts(normalize=True):
        before_split_entropy -= p * math.log2(p+1e-8)

    after_split_entropy = 0
    if attribute_types[column_name] == "numeric":
        median = np.median(data[column_name])
        below_avg_subset = data[data[column_name] <= median]
        above_avg_subset = data[data[column_name] > median]
        after_split_entropy += calc_entropy(below_avg_subset,
                                            data, column_name)
        after_split_entropy += calc_entropy(above_avg_subset,
                                            data, column_name)

    else:
        for val in attribute_values[column_name]:
            sv = data[data[column_name] == val]
            after_split_entropy += calc_entropy(sv, data, column_name)

    return before_split_entropy - after_split_entropy


# Construct an ID3 tree from the given data, eventually returns the root node of the tree.
def ID3(data, candidate_attributes, branch_value, depth, max_depth, size):
    root = Node(branch_value=branch_value)

    _labels = data['label'].to_numpy()
    if np.all(_labels == _labels[0]):
        root.label = _labels[0]
        return root

    if len(candidate_attributes) == 0 or depth >= max_depth:
        root.label = data['label'].mode()[0]
        return root

    best_score = float("-inf")

    G = []
    for _ in range(size):
        G.append(candidate_attributes[r.randint(
            0, len(candidate_attributes)-1)])

    split_column = None

    for column_name in G:
        if data[[column_name]].shape[0] > 0:
            _score = ig_gain(data[[column_name, 'label']],
                             column_name)
            if _score > best_score:
                best_score = _score
                split_column = column_name

    root.split_attribute = split_column

    # If it's numeric, split on the median
    if attribute_types[split_column] == "numeric":
        median = np.median(data[split_column])
        root.median = median
        below_avg_subset = data[data[split_column] <= median]
        above_avg_subset = data[data[split_column] > median]
        node_or_ID3_call(root, data, candidate_attributes, split_column, depth,
                         max_depth,  below_avg_subset, "below_avg", size)
        node_or_ID3_call(root, data, candidate_attributes, split_column, depth,
                         max_depth, above_avg_subset, "above_avg", size)

    # If it's categorical split on each category
    else:
        for val in attribute_values[split_column]:
            subset = data[data[split_column] == val]
            node_or_ID3_call(root, data, candidate_attributes, split_column,
                             depth, max_depth,  subset, val, size)

    return root


# Help method to determine if we should make a leaf/label for this attribute or if we should continue the ID3 recursion
def node_or_ID3_call(root, example_data, attributes, best_col_name, depth, max_depth, subset, val, size):
    if subset.shape[0] == 0:
        root.children.append(
            Node(branch_value=val, label=example_data['label'].mode()[0]))

    else:
        root.children.append(
            ID3(subset, [a for a in attributes if a != best_col_name], val, depth + 1, max_depth, size))


def PredictRow(root, row):
    node = root
    while node.split_attribute:
        for child in node.children:
            if (child.branch_value == "below_avg"):
                if (node.median >= getattr(row, node.split_attribute)):
                    node = child
                    break
            elif (child.branch_value == "above_avg"):
                if (node.median < getattr(row, node.split_attribute)):
                    node = child
                break
            elif child.branch_value == getattr(row, node.split_attribute):
                node = child
                break
    return node.label
# endregion

# region Bagged


def Bagged(training_data, columns, size):
    m = training_data.shape[0]
    # Pick t samples
    samples = []
    for _ in range(m):
        index = r.randint(0, m-1)
        s = training_data.iloc[[index]]
        samples.append(s)
    x_t = pd.concat(samples)
    trees.append(ID3(x_t, columns, None, 0, 20, size))


def BaggedPredict(data):
    results = []
    for row in data.itertuples(index=False):
        results.append(BaggedPredictRow(row))
    return np.mean(results == data['label'])


def BaggedPredictRow(row):
    predictions = []
    for tree in trees:
        predictions.append(PredictRow(tree, row))
    return max(set(predictions), key=predictions.count)


def BaggedPredictRow(_trees, row):
    predictions = []
    for tree in _trees:
        predictions.append(PredictRow(tree, row))
    return max(set(predictions), key=predictions.count)
# endregion


def Report_2_2_e(_columns, _attribute_values, _attribute_types, _labels, training_data_filepath, test_data_filepath):
    print("Random Forests Bias/Var")
    # Get and set the global variables
    global columns
    global attribute_values
    global attribute_types
    global labels
    columns = _columns
    attribute_values = _attribute_values
    attribute_types = _attribute_types
    labels = _labels

    training_data = pd.read_csv(training_data_filepath)
    training_data.columns = columns
    test_data = pd.read_csv(test_data_filepath)
    test_data.columns = columns

    bagged_predictors = {}
    for i in range(100):
        sample_data = training_data.sample(n=1000)
        for t in range(10):
            Bagged(sample_data, columns[:-1], 4)
        bagged_predictors[i] = trees.copy()
        trees.clear()
        print(i)

    bias_sum = 0
    var_sum = 0
    for row in test_data.itertuples(index=False):
        # Compute prediction of 100 single trees.
        predictions = []
        for i in range(100):
            root = bagged_predictors[i][0]
            predictions.append(1 if PredictRow(root, row) == "yes" else 0)

        avg = np.mean(predictions)
        gtl = 1 if row.label == "yes" else 0
        bias_sum += (gtl-avg)**2

        sum = 0
        for i in range(100):
            sum += ((1 if row.label == 'yes' else 0) - avg)**2
        var_sum += sum/99
    avg_bias_single = bias_sum / 100
    avg_var_single = var_sum / 100
    general_squared_error_single = avg_bias_single + avg_var_single

    print("Single, Bias, Var, GSE")
    print(f"{avg_bias_single}, {avg_var_single}, {general_squared_error_single}")

    bias_sum = 0
    var_sum = 0
    for row in test_data.itertuples(index=False):
        # Compute prediction of 100 single trees.
        predictions = []
        for i in range(100):
            predictions.append(1 if BaggedPredictRow(
                bagged_predictors[i], row) == "yes" else 0)

        avg = np.mean(predictions)
        gtl = 1 if row.label == "yes" else 0
        bias_sum += (gtl-avg)**2

        sum = 0
        for i in range(100):
            sum += ((1 if row.label == 'yes' else 0) - avg)**2
        var_sum += sum/99
    avg_bias_bagged = bias_sum / 100
    avg_var_bagged = var_sum / 100
    general_squared_error_bagged = avg_bias_bagged + avg_var_bagged

    print("Bagged, Bias, Var, GSE")
    print(f"{avg_bias_bagged}, {avg_var_bagged}, {general_squared_error_bagged}")


def Report_2_2_d(_columns, _attribute_values, _attribute_types, _labels, training_data_filepath, test_data_filepath):
    # Get and set the global variables
    global columns
    global attribute_values
    global attribute_types
    global labels
    columns = _columns
    attribute_values = _attribute_values
    attribute_types = _attribute_types
    labels = _labels

    training_data = pd.read_csv(training_data_filepath)
    training_data.columns = columns
    test_data = pd.read_csv(test_data_filepath)
    test_data.columns = columns

    print("Random")
    feature_subset_sizes = [2, 4, 6]
    for size in feature_subset_sizes:
        print(f"Feature Subset Size = {size}")
        print("T, train, test")
        for T in range(1, 501):
            Bagged(training_data, columns[:-1], size)
            result1 = "{0:.4f}".format(BaggedPredict(training_data))
            result2 = "{0:.4f}".format(BaggedPredict(test_data))
            print(f"{T}, {result1}, {result2}")


def main():
    if len(sys.argv) == 2:
        if sys.argv[1] == '3':
            Report_2_2_d(bank_columns, bank_attribute_values, bank_attribute_types,
                         bank_labels,  "EnsembleLearning/bank/train.csv", "EnsembleLearning/bank/test.csv")
        elif sys.argv[1] == '4':
            Report_2_2_e(bank_columns, bank_attribute_values, bank_attribute_types,
                         bank_labels,  "EnsembleLearning/bank/train.csv", "EnsembleLearning/bank/test.csv")
        else:
            print("Invalid arguement given.")


main()
