import numpy as np
import pandas as pd
import math
import sys
from bank.formatted_data import bank_attribute_types, bank_columns, bank_attribute_values, bank_labels

columns = []
attribute_values = {}
attribute_types = {}
labels = []

A_t = []
H_t = []

# region ID3


class Node:
    def __init__(self, branch_value=None, split_attribute=None, label=None):
        self.split_attribute = split_attribute
        self.branch_value = branch_value
        self.label = label
        self.median = 0
        self.children = []


def calc_entropy(subset, data):
    subset_entropy = 0
    if subset.shape[0] == 0:
        return 0
    total_score = sum(subset["score"])
    for label in labels:
        p = sum(subset[subset['label'] == label]["score"])/total_score
        if p == 0:
            continue
        subset_entropy -= (p) * math.log2(p)
    return subset_entropy * (total_score / sum(data["score"]))


def ig_gain(data, column_name):
    before_split_entropy = 0

    p1 = sum(data[data['label'] == "yes"]['score'])
    p0 = sum(data[data['label'] == "no"]['score'])
    if p1 != 0:
        before_split_entropy -= p1 * math.log2(p1)
    if p0 != 0:
        before_split_entropy -= p0 * math.log2(p0)

    after_split_entropy = 0
    if len(attribute_values[column_name]) == 0:
        #median = np.median(data[column_name])
        median = _median(data, column_name, "score")
        below_avg_subset = data[data[column_name] <= median]
        above_avg_subset = data[data[column_name] > median]
        after_split_entropy += calc_entropy(below_avg_subset,
                                            data)
        after_split_entropy += calc_entropy(above_avg_subset,
                                            data)

    else:
        for val in attribute_values[column_name]:
            sv = data[data[column_name] == val]
            after_split_entropy += calc_entropy(sv, data)
    return before_split_entropy - after_split_entropy


def _median(df, val, weight):
    df_sorted = df.sort_values(val)
    cumsum = df_sorted[weight].cumsum()
    cutoff = df_sorted[weight].sum() / 2.
    return df_sorted[cumsum >= cutoff][val].iloc[0]

# Construct an ID3 tree from the given data, eventually returns the root node of the tree.


def ID3(data, candidate_attributes, branch_value, depth, max_depth):
    root = Node(branch_value=branch_value)

    _labels = data['label'].to_numpy()
    if np.all(_labels == _labels[0]):
        root.label = _labels[0]
        return root

    if len(candidate_attributes) == 0 or depth >= max_depth:
        p1 = sum(data[data['label'] == "yes"]['score'])
        p0 = sum(data[data['label'] == "no"]['score'])
        root.label = "yes" if p1 > p0 else "no"
        return root

    best_score = float("-inf")
    split_column = None
    for column_name in candidate_attributes:
        if data[[column_name]].shape[0] > 0:
            _score = ig_gain(data[[column_name, 'score', 'label']],
                             column_name)
            if _score > best_score:
                best_score = _score
                split_column = column_name

    root.split_attribute = split_column

    # If it's numeric, split on the median
    if len(attribute_values[split_column]) == 0:
        #median = np.median(data[split_column])
        median = _median(data, split_column, "score")
        root.median = median
        below_avg_subset = data[data[split_column] <= median]
        above_avg_subset = data[data[split_column] > median]
        node_or_ID3_call(root, data, candidate_attributes, split_column, depth,
                         max_depth, below_avg_subset, "below_avg")
        node_or_ID3_call(root, data, candidate_attributes, split_column, depth,
                         max_depth, above_avg_subset, "above_avg")

    # If it's categorical split on each category
    else:
        for val in attribute_values[split_column]:
            subset = data[data[split_column] == val]
            node_or_ID3_call(root, data, candidate_attributes, split_column,
                             depth, max_depth, subset, val)

    return root


# Helper method to determine if we should make a leaf/label for this attribute or if we should continue the ID3 recursion
def node_or_ID3_call(root, example_data, attributes, best_col_name, depth, max_depth, subset, val):
    if subset.shape[0] == 0:
        p1 = sum(example_data[example_data['label'] == "yes"]['score'])
        p0 = sum(example_data[example_data['label'] == "no"]['score'])

        root.children.append(
            Node(branch_value=val, label=("yes" if p1 > p0 else "no")))

    else:
        root.children.append(
            ID3(subset, [a for a in attributes if a != best_col_name], val, depth + 1, max_depth))

# For each row of the test data, use the ID# tree to make a prediction and put it in a list.


def Predict(root, test):
    results = []
    for row in test.itertuples(index=False):
        results.append(PredictRow(root, row))
    return np.mean(results == test['label'])


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


def AdaBoost(training_data, columns, T):
    global H_t, A_t
    H_t = [0]*T
    A_t = [0]*T
    d_1 = [1/float(training_data.shape[0])]*training_data.shape[0]
    training_data["score"] = d_1

    for t in range(1, T+1):
        # Find classifier h_t
        h_t = ID3(training_data, columns, None, 0, 1)

        # Compute its vote
        e_t = 0.0
        for x in training_data.itertuples(index=False):
            h_x_i = PredictRow(h_t, x)
            y_i = x.label
            if(h_x_i != y_i):
                e_t += x.score
        a_t = 0.5 * np.log((1-e_t)/e_t)

        # Update the weights for the training examples
        index = 0
        for x in training_data.itertuples(index=False):
            h_x_i = PredictRow(h_t, x)
            y_i = x.label
            v = 0
            if(h_x_i == y_i):
                v = math.exp(-a_t)
            else:
                v = math.exp(a_t)
            training_data.at[index, 'score'] = x.score*v
            index += 1

        index = 0
        z_t = float(sum(training_data.score))
        for x in training_data.itertuples(index=False):
            training_data.at[index, 'score'] = x.score/z_t
            index += 1

        # Store values for final predictions later
        H_t[t-1] = h_t
        A_t[t-1] = a_t


def AdaPredict(data):
    results = []
    for x in data.itertuples(index=False):
        sum = 0
        for t in range(len(A_t)):
            prediction = PredictRow(H_t[t], x)
            p = 0
            if prediction == "yes":
                p = 1
            else:
                p = -1
            sum += A_t[t]*p
        if(sum < 0):
            results.append("no")
        else:
            results.append("yes")

    return np.mean(results == data['label'])


def AdaStumpPredict(data):
    results = []
    for stump in H_t:
        results.append(Predict(stump, data))
    return np.mean(results)


def Report_2_2_a(_columns, _attribute_values, _attribute_types, _labels, training_data_filepath, test_data_filepath):
    # Get and set the global variables
    global columns, attribute_values, attribute_types, labels
    columns = _columns
    attribute_values = _attribute_values
    attribute_types = _attribute_types
    labels = _labels

    training_data = pd.read_csv(training_data_filepath)
    training_data.columns = columns

    test_data = pd.read_csv(test_data_filepath)
    test_data.columns = columns
    print("AdaBoost")
    print("T, train, test, stumptrain, stumptest")
    for T in range(1, 501, 5):
        AdaBoost(training_data, columns[:-1], T)
        result1 = "{0:.4f}".format(AdaPredict(training_data))
        result2 = "{0:.4f}".format(AdaPredict(test_data))
        result3 = "{0:.4f}".format(AdaStumpPredict(training_data))
        result4 = "{0:.4f}".format(AdaStumpPredict(test_data))
        print(f"{T}, {result1}, {result2}, {result3}, {result4}")


def main():
    if len(sys.argv) == 2:
        if sys.argv[1] == '0':
            Report_2_2_a(bank_columns, bank_attribute_values, bank_attribute_types,
                         bank_labels,  "EnsembleLearning/bank/train.csv", "EnsembleLearning/bank/test.csv")
        else:
            print("Invalid argument given.")


main()
