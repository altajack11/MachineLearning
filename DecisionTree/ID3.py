import numpy as np
import pandas as pd
import math
# Imports all the collumn and attribute data from another file
from formatted_data import car_attribute_types, car_columns, car_attribute_values, car_labels, bank_attribute_types, bank_columns, bank_attribute_values, bank_labels

# Declare global variables that get set for the data being processed.
columns = []
attribute_values = {}
attribute_types = {}
labels = []

# Simple node class for tree implementation.


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


# Calculates the majority error of a single subset
def calc_me(subset, data, column_name):
    subset_me = 0
    if subset.shape[0] == 0:
        return 0
    for label in labels:
        subset_me = max(subset_me, subset[subset['label'] == label].shape[0])
    subset_me = (subset.shape[0] - subset_me) / subset.shape[0]
    return subset_me * \
        (subset.shape[0] / data[column_name].shape[0])


# Calculates the entire majority error of splitting on a collumn name
def me_gain(data, column_name):
    before_split_me = 0
    for p in data['label'].value_counts():
        before_split_me = max(p, before_split_me)

    before_split_me = (data['label'].shape[0] -
                       before_split_me) / data['label'].shape[0]

    after_split_me = 0
    if attribute_types[column_name] == "numeric":
        median = np.median(data[column_name])
        below_avg_subset = data[data[column_name] <= median]
        above_avg_subset = data[data[column_name] > median]
        after_split_me += calc_me(below_avg_subset, data, column_name)
        after_split_me += calc_me(above_avg_subset, data, column_name)
    else:
        for a_val in attribute_values[column_name]:
            sv = data[data[column_name] == a_val]
            after_split_me += calc_me(sv, data, column_name)

    return before_split_me - after_split_me


# Calculates the gini index of a single subset
def calc_gi(subset, data, column_name):
    subset_gi = 0
    if subset.shape[0] == 0:
        return 0
    for label in labels:
        p = subset[subset['label'] == label].shape[0]
        subset_gi += (p / subset.shape[0])**2
    subset_gi = 1 - subset_gi
    return subset_gi * (subset.shape[0] / data[column_name].shape[0])


# Calculates the entire gini index of splitting on a collumn name
def gi_gain(data, column_name):
    before_split_gi = 0
    for p in data['label'].value_counts(normalize='True'):
        before_split_gi += p**2

    before_split_gi = 1 - before_split_gi

    after_split_gi = 0
    if attribute_types[column_name] == "numeric":
        median = np.median(data[column_name])
        below_avg_subset = data[data[column_name] <= median]
        above_avg_subset = data[data[column_name] > median]
        after_split_gi += calc_gi(below_avg_subset, data, column_name)
        after_split_gi += calc_gi(above_avg_subset, data, column_name)
    else:
        for a_val in attribute_values[column_name]:
            sv = data[data[column_name] == a_val]
            after_split_gi += calc_gi(sv, data, column_name)

    return before_split_gi - after_split_gi


# Determines which scoring method to use and returns the score for this collumn
def score(data, column_name, gain_type):
    if gain_type == 'ME':
        return me_gain(data, column_name)
    elif gain_type == 'GI':
        return gi_gain(data, column_name)
    else:
        return ig_gain(data, column_name)


# Construct an ID3 tree from the given data, eventually returns the root node of the tree.
def ID3(data, candidate_attributes, branch_value, depth, max_depth, gain_type):
    root = Node(branch_value=branch_value)

    _labels = data['label'].to_numpy()
    if np.all(_labels == _labels[0]):
        root.label = _labels[0]
        return root

    if len(candidate_attributes) == 0 or depth >= max_depth:
        root.label = data['label'].mode()[0]
        return root

    best_score = float("-inf")
    split_column = None
    for column_name in candidate_attributes:
        if data[[column_name]].shape[0] > 0:
            _score = score(data[[column_name, 'label']],
                           column_name, gain_type)
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
                         max_depth, gain_type, below_avg_subset, "below_avg")
        node_or_ID3_call(root, data, candidate_attributes, split_column, depth,
                         max_depth, gain_type, above_avg_subset, "above_avg")

    # If it's categorical split on each category
    else:
        for val in attribute_values[split_column]:
            subset = data[data[split_column] == val]
            node_or_ID3_call(root, data, candidate_attributes, split_column,
                             depth, max_depth, gain_type, subset, val)

    return root


# Help method to determine if we should make a leaf/label for this attribute or if we should continue the ID3 recursion
def node_or_ID3_call(root, example_data, attributes, best_col_name, depth, max_depth, gain_type, subset, val):
    if subset.shape[0] == 0:
        root.children.append(
            Node(branch_value=val, label=example_data['label'].mode()[0]))

    else:
        root.children.append(
            ID3(subset, [a for a in attributes if a != best_col_name], val, depth + 1, max_depth, gain_type))


# For each row of the test data, use the ID# tree to make a prediction and put it in a list.
def Predict(root, test):
    results = []
    for row in test.itertuples(index=False):
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
        results.append(node.label)
    return results


# Replace any unknown value with the most common value in that column
def replace_unknowns(data, columns):

    for col in columns:
        sorted_by_count = data[col].value_counts().sort_values(
            ascending=False).index.tolist()
        if (sorted_by_count[0] == "unknown"):
            sorted_by_count[0] = sorted_by_count[1]
        data.replace({col: 'unknown'}, sorted_by_count[0], inplace=True)


# Print a report with all the given information
def generate_report(_columns, _attribute_values, _attribute_types, _labels, max_max_depth, training_data_filepath, test_data_filepath, fill_unknowns=False):
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
    if(fill_unknowns):
        replace_unknowns(training_data, columns)
        replace_unknowns(test_data, columns)

    tests = [{"name": "test ", "data": test_data},
             {"name": "train", "data": training_data}]
    gain_types = ["IG", "ME", "GI"]
    create_decorator(max_max_depth)
    for gain_type in gain_types:
        for test in tests:
            print(f"{test['name']} {gain_type}", end=" ")
            for max_depth in range(1, max_max_depth + 1):
                trained_tree = ID3(
                    training_data, columns[:-1], None, 0, max_depth, gain_type)
                predictions = Predict(trained_tree, test['data'])
                result = "{0:.4f}".format(
                    np.mean(predictions == test['data']['label']))
                print(
                    f"{result}", end=" ")
            print()


# Creates a decorator to better describe the printout
def create_decorator(max_max_depth):
    decorator = "Data  SC"
    for i in range(1, max_max_depth+1):
        if i > 9:
            decorator += f" max={i}"
        else:
            decorator += f"  max={i}"
    print(decorator)
    line = ""
    for i in range(len(decorator)):
        line += "-"
    print(line)


def main():
    print("The bank report takes a long time to run and the rows also fill up all at once, so if it seems like nothing's happening, know that it is working in the background")
    generate_report(car_columns, car_attribute_values, car_attribute_types,
                    car_labels, 6, "DecisionTree/car/train.csv", "DecisionTree/car/test.csv")
    generate_report(bank_columns, bank_attribute_values, bank_attribute_types,
                    bank_labels, 16, "DecisionTree/bank/train.csv", "DecisionTree/bank/test.csv")
    generate_report(bank_columns, bank_attribute_values, bank_attribute_types,
                    bank_labels, 16, "DecisionTree/bank/train.csv", "DecisionTree/bank/test.csv", fill_unknowns=True)


main()
