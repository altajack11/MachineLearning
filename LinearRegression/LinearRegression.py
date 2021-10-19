import numpy as np
import pandas as pd
import math
import sys

# Declare global variables that get set for the data being processed.
columns = ['Cement',
           'Slag',
           'Fly ash',
           'education',
           'default',
           'balance',
           'housing',
           "output"]
y = 8

# region helper functions


def NormWeight(w_t, w):
    v = [0]*len(w_t)
    for i in range(len(w_t)):
        v[i] = (w_t[i] - w[i]) ** 2

    return math.sqrt(sum(v))


def Transpose(w, x):
    result = 0
    for i in range(len(w)):
        result += w[i] * x[i]
    return result


def Cost(w, data):
    sum = 0
    for x_i in data.itertuples(index=False):
        sum += (x_i[y] - Transpose(w, x_i))**2
    return sum / 2
# endregion

# region batch


def BatchGradientDescent(data, r):
    w = [0]*y
    w_t = [0]*y
    diff = float('inf')
    i = 0
    print(f"J(w{i}): {Cost(w, data)}")
    while diff > 10**(-6):
        Delta_J = [0]*y
        for j in range(y):
            sum = 0
            for x_i in data.itertuples(index=False):
                sum -= (x_i[y] - Transpose(w, x_i))*x_i[j]
            Delta_J[j] = sum
        for j in range(y):
            w_t[j] = w[j] - r*Delta_J[j]
        diff = NormWeight(w_t, w)
        w = w_t.copy()
        i += 1
        print(f"J(w{i}): {Cost(w, data)}")
    return w


def generate_batch_report(training_data_filepath, test_data_filepath):
    # Get and set the global variables
    training_data = pd.read_csv(training_data_filepath)
    training_data.columns = columns
    d_1 = [1]*training_data.shape[0]
    training_data.insert(0, "bias", d_1)

    test_data = pd.read_csv(test_data_filepath)
    test_data.columns = columns
    d_1 = [1]*test_data.shape[0]
    test_data.insert(0, "bias", d_1)

    print("Linear Regression: Batch Gradient Descent")
    r = 0.015625
    w = BatchGradientDescent(training_data, r)

    print("Learned Weight Vector:")
    print(w)
    print("Learing Rate r:")
    print(r)
    print("Cost of Test Data")
    print(Cost(w, test_data))
# endregion

# region stochatic


def StochasticGradientDescent(data, r):
    w = [0]*y
    diff = float('inf')
    for _ in range(1000):
        for x_i in data.itertuples(index=False):
            for j in range(y):
                w[j] += r*(x_i[y] - Transpose(w, x_i))*x_i[j]
    return w


def generate_stochastic_report(training_data_filepath, test_data_filepath):
    # Get and set the global variables
    training_data = pd.read_csv(training_data_filepath)
    training_data.columns = columns
    d_1 = [1]*training_data.shape[0]
    training_data.insert(0, "bias", d_1)

    test_data = pd.read_csv(test_data_filepath)
    test_data.columns = columns
    d_1 = [1]*test_data.shape[0]
    test_data.insert(0, "bias", d_1)

    print("Linear Regression: Stochastic Gradient Descent")
    r = 0.015625
    w = StochasticGradientDescent(training_data, r)

    print("Learned Weight Vector:")
    print(w)
    print("Learing Rate r:")
    print(r)
    print("Cost of Test Data")
    print(Cost(w, test_data))
# endregion


def main():

    if len(sys.argv) == 2:
        if sys.argv[1] == '5':
            generate_batch_report("LinearRegression/concrete/train.csv",
                                  "LinearRegression/concrete/test.csv")
        elif sys.argv[1] == '6':
            generate_stochastic_report(
                "LinearRegression/concrete/train.csv", "LinearRegression/concrete/test.csv")
        else:
            print("Invalid argument given.")


main()
