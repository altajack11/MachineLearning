import numpy as np
import pandas as pd
import math
import sys

# Declare global variables that get set for the data being processed.
columns = ['variance',
           'skewness',
           'curtosis',
           'entropy',
           "label"]
y = 5

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


def VectorScale(m, s):
    v = [0]*len(m)
    for i in range(len(m)):
        v[i] = m[i]*s
    return v


def VectorAdd(m, n):
    if len(m) != len(n):
        raise "error"
    v = [0]*len(m)
    for i in range(len(m)):
        v[i] = m[i]+n[i]
    return v


# endregion

# region Averaged
def Averaged_Predict(w, data):
    predictions = []
    for x_i in data.itertuples(index=False):
        classified = Transpose(w, x_i[:-1])
        p = 1 if classified >= 0 else 0
        predictions.append(p)

    return np.mean(predictions == data['label'])


def AveragedPerceptron(data, r, T):
    w = [0]*y
    a = [0]*y
    for _ in range(1, T+1):
        shuffled = data.sample(frac=1)
        for x_i in shuffled.itertuples(index=False):
            y_i = 1 if x_i[-1] == 1 else -1
            val = Transpose(w, x_i[:-1])*y_i
            if val <= 0:
                val = VectorScale(x_i[:-1], y_i*r)
                w = VectorAdd(w, val)
            a = VectorAdd(a, w)
    return a


def Report_AveragedPerceptron(training_data_filepath, test_data_filepath):
    # Get and set the global variables
    training_data = pd.read_csv(training_data_filepath)
    training_data.columns = columns
    d_1 = [1]*training_data.shape[0]
    training_data.insert(0, "bias", d_1)

    test_data = pd.read_csv(test_data_filepath)
    test_data.columns = columns
    d_1 = [1]*test_data.shape[0]
    test_data.insert(0, "bias", d_1)

    print("Perceptrion: Averaged")
    r = .25
    T = 10
    a = AveragedPerceptron(training_data, r, T)

    print("Learned Weight Vector:")
    print(a)
    print("Learing Rate r:")
    print(r)
    print("predictions of Test Data")
    print(Averaged_Predict(a, test_data))
# endregion

# region Standard


def Standard_Predict(w, data):
    predictions = []
    for x_i in data.itertuples(index=False):
        classified = Transpose(w, x_i[:-1])
        p = 1 if classified >= 0 else 0
        predictions.append(p)

    return np.mean(predictions == data['label'])


def StandardPerceptron(data, r, T):
    w = [0]*y
    for _ in range(1, T+1):
        shuffled = data.sample(frac=1)
        for x_i in shuffled.itertuples(index=False):
            y_i = 1 if x_i[-1] == 1 else -1
            val = Transpose(w, x_i[:-1])*y_i
            if val <= 0:
                val = VectorScale(x_i[:-1], y_i*r)
                w = VectorAdd(w, val)
    return w


def Report_StandardPerceptron(training_data_filepath, test_data_filepath):
    # Get and set the global variables
    training_data = pd.read_csv(training_data_filepath)
    training_data.columns = columns
    d_1 = [1]*training_data.shape[0]
    training_data.insert(0, "bias", d_1)

    test_data = pd.read_csv(test_data_filepath)
    test_data.columns = columns
    d_1 = [1]*test_data.shape[0]
    test_data.insert(0, "bias", d_1)

    print("Perceptrion: Standard")
    r = .25
    T = 10
    w = StandardPerceptron(training_data, r, T)

    print("Learned Weight Vector:")
    print(w)
    print("Learing Rate r:")
    print(r)
    print("predictions of Test Data")
    print(Standard_Predict(w, test_data))
# endregion

# region Voted


def Voted_Predict(w, c, data):
    predictions = []
    for x_i in data.itertuples(index=False):
        sum = 0
        for i in range(len(w)):
            p = 1 if Transpose(w[i], x_i[:-1]) >= 0 else -1
            sum += p*c[i]

        p = 1 if sum >= 0 else 0
        predictions.append(p)

    return np.mean(predictions == data['label'])


def VotedPerceptron(data, r, T):
    w = []
    w0 = [0]*y
    w.append(w0)
    c = [0]
    m = 0
    n = 0

    for _ in range(1, T+1):
        for x_i in data.itertuples(index=False):
            y_i = 1 if x_i[-1] == 1 else -1
            val = Transpose(w[m], x_i[:-1])*y_i
            if val <= 0:
                val = VectorScale(x_i[:-1], y_i*r)
                w.append(VectorAdd(w[m], val))
                c.append(n)
                m = m+1
                n = 1
            else:
                n = n+1
    return w, c


def Report_VotedPerceptron(training_data_filepath, test_data_filepath):
    # Get and set the global variables
    training_data = pd.read_csv(training_data_filepath)
    training_data.columns = columns
    d_1 = [1]*training_data.shape[0]
    training_data.insert(0, "bias", d_1)

    test_data = pd.read_csv(test_data_filepath)
    test_data.columns = columns
    d_1 = [1]*test_data.shape[0]
    test_data.insert(0, "bias", d_1)

    print("Perceptrion: Voted")
    r = .25
    T = 10
    w, c = VotedPerceptron(training_data, r, T)

    print("w, c")
    for i in range(len(w)):
        if i % 10 == 0:
            print(f" {c[i]}: {printA(w[i])}")
    print("Learing Rate r:")
    print(r)
    print("predictions of Test Data")
    print(Voted_Predict(w, c, test_data))
# endregion


def printA(array):
    a = []
    for i in array:
        a.append("{0:.2f}".format(i))
    return a


def main():

    if len(sys.argv) == 2:
        if sys.argv[1] == '7':
            Report_StandardPerceptron(
                "Perceptron/bank-note/train.csv", "Perceptron/bank-note/test.csv")
        elif sys.argv[1] == '8':
            Report_VotedPerceptron(
                "Perceptron/bank-note/train.csv", "Perceptron/bank-note/test.csv")
        elif sys.argv[1] == '9':
            Report_AveragedPerceptron(
                "Perceptron/bank-note/train.csv", "Perceptron/bank-note/test.csv")
        else:
            print("Invalid argument given.")


main()
