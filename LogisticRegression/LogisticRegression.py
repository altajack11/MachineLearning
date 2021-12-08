import numpy as np
import pandas as pd
import sys


def LogisticRegression_MAP(training_data, r0, a, epochs, var):
    w = np.array([0]*5)
    r = r0
    for t in range(1, epochs+1):
        training_data_shuffled = training_data.sample(frac=1, random_state=1)
        for x_i in training_data_shuffled.values:
            y_i = 1 if x_i[-1] == 1 else -1

            s = (w.T @ x_i[:-1]) * y_i
            gradient = -training_data.shape[0] * y_i * x_i[:-1] / (1 + np.exp(s)) + (w / var**2)
            w = w - (r * gradient)

        r = ( r0) / (1 + ((r0 / a) * t))
    return w
    
def LogisticRegression_ML(training_data, r0, a, epochs):
    w = np.array([0]*5)
    r = r0
    for t in range(1, epochs+1):
        training_data_shuffled = training_data.sample(frac=1, random_state=1)
        for x_i in training_data_shuffled.values:
            y_i = 1 if x_i[-1] == 1 else -1

            s = (w.T @ x_i[:-1]) * y_i
            gradient = -training_data.shape[0] * y_i * x_i[:-1] / (1 + np.exp(s))
            w = w - (r * gradient)

        r = ( r0) / (1 + ((r0 / a) * t))
    return w

def Predict(data, w):
    predictions = []
    for x_i in data.values:
        p = 1 if np.dot(w.T, x_i[:-1]) >= 0 else -1
        predictions.append(p)

    return np.mean(predictions == data['label'])



def LogisticRegression_GenerateReport(training_data_filepath, test_data_filepath):
    columns = ['variance', 'skewness', 'curtosis',  'entropy', "label"]
    # Get and set the global variables
    training_data = pd.read_csv(training_data_filepath, header=None)
    training_data.columns = columns
    d_1 = [1]*training_data.shape[0]
    training_data.insert(0, "bias", d_1)
    training_data.label.replace(0, -1, inplace=True)

    test_data = pd.read_csv(test_data_filepath, header=None)
    test_data.columns = columns
    d_1 = [1]*test_data.shape[0]
    test_data.insert(0, "bias", d_1)
    test_data.label.replace(0, -1, inplace=True)

    r0 = 0.01
    a = 0.1
    epochs = 100
    vars = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]

    print("Logistic Regression")
    print("MAP")
    for var in vars:
        w = LogisticRegression_MAP(training_data, r0, a, epochs, var)
        print(f"Variance: {var}, Train Predictions: {np.around(Predict(training_data, w), 4)}, Test Predictions: {np.around(Predict(test_data, w), 4)}") 
        print()
    print("ML")
    w = LogisticRegression_ML(training_data, r0, a, epochs)
    print(f"Train Predictions: {np.around(Predict(training_data, w), 4)}, Test Predictions: {np.around(Predict(test_data, w), 4)}") 



def main():
    if len(sys.argv) == 2:
        if sys.argv[1] == '15':
            LogisticRegression_GenerateReport("LogisticRegression/bank-note/train.csv",
                                  "LogisticRegression/bank-note/test.csv")
        else:
            print("Invalid argument given.")


main()
