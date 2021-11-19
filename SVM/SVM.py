import numpy as np
import pandas as pd
from scipy.optimize import minimize
import sys

# Declare global variables that get set for the data being processed.
columns = ['variance',
           'skewness',
           'curtosis',
           'entropy',
           "label"]
l = 5

cutoff = 0.000001

# region Primal

def SVM_Model(training_data, r0, a, T, C, N, learning_rate_schedule):
    w = np.array([0]*l)
    r = r0
    for epoch in range(1, T+1):
        training_data_shuffled = training_data.sample(frac=1, random_state=1)
        for x_i in training_data_shuffled.values:
            y_i = 1 if x_i[l] == 1 else -1

            # We can do this with the gradient
            gradient = np.copy(w)
            gradient[0] = 0
            if (1 - (y_i * np.dot(w.T, x_i[:-1])) > 0):
                gradient = gradient - ((C * N * y_i) * x_i[:-1])

            w = w - (r * gradient)

        r = learning_rate_schedule(r0, a, epoch)

    return w


def SVM_Predict(data, w):
    predictions = []
    for x_i in data.values:
        p = 1 if np.dot(w.T, x_i[:-1]) >= 0 else 0
        predictions.append(p)

    return np.mean(predictions == data['label'])


def PrimalSVM_GenerateReport(training_data_filepath, test_data_filepath):

    training_data = pd.read_csv(training_data_filepath, header=None)
    training_data.columns = columns
    d_1 = [1]*training_data.shape[0]
    training_data.insert(0, "bias", d_1)

    test_data = pd.read_csv(test_data_filepath, header=None)
    test_data.columns = columns
    d_1 = [1]*test_data.shape[0]
    test_data.insert(0, "bias", d_1)

    r0 = 0.0001
    a = 0.5
    T = 100
    C = [0.1145, 0.5727, 0.8018]
    N = training_data.shape[0]

    def learning_rate_schedule_1(r0, a, t): return (
        r0) / (1 + ((r0 / a) * t))
    def learning_rate_schedule_2(r0, a, t): return (
        (r0) / (1 + t))

    print("PRIMAL SVM")
    print("Learning Rate Schedule: gamma_0 / 1 + (gamma_0 / a)t")
    print(f"Initial Gamma: {r0} a: {a}")
    ModelPredictPrint_ForEach_C(training_data, test_data, r0, a,
                                T, C, N, learning_rate_schedule_1)
    print()
    print(f"Learning Rate Schedule: gamma_0 / 1 + t")
    print(f"Initial Gamma: {r0}")
    ModelPredictPrint_ForEach_C(training_data, test_data, r0, a,
                                T, C, N, learning_rate_schedule_2)
    print()

    return


def ModelPredictPrint_ForEach_C(training_data, test_data, r0, a, T, C, N, learning_rate_schedule):
    for c in C:
        w = SVM_Model(
            training_data, r0, a, T, c, N, learning_rate_schedule)
        print(f"C: {c}")
        print(f"Learned W: {np.around(w, 4)}")
        print(
            f"Train Predictions: {'{0:.4f}'.format(SVM_Predict(training_data, w))}")
        print(
            f"Test Predictions: {'{0:.4f}'.format(SVM_Predict(test_data, w))}")
        print()


# endregion

# region Dual

def DualSVM_Model(data, C):
    x = data.drop(columns=['label']).values
    y = data['label'].values

    def objective_function(a):
        sum = 0
        for i in range(0, data.shape[0]):
            sum += np.sum(np.dot(x[i], x.T) * (y[i] * y) * (a[i] * a))
        return (0.5*sum) - a.sum()

    a0 = np.random.uniform(low=0, high=C, size=data.shape[0])
    constraints = {'type': 'eq', 'fun': lambda alpha: np.sum(
        alpha*y), 'jac': lambda _: y}
    bounds = [(0, C)] * data.shape[0]
    optimal_lagrangian_multipliers = minimize(objective_function, a0, method='SLSQP',
                                              bounds=bounds, constraints=constraints).x
    s = optimal_lagrangian_multipliers.shape[0]

    w = np.zeros(x.shape[1])
    for i in range(s):
        w += optimal_lagrangian_multipliers[i] * y[i] * x[i]

    b = sum([y[i] - np.dot(w.T, x[i]) for i in range(s)]) / s

    return w, b


def DualSVM_Predict(data, w, b):
    predictions = []
    for x_i in data.drop(columns=['label']).values:
        predictions.append(np.sign(np.dot(w.T, x_i) + b))

    return np.mean(predictions == data['label'].values)


def DualSVM_GenerateReport(training_data_filepath, test_data_filepath):

    training_data = pd.read_csv(training_data_filepath, header=None)
    training_data.columns = columns
    training_data.loc[training_data['label'] == 0] = -1

    test_data = pd.read_csv(test_data_filepath, header=None)
    test_data.columns = columns
    test_data.loc[test_data['label'] == 0] = -1

    C = [0.1145, 0.5727, 0.8018]

    print("Dual SVM")
    for c in C:
        w, b = DualSVM_Model(
            training_data, c)
        print(f"C: {c}")
        print()
        print(f"Learned w: {np.around(w,4)}")
        print()
        print(f"Learned b: {'{0:.4f}'.format(b)}")
        print()
        print(
            f"Train Predictions: {'{0:.4f}'.format(DualSVM_Predict(training_data, w, b))}")
        print()
        print(
            f"Test Predictions: {'{0:.4f}'.format(DualSVM_Predict(test_data, w, b))}")
        print()
        print()

    return


# endregion

# region DualKernel

def kernel(x_i, x_j, gamma):
    x = x_i - x_j
    return np.exp(-np.sum(np.square(x)) / gamma)


def DualKernelSVM_Model(data, C, gamma):
    x = data.drop(columns=['label']).values
    y = data['label'].values

    kern_matrix = np.zeros((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            kern_matrix[i][j] = kernel(x[i], x[j], gamma)

    def objective_function(a):
        sum = 0
        for i in range(data.shape[0]):
            all_y = y[i] * y
            all_a = a[i] * a
            all_x = np.dot(kern_matrix[i], kern_matrix.T)
            sum += np.sum(all_x * all_y * all_a)

        return (0.5*sum) - a.sum()

    a0 = np.random.uniform(low=0, high=C, size=data.shape[0])
    constraints = {'type': 'eq', 'fun': lambda alpha: np.sum(
        alpha*y), 'jac': lambda _: y}
    bounds = [(0, C)] * data.shape[0]

    optimal_lagrangian_multipliers = minimize(objective_function, a0, method='SLSQP',
                                              bounds=bounds, constraints=constraints).x

    return optimal_lagrangian_multipliers, kern_matrix


def DualKernelSVM_Predict(data, a, k):
    x = data.drop(columns=['label']).values
    y = data['label'].values

    predictions = []
    for i in range(x.shape[0]):
        prediction = 0
        for j in range(a.shape[0]):
            if a[j] >= cutoff:
                prediction += a[j] * y[i] * k[i][j]
        predictions.append(np.sign(prediction))
    return np.mean(predictions == y)


def DualKernelSVM_GenerateReport(training_data_filepath, test_data_filepath):

    training_data = pd.read_csv(training_data_filepath, header=None)
    training_data.columns = columns
    training_data.loc[training_data['label'] == 0] = -1

    test_data = pd.read_csv(test_data_filepath, header=None)
    test_data.columns = columns
    test_data.loc[test_data['label'] == 0] = -1

    C = [0.1145, 0.5727, 0.8018]
    gammas = [0.1, 0.5, 1, 5, 100]

    print("Dual Kernel SVM")
    for c in C:
        _a = []
        for g in gammas:
            a, k = DualKernelSVM_Model(
                training_data, c, g)
            count = 0
            if g != 0.1:
                for i in range(len(a)):
                    if(a[i] > cutoff and _a[i] > cutoff):
                        count += 1
            print(
                f"C: {c}, gamma: {g}, support vectors: {len([i for i in a if i > cutoff])}, overlapping support vectors: {count}, Train Predictions: {'{0:.4f}'.format(DualKernelSVM_Predict(training_data, a, k))}, Test Predictions: {'{0:.4f}'.format(DualKernelSVM_Predict(test_data, a, k))}")
            print()
            _a = a.copy()
        print()

    return

# endregion


def main():
    if len(sys.argv) == 2:
        if sys.argv[1] == '10':
            PrimalSVM_GenerateReport(
                "SVM/bank-note/train.csv", "SVM/bank-note/test.csv")
        elif sys.argv[1] == '11':
            DualSVM_GenerateReport(
                "SVM/bank-note/train.csv", "SVM/bank-note/test.csv")
        elif sys.argv[1] == '12':
            DualKernelSVM_GenerateReport(
                "SVM/bank-note/train.csv", "SVM/bank-note/test.csv")
        else:
            print("Invalid argument given.")


main()
