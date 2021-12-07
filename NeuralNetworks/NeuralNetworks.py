from typing import OrderedDict
import numpy as np
import pandas as pd
import sys
from scipy.special import expit as sig
import torch
import torch.nn as nn

# region NN_Basic
def NeuralNetwork_GenerateReport(training_data_filepath, test_data_filepath):
    columns = ['variance', 'skewness', 'curtosis',  'entropy', "label"]

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

    widths = [5, 10, 25, 50, 100]
    epoch = 100
    gamma = 0.1
    learning_rate = 0.1
    d = 0.1
    
    print("Neural Network Stochastic Predictions")
    print("Edge weights initialized to random numbers")
    for width in widths: 
        nodes, w = NeuralNetwork(training_data, width, epoch, gamma, learning_rate, d, True)
        print(f"Width: {width}, Train Predictions: {Predict(training_data, nodes, w)}, Test Predictions: {Predict(test_data, nodes, w)}")
    print("Edge weights initialized to zeros")
    for width in widths: 
        nodes, w = NeuralNetwork(training_data, width, epoch, gamma, learning_rate, d)
        print(f"Width: {width}, Train Predictions: {Predict(training_data, nodes, w)}, Test Predictions: {Predict(test_data, nodes, w)}")


def Predict(data, nodes, w):
    predictions = []
    for x_i in data.values:
        ForwardPropagation(x_i, nodes, w)
        predictions.append(nodes[-1][0][0])
    return np.mean([1 if i >= 0 else -1 for i in predictions] == data['label'])


def NeuralNetwork(data, width, epoch, gamma, learning_rate, d, random = False):
    nodes = [np.ones([i, 1]) for i in [5, width, width, 1]]   
    w = [np.zeros([width - 1, 5]), np.zeros([width - 1, width]), np.zeros([1, width])]
    dw = [np.zeros([width - 1, 5]), np.zeros([width - 1, width]), np.zeros([1, width])]
    if random:
        w = [np.random.normal(0, 1, (width - 1, 5)), np.random.normal(0, 1, (width - 1, width)), np.random.normal(0, 1, (1, width))]

    for t in range(epoch):
        for x_i in data.sample(frac=1, random_state=1).values:
            ForwardPropagation(x_i, nodes, w)
            BackPropagation(x_i[-1].reshape([1,1]), width, nodes, w, dw)
            for i in range(3):
                w[i] = w[i] - learning_rate * dw[i]
        learning_rate = gamma / (1 + ((gamma/d) * t))
    return nodes, w


def ForwardPropagation(x, nodes, w):
    nodes[0] = x[:-1].reshape([5, 1])
    nodes[1][:-1,:] = sig((w[0] @ nodes[0].reshape([-1,1])))
    nodes[2][:-1,:] = sig((w[1] @ nodes[1].reshape([-1,1])))
    nodes[3] = w[2] @ nodes[2]


def BackPropagation(y, width, nodes, w, dw):
    dLdz = nodes[-1] - y
    dzdw =  np.tile(nodes[-2], [1, 1]).T
    dw[-1] = dLdz*dzdw
    dzdz = w[-1][:, :-1]
    dzdz, dLdz = sig_der(width, nodes, w, dw, dLdz, dzdz, 1)
    sig_der(width, nodes, w, dw, dLdz, dzdz, 0)


def sig_der(width, nodes, w, dw, dLdz, dzdz, layer):
    dsig = nodes[layer + 1][:-1] * ( 1 - nodes[layer + 1][:-1])
    dzdw = dsig * np.tile(nodes[layer], [1, width - 1]).T
    dLdz = dzdz.T @ dLdz
    dw[layer] = dLdz*dzdw
    dzdz = dsig * w[layer] 
    dzdz = dzdz[:, :-1]
    return dzdz, dLdz
# endregion

class NeuralNet(nn.Module):
    
    def __init__(self, layer_dimensions, epochs=1000, ReLU=True):
        super(NeuralNet, self).__init__()      
        self.epochs = epochs          

        self.weights_init = nn.init.xavier_normal_ if not ReLU else nn.init.kaiming_uniform_
        def init_weights(model):
            if isinstance(model, nn.Linear):
                self.weights_init(model.weight)
                torch.nn.init.ones_(model.bias)

        structure = OrderedDict([("1", nn.Linear(layer_dimensions[0], layer_dimensions[1]))])
        for i in range(2,len(layer_dimensions)):
            structure[str(2*i - 2)] = nn.Tanh() if not ReLU else nn.ReLU()
            structure[str(2*i - 1)] = nn.Linear(layer_dimensions[i-1], layer_dimensions[i])   

        self.model = nn.Sequential(structure)

        self.model.apply(init_weights)

    def forward(self, x):              
        out = torch.from_numpy(np.float32(x.copy()))
        out.requires_grad = True        
        return self.model(out)


    def predict(self, x, y):
        p = self.forward(x).detach().numpy()
        p = [[1] if i >= 0 else [-1] for i in p]
        return np.mean(p == y)
            

    def train(self, train_x, train_y):
        y = torch.from_numpy(np.float32(train_y.copy()))
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters())
        
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = criterion(self.forward(train_x), y)
            loss.backward()
            optimizer.step()

def PyTorch_GenerateReport(training_data_filepath, test_data_filepath):
    columns = ['variance', 'skewness', 'curtosis',  'entropy', "label"]

    training_data = pd.read_csv(training_data_filepath, header=None)
    training_data.columns = columns
    d_1 = [1]*training_data.shape[0]
    training_data.insert(0, "bias", d_1)
    training_data.label.replace(0, -1, inplace=True)
    train_y = training_data.label.values.reshape(-1, 1)
    train_x = training_data.drop(columns=['label'])

    test_data = pd.read_csv(test_data_filepath, header=None)
    test_data.columns = columns
    d_1 = [1]*test_data.shape[0]
    test_data.insert(0, "bias", d_1)
    test_data.label.replace(0, -1, inplace=True)
    test_y = test_data.label.values.reshape(-1, 1)
    test_x = test_data.drop(columns=['label'])

    afs= ["tahn", "RELU"]
    depths = [3,5,9]
    widths = [5, 10, 25, 50, 100]

    for af in afs:
        for depth in depths:
            for width in widths:               
                dimensions = [train_x.shape[1]]
                for _ in range(depth-2):
                    dimensions.append(width)
                dimensions.append(1)

                model = NeuralNet(dimensions, ReLU=(af=="RELU"))
                model.train(train_x.values.reshape((-1, train_x.shape[1])), train_y)

                print(f"{af}, Depth: {depth}, Width: {width}, train predictions: {model.predict(train_x, train_y).round(3)}, test predictions: {model.predict(test_x, test_y).round(3)}")

def main():
    if len(sys.argv) == 2:
        if sys.argv[1] == '13':
            NeuralNetwork_GenerateReport("NeuralNetworks/bank-note/train.csv", "NeuralNetworks/bank-note/test.csv")
        elif sys.argv[1] == '14':
            PyTorch_GenerateReport("NeuralNetworks/bank-note/train.csv", "NeuralNetworks/bank-note/test.csv")
        else:
            print("Invalid argument given.")


main()
