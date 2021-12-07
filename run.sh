#!/bin/bash

arg="$1"

case "$arg" in
    0 ) python3 EnsembleLearning/AdaBoost.py 0;;
    1 ) python3 EnsembleLearning/BaggedTrees.py 1;;
    2 ) python3 EnsembleLearning/BaggedTrees.py 2;;
    3 ) python3 EnsembleLearning/RandomForests.py 3;;
    4 ) python3 EnsembleLearning/RandomForests.py 4;;
    5 ) python3 LinearRegression/LinearRegression.py 5;;
    6 ) python3 LinearRegression/LinearRegression.py 6;;
    7 ) python3 Perceptron/Perceptron.py 7;;
    8 ) python3 Perceptron/Perceptron.py 8;;
    9 ) python3 Perceptron/Perceptron.py 9;;
    10 ) python3 SVM/SVM.py 10;;
    11 ) python3 SVM/SVM.py 11;;
    12 ) python3 SVM/SVM.py 12;;
    13 ) python3 NeuralNetworks/NeuralNetworks.py 13;;
    14 ) python3 NeuralNetworks/NeuralNetworks.py 14;;
    15 ) python3 LogisticRegression/LogisticRegression.py 15;;
esac