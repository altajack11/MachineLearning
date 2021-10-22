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
esac