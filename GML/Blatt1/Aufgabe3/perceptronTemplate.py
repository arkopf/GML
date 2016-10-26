# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 10:36:16 2016

@author: Jonas Schneider
"""

from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
  

# Linear target function (f(x))
def func(x, a, b):
    if a+b*x[0] < x[1]:
        return -1
    else:
        return 1

# Plots a given 2-dimensional perceptron in a quadratic input space. The -1 and +1 regions are
# colored differently and if the dataset that was used for training is provided, the points for
# -1 are plotted as o and the points for +1 are plotted as x
#
# @param p The perceptron that is to be plotted
# @param pts The training data that is to be plotted. If left None, no data points will be plotted
# @param func The target function. If left None, it will not be plotted
# @param mini The minimum border of the input space
# @param maxi The maximum border of the input space
# @param res The plotting resolution, how many points should be used per dimension
def plotPerceptron(p, pts=None, func=None, mini=-1, maxi=1, res=500):
    return

# The Perceptron with the Standard Perceptron Learning Algorithm
class Perceptron:
    # Initialize the weight vector with random values
    def __init__(self, dim):
        return
    # Calculate and return the class for the given input instance x
    # @param x The given input instance
    # @return The output value of the perceptron {-1,1}
    def classify(self, x):
        return
        
    # Perform a learning step for a given training datum with input values x
    # and output value y in {-1,1}
    # @param x The given input instance
    # @param y The desired output value
    # @return False if the perceptron did not produce the desired output value, i.e. the learning adaptation has been performed
    #         True if the perceptron already produced the correct output value, i.e. no adaptation has been performed
    def learn(self, x, y):
        return

    # Perform the complete perceptron learning algorithm on the dataset (x_i, y_i)
    # @param dataset The complete dataset given as a 2D list [inputvalues, outputvalues]
    # with inputvalues being a list of all input values which again are a list of coordinates for each dimension
    # and output values a list of all desired output values
    def learnDataset(self, dataset):
        return

# Main Program        
if __name__ == "__main__":
    # Generate a 2D-Perceptron
    p = Perceptron(2)
    # Choose the target function's parameter randomly from [-1;1]
    a = np.random.rand()*2-1
    b = np.random.rand()*2-1
    # Create the target function with fixed a and b
    targetFunc = lambda x: func(x,a,b)
    # Set the number of training data
    numTrain = 1000
    # Create the input values for the training data
    trainX = np.random.rand(numTrain, 2)*2-1
    # Create the output values for the training data
    trainY = []
    for el in trainX:
        trainY.append(targetFunc(el))
    # Learn on the whole dataset
    dataset = list(zip(trainX, trainY))
    p.learnDataset(dataset)
    print('Terminated')
    # Plot the resulting approximation, training data and target function
    plotPerceptron(p, dataset, targetFunc)