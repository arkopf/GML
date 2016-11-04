# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 10:36:16 2016

@author: Jonas Schneider
"""

from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from matplotlib.cbook import pts_to_midstep
  

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
    valX, valY = zip(*pts)   
    valXx = list()
    valXo = list()
    i = 0
    while i<len(valX):
        if valY[i]>0:
            valXx.append(valX[i])
        else:
            valXo.append(valX[i])
        i=i+1
    for point in valXo:
        plt.plot(point[0], point[1], 'ro')
    for point in valXx:
        plt.plot(point[0], point[1], 'bx')
    #plt.plot(p.weight)
    print("p: ")
    print(p.weight[0]/-p.weight[2])
    print(p.weight[1]/-p.weight[2])
    #print("func: ")
    #print(func)
    x = np.arange(-1,2,2)
    plt.plot(x, p.weight[1]/-p.weight[2]*x + p.weight[0]/-p.weight[2], '-')
    #plt.plot(x, p.weight[1]/-p.weight[2]*x + p.weight[0]/-p.weight[2], '.')
    plt.plot(x, b*x + a, '-g')
    
    #plt.plot(func, '-')
    plt.axis([-1, 1, -1, 1])
    plt.show()
    return

# The Perceptron with the Standard Perceptron Learning Algorithm
class Perceptron:
    # Initialize the weight vector with random values
    def __init__(self, dim):
        self.weight = np.random.rand(dim+1)*2-1

    # Calculate and return the class for the given input instance x
    # @param x The given input instance
    # @return The output value of the perceptron {-1,1}
    def classify(self, x):
        if x[0]*self.weight[1]+x[1]*self.weight[2]+self.weight[0]<0:
            return -1
        else:
            return 1
        
    # Perform a learning step for a given training datum with input values x
    # and output value y in {-1,1}
    # @param x The given input instance
    # @param y The desired output value
    # @return False if the perceptron did not produce the desired output value, i.e. the learning adaptation has been performed
    #         True if the perceptron already produced the correct output value, i.e. no adaptation has been performed
    def learn(self, x, y):
        #print(x)
        calcY = self.classify(x)
        if calcY==y:
            return True
        else:
            vecX = [1,x[0],x[1]]
            self.weight[0]=self.weight[0]+y*vecX[0]
            self.weight[1]=self.weight[1]+y*vecX[1]
            self.weight[2]=self.weight[2]+y*vecX[2]
            #print(self.weight)
            return False

    # Perform the complete perceptron learning algorithm on the dataset (x_i, y_i)
    # @param dataset The complete dataset given as a 2D list [inputvalues, outputvalues]
    # with inputvalues being a list of all input values which again are a list of coordinates for each dimension
    # and output values a list of all desired output values
    def learnDataset(self, dataset):
        optimal = False
        count=0
        while(optimal==False):
            optimal=True
            count=count+1
            for valX, valY in dataset:
                if self.learn(valX, valY)==False:
                    optimal=False
                    #print("False"+str(valY))
                #print("next")
        print(self.weight)
        return count

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
    numTrain = 10
    # Create the input values for the training data
    trainX = np.random.rand(numTrain, 2)*2-1
    # Create the output values for the training data
    trainY = []
    for el in trainX:
        trainY.append(targetFunc(el))
    # Learn on the whole dataset
    dataset = list(zip(trainX, trainY))
    print("a: "+str(a))
    print("b: "+str(b))
    print("Iterations: "+str(p.learnDataset(dataset)))
    print('Terminated')
    # Plot the resulting approximation, training data and target function
    plotPerceptron(p, dataset, targetFunc)