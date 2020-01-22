# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 11:43:19 2018

@author: User
"""

import math
import numpy as np
import _pickle as cPickle
import gzip




class mlp:
    
    def __init__(self, inputs, targets, n_hidden):
        
        self.inputNodes = len(np.transpose(inputs))
        self.hiddenNodes = n_hidden
        self.outputNodes = len(np.transpose(targets))
        self.nData = len(inputs)
        self.beta = 1
        
        self.weights1 = np.random.rand(self.inputNodes+1, self.hiddenNodes)-0.5*(2/np.sqrt(self.inputNodes))
        self.weights2 = np.random.rand(self.hiddenNodes+1, self.ouputNodes)-0.5*(2/np.sqrt(self.inputNodes))
    
    def forprop(self, inputs):
        # Dot inputs with weight matrix and apply sigmoid Activation function
        self.hiddenSum = np.dot(inputs, self.weights1)
        self.hiddenActivation = 1 / (1 + math.exp(-self.hiddenSum))
        self.hiddenActivation = np.concatenate((self.hiddenActivation, np.ones(self.nData, 1)), axis =1)
        
        # Dot hidden output values with weight matrix
        self.outputSum = np.dot(self.hiddenActivation, self.weights2)
        self.outputActivation = 1 / (1 + math.exp(-self.outputSum))
        self.outputActivation = np.concatenate((self.outputActivation, np.ones(self.nData, 1)), axis =1)
        # Activations -> Outputs

    def mlptrain(self, inputs, targets, eta, alpha):
        self.outputs = forprop(inputs)
        inputs = np.concatenate(inputs, np.ones((self.nData, 1)), axis = 1)
        testInputs = np.concatenate(inputs, np.ones((self.nData, 1)), axis = 1)
        
        weightUpdate1 = np.empty((self.inputNodes+1, self.hiddenNodes))
        weightUpdate2 = np.empty((self.hiddenNodes+1, self.ouputNodes))
        

               
f = gzip.open('mnist.pkl.gz','rb')
tset, vset, teset = cPickle.load(f, encoding='latin1')
f.close()
len(tset[0][0])

test_in = tset[0][0:1:1]
train_target = np.zeros((nTrain,10))

test_target = vset
nTrain = 
for i in range(nTrain):
    train_target[i,tset[1][i]] = 1

nTest = 
for i in range(nTest):
    test_target[i, vset[1][i]] = 1
tset[0][1]

test_in.shape
nn =mlp()