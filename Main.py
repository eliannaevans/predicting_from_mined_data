'''
Created on Feb 13, 2021

@author: elibean
'''
import neural_network.NeuralNetwork as N
import my_scrapy
import numpy as np

brain = N.NeuralNetwork(2, 10, 10, 1)

my_input = np.array([[0,0],[0,1],[1,0],[1,1]])
expectedOutput = np.array([ [0], [1], [1], [0] ])

#brain.fit(my_input, expectedOutput, 1000)

for row in my_input:
    print(row)
    print(brain.predict(row))