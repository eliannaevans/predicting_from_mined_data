'''
Created on Feb 13, 2021

@author: elibean
'''
import numpy as np
import neural_network.Matrix as M

class NeuralNetwork():
    '''
    classdocs
    '''
    echoweights= foxtrotWeights= outputWeights = None
    echoBias= foxtrotBias= outputBias = None
    learnRate = .01

    def __init__(self, my_input, echo, foxtrot, output):
        '''
        Constructor
        '''
        
        self.echoWeights = M.Matrix(echo, my_input)
        self.foxtrotWeights = M.Matrix(foxtrot, echo)
        self.outputWeights = M.Matrix(output, foxtrot)
        
        self.echoBias = M.Matrix(echo, 1)
        self.foxtrotBias = M.Matrix(foxtrot, 1)
        self.outputBias = M.Matrix(output, 1)
        
    def predict(self, my_input):
        echo = M.Matrix.multiplyMatrix(self.echoWeights, my_input)
        echo = echo.add(self.echoBias)
        echo.sigmoid()
        
        foxtrot = M.Matrix.multiplyMatrix(echo, self.foxtrotWeights)
        foxtrot = foxtrot.add(self.foxtrotBias)
        foxtrot.sigmoid()
        
        output = M.Matrix.multiplyMatrix(self.outputWeights, foxtrot)
        output = output.add(self.outputBias)
        output.sigmoid()
        
        return output.toArray()
    
    def train(self, x, y):
        #forward propogate
        echo = M.Matrix.multiplyMatrix(M.Matrix, self.echoWeights, x)
        echo = echo.add(self.echoBias)
        echo.sigmoid()
        
        foxtrot = M.Matrix.multiplyMatrix(echo, self.foxtrotWeights)
        foxtrot = foxtrot.add(self.foxtrotBias)
        foxtrot.sigmoid()
        
        output = M.Matrix.multiplyMatrix(self.outputWeights, foxtrot)
        output = output.add(self.outputBias)
        output.sigmoid()
        
        target = M.Matrix(x.shape[0], x.shape[1], x)
        
        error = M.Matrix.subtract(target, output)
        
        gradient = output.disigmoid()
        gradient.dot(error)
        gradient.multiply(self.learnRate)
        
        foxtrotTrans = foxtrot.transpose()
        outputWeightsDelta = M.Matrix.multiplyMatrix(gradient, foxtrotTrans)
        
        self.outputWeights = self.outputWeights.add(outputWeightsDelta)
        self.outputBias = self.outputBias.add(gradient)
        
        outputWeightsTrans = self.outputWeights.transpose()
        foxtrotError = M.Matrix.multiplyMatrix(outputWeightsTrans, error)
        
        foxtrotGradient = foxtrot.disigmoid()
        foxtrotGradient.dot(foxtrotError)
        foxtrotGradient.multiply(self.learnRate)
        
        echoTrans = echo.transpose()
        foxtrotWeightsDelta = M.Matrix.multiplyMatrix(foxtrotGradient, echoTrans)
        
        self.foxtraotWeights = self.foxtrotWeights.add(foxtrotWeightsDelta)
        self.foxtrotBias = self.foxtrotBias.add(foxtrotGradient)
        
        foxtrotWeightsTrans = self.foxtrotWeights.transpose()
        echoError = M.Matrix.multiplyMatrix(foxtrotWeightsTrans, foxtrotError)
        
        echoGradient = echo.disigmoid()
        echoGradient.dot(echoError)
        echoGradient.multiply(self.learnRate)
        
        xTrans = x.transpose()
        echoWeightsDelta = M.Matrix.multiplyMatrix(echoGradient, xTrans)
        
        self.echoWeights = self.echoWeights.add(echoWeightsDelta)
        self.echoBias = self.echoBias.add(echoGradient)
        
    def fit(self, x, y, epochs):
        for num in range(epochs):
            sample = (int)(np.random.randint(0, 1) * x.shape[0])
            self.train(x[sample], y[sample])
        