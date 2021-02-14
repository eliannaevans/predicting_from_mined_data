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
        echo = M.Matrix.multiplyMatrix(M.Matrix, self.echoWeights, my_input)
        echo = echo.addMatrix(self.echoBias)
        echo.sigmoid()
        
        print(echo.data)
        
        foxtrot = M.Matrix.multiplyMatrix(M.Matrix, echo, self.foxtrotWeights)
        foxtrot = foxtrot.addMatrix(self.foxtrotBias)
        foxtrot.sigmoid()
        
        print(foxtrot.data)
        
        output = M.Matrix.multiplyMatrix(M.Matrix, self.outputWeights, foxtrot)
        output = output.addMatrix(self.outputBias)
        output.sigmoid()
        
        return output.data
    
    def train(self, x, y):
        #forward propogate
        echo = M.Matrix.multiplyMatrix(M.Matrix, self.echoWeights, x)
        echo = echo.addMatrix(self.echoBias)
        echo.sigmoid()
        
        foxtrot = M.Matrix.multiplyMatrix(M.Matrix, echo, self.foxtrotWeights)
        foxtrot = foxtrot.addMatrix(self.foxtrotBias)
        foxtrot.sigmoid()
        
        output = M.Matrix.multiplyMatrix(M.Matrix, self.outputWeights, foxtrot)
        output = output.addMatrix(self.outputBias)
        output.sigmoid()
        
        target = M.Matrix(y.shape[0], 1, y)
        
        error = M.Matrix.subtract(M.Matrix, target, output)
        
        gradient = output.disigmoid()
        gradient = M.Matrix.dot(M.Matrix, gradient, error)
        gradient.multiply(self.learnRate)
        
        foxtrotTrans = foxtrot.transpose()
        outputWeightsDelta = M.Matrix.multiplyMatrix(M.Matrix, gradient, foxtrotTrans)
        
        self.outputWeights = self.outputWeights.addMatrix(outputWeightsDelta)
        self.outputBias = self.outputBias.addMatrix(gradient)
        
        outputWeightsTrans = self.outputWeights.transpose()
        foxtrotError = M.Matrix.multiplyMatrix(M.Matrix, outputWeightsTrans, error)
        
        foxtrotGradient = foxtrot.disigmoid()
        foxtrotGradient = M.Matrix.dot(M.Matrix, foxtrotGradient, foxtrotError)
        foxtrotGradient.multiply(self.learnRate)
        
        echoTrans = echo.transpose()
        foxtrotWeightsDelta = M.Matrix.multiplyMatrix(M.Matrix, foxtrotGradient, echoTrans)
        
        self.foxtraotWeights = self.foxtrotWeights.addMatrix(foxtrotWeightsDelta)
        self.foxtrotBias = self.foxtrotBias.addMatrix(foxtrotGradient)
        
        foxtrotWeightsTrans = self.foxtrotWeights.transpose()
        echoError = M.Matrix.multiplyMatrix(M.Matrix, foxtrotWeightsTrans, foxtrotError)
        
        echoGradient = echo.disigmoid()
        echoGradient = M.Matrix.dot(M.Matrix, echoGradient, echoError)
        echoGradient.multiply(self.learnRate)
        
        xTrans = x.transpose()
        echoWeightsDelta = M.Matrix.multiplyMatrix(M.Matrix, echoGradient, xTrans)
        
        self.echoWeights = self.echoWeights.addMatrix(echoWeightsDelta)
        self.echoBias = self.echoBias.addMatrix(echoGradient)
        
    def fit(self, x, y, epochs):
        for num in range(epochs):
            sample = (int)(np.random.randint(0, 1) * x.shape[0])
            self.train(x[sample], y[sample])
        