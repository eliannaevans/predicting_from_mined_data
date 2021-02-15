'''
Created on Feb 13, 2021

@author: elibean
'''
import numpy as np
from numpy import math

class Matrix():
    '''
    classdocs
    '''
    data = np.array(None) #placeholder
    numRows = 0
    numCols = 0


    def __init__(self, rows, cols, data=np.array(None)):
        '''
        Constructor
        '''
        
        self.numRows = rows
        self.numCols = cols
        if data.any() == None:
            self.data = np.zeros((self.numRows, self.numCols))
        
            for i in range(self.numRows):
                for j in range(self.numCols):
                    self.data[i,j] = np.random.uniform()*2 - 1
        else:
            self.data = data
            
    def add(self, num):
        for i in range(self.numRows):
            for j in range(self.numCols):
                self.data[i,j] += num
        
    def addMatrix(self, m):
        for i in range(m.numRows):
            for j in range(m.numCols):
                self.data[i,j] += m.data[i,j]
    
    def subtract(self, a, b):
        s = np.zeros((a.numRows, a.numCols))
        for i in range(a.numRows):
            for j in range(a.numCols):
                s[i,j] = a.data[i,j] - b.data[i,j]
        return Matrix(a.numRows, a.numCols, s)
    
    def transpose(self):
        return Matrix(self.numRows, self.numCols, self.data.T)
    
    def dot(self, a, b):
        m = np.zeros((a.numRows, b.numCols))
        
        for i in range(a.numRows):
            for j in range(b.numCols):
                dot_sum = 0
                print(a.data.shape)
                if b.numRows == 1:
                    for k in range(a.numCols):
                        dot_sum += a.data[i,k] * b.data[j]
                else:
                    for k in range(a.numCols):
                        dot_sum += a.data[i,k] * b.data[k,j]
                m[i,j] = dot_sum
        
        return Matrix(a.numRows, b.numCols, m)
    
    def multiplyMatrix(self, m):
        for i in range(m.numRows):
            for j in range(m.numCols):
                self.data[i,j] *= m.data[i,j]
            
    def multiply(self, num):
        for i in range(self.numRows):
            for j in range(self.numCols):
                self.data[i,j] *= num
            
    def sigmoid(self):
        for i in range(self.numRows):
            for j in range(self.numCols):
                self.data[i,j] = 1/(1 + math.exp(-self.data[i,j]))
            
    def disigmoid(self):
        m = Matrix(self.numRows, self.numCols)
        for i in range(self.numRows):
            for j in range(self.numCols):
                m.data[i,j] = self.data[i,j]*(1-self.data[i,j])
            
        return m