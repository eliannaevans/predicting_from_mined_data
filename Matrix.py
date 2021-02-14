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
    data = np.zeros(1)
    numRows = 0
    numCols = 0


    def __init__(self, rows, cols, data=np.zeros(1)):
        '''
        Constructor
        '''
        self.numRows = rows
        self.numCols = cols
        if data.all() == 0:
            self.data = np.array((self.numRows, self.numCols))
        
            for spot in np.nditer(self.data):
                spot = np.random.randint(-1, 1)
        else:
            self.data = data
            
    def add(self, num):
        data = np.array((self.numRows, self.numCols))
        for spot in np.nditer(self.data): spot += num
        return Matrix(self.numRows, self.numCols, data)
        
    def addMatrix(self, m):
        for spot, entry in zip(np.nditer(self.data), np.nditer(m.data)):
            spot += entry
    
    def subtract(self, a, b):
        m = a
        for spot, entry, place in zip(np.nditer(m.data), np.nditer(a.data), np.nditer(b.data)):
            spot = entry - place
            
        return m
    
    def transpose(self, m):
        return m.data.transpose()
    
    def dot(self, a, b):
        m = Matrix(1, b.numCols)
        
        row = 0
        for column in b.data:
            sum = 0
            for spot, entry in zip(a.data.transpose()[row], column):
                sum += spot*entry
            m[0,row] = sum
            
        return m
    
    def multiplyMatrix(self, a, b):
        m = Matrix(a.numRows, a.numCols)
        for spot, entry, place in zip(np.nditer(m.data), np.nditer(a.data), np.nditer(b.data)):
            spot = entry*place
            
        return m
            
    def multiply(self, num):
        for spot in np.nditer(self.data):
            spot *= num
            
    def sigmoid(self):
        for spot in np.nditer(self.data):
            spot = 1/(1 + math.exp(-spot))
            
    def disigmoid(self):
        m = Matrix(self.numRows, self.numCols)
        for spot, entry in zip(np.nditer(m.data), np.nditer(self.data)):
            spot = entry*(1 - entry)
            
        return m
    
    def toArray(self):
        return self.data