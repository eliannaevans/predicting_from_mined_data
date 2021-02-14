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
            self.data = np.array((self.numRows, self.numCols))
        
            for spot in np.nditer(self.data):
                spot = np.random.uniform()*2 - 1
        else:
            self.data = data
            
    def add(self, num):
        s = np.zeros((self.numRows, self.numCols))
        for spot, entry in zip(np.nditer(s), np.nditer(self.data)):
            spot = entry + num
        return Matrix(self.numRows, self.numCols, s)
        
    def addMatrix(self, m):
        s = np.zeros((self.numRows, self.numCols))
        for spot, entry, place in zip(np.nditer(s.data), np.nditer(self.data), np.nditer(m.data)):
            spot = entry + place
        return Matrix(self.numRows, self.numCols, s)
    
    def subtract(self, a, b):
        s = np.zeros((a.numRows, a.numCols))
        for spot, entry, place in zip(np.nditer(s.data), np.nditer(a.data), np.nditer(b.data)):
            spot = entry - place
        return Matrix(a.numRows, a.numCols, s)
    
    def transpose(self):
        return Matrix(self.numRows, self.numCols, self.data.transpose())
    
    def dot(self, a, b):
        m = np.zeros((1, b.numCols))
        
        count = 0
        for column in b.data:
            sum = 0
            for spot, entry in zip(a.data.transpose(), column):
                sum += spot*entry
            m[0,count] = sum
            count += 1
            
        return Matrix(1, b.numCols, m)
    
    def multiplyMatrix(self, a, b):
        m = np.zeros((a.numRows, a.numCols))
        for spot, entry, place in zip(np.nditer(m.data), np.nditer(a.data), np.nditer(b.data)):
            spot = entry*place
            
        return Matrix(a.numRows, a.numCols, m)
            
    def multiply(self, num):
        m = np.zeros((self.numRows, self.numCols))
        for spot, entry in zip(np.nditer(m), np.nditer(self.data)):
            spot = entry*num
            
        return Matrix(self.numRows, self.numCols, m)
            
    def sigmoid(self):
        for spot in np.nditer(self.data):
            spot = 1/(1 + math.exp(-spot))
            
    def disigmoid(self):
        m = Matrix(self.numRows, self.numCols)
        for spot, entry in zip(np.nditer(m.data), np.nditer(self.data)):
            spot = entry*(1 - entry)
            
        return m