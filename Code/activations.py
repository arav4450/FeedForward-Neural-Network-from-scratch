
import numpy as np

# Softmax computation

class Softmax:

  def forward(self,x):

      rows, columns = x.shape
      output = np.zeros((rows, 10))
      i = 0
      while(i<rows):
        vector = x[i]
        try:
          e = np.exp(vector)
        except OverflowError as oe:
          print("After overflow", oe)
        output[i] = e / e.sum()
        i +=1

      return output

# Sigmoid activation

class Sigmoid:

    def __init__(self,):
        self.H = 1.0
        self.dH = 1.0

    def forward(self,x):
        self.H = 1/(1 + np.exp(-x))
        return self.H

    def backward(self,x):
        sig_H = 1/(1 + np.exp(-x))
        self.dH = (1 - sig_H ) * sig_H 
        return self.dH

# Relu activation

class Relu:

    def __init__(self,):
        self.H = 1.0
        self.dH = 1.0

    def forward(self,x):
        self.H = np.maximum(0, x)
        return self.H

    def backward(self,x):
        #self.dH = np.where(x<=0,0.01*x,1)
        self.dH = np.where(x<=0,0,1)
        return self.dH
      
# Tanh activation

class Tanh:

    def __init__(self,):
        self.H = 1.0
        self.dH = 1.0

    def forward(self,x):
        self.H = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return self.H

    def backward(self,x):
        sig_H = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        self.dH = 1 - sig_H  ** 2
        return self.dH