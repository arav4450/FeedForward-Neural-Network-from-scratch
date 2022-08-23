import numpy as np


class Linear_Layer:

    def __init__(self, inputs, neurons, activation):
        np.random.seed(42)
        self.W = np.random.randn(neurons, inputs) *.01
        self.b = np.zeros((neurons, 1))
        self.activation = activation

        self.update_W = np.zeros((neurons, inputs))
        self.update_b = np.zeros((neurons, 1))

        self.update_W_V = np.zeros((neurons, inputs))
        self.update_b_V = np.zeros((neurons, 1))

       

    def forward(self, H_prev):
        self.H_prev = H_prev
        self.A = np.dot(self.W, self.H_prev.transpose()) + self.b
        self.A = self.A.transpose()
        self.H = self.activation.forward(self.A)
        return self.H

    def backprop(self, dH):
        if(type(self.activation).__name__ != 'Softmax'):
           dA = np.multiply(self.activation.backward(self.A), dH)
        else:
           dA = dH
        dW = 1/dA.shape[0] * np.dot(dA.transpose(),self.H_prev)
        db = 1/dA.shape[0] * np.sum(dA.transpose(), axis=1, keepdims=True)
        dH_prev = np.dot(dA,self.W)

        return dH_prev, dW, db

    def updation(self, dW, db):
        
        self.W = self.W -  dW
        self.b = self.b -  db
        