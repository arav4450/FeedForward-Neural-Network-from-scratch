
import numpy as np
from layer import Linear_Layer
from activations import Softmax


class Model:
  
    def __init__(self, input_dim, num_classes, num_layers, num_neurons, activation):

      self.layers = []
      
      i = 0
      while(i<num_layers):
        if(i == 0):
          layer = Linear_Layer(input_dim, num_neurons, activation)
        else:
          layer = Linear_Layer(num_neurons, num_neurons, activation)
        self.layers.append(layer)
        i+= 1

      layer = Linear_Layer(num_neurons,num_classes, Softmax())
      self.layers.append(layer)
     
    def forward(self,x):
      
      value = 0
      j = 0
      while( j < len(self.layers)):

        curr_layer = self.layers[j]
        if(j == 0):
          value = curr_layer.forward(x)
        else:
          prev_value = value
          value = curr_layer.forward(prev_value)
        
        j+= 1

      return value