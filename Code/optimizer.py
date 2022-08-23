
import numpy as np


# Minibatch GD

class MinibatchGD:

  def compute(self,layer,dW,db,lr,epoch):
    #print(np.max(dW))
    #print(np.min(dW))

    update_W = lr * dW
    update_b = lr * db

    return update_W, update_b


# Stochastic GD

class SGD:

  def compute(self,layer,dW,db,lr,epoch):

    update_W = lr * dW
    update_b = lr * db

    return update_W, update_b


# Momentum GD

class Momentum:

  def __init__(self,):

    self.gamma = .9

  def compute(self,layer,dW,db,lr,epoch):

    update_w_prev = layer.update_W
    update_b_prev = layer.update_b

    update_w = self.gamma * update_w_prev + lr * dW
    update_b = self.gamma * update_b_prev + lr * db


    layer.update_W = update_w
    layer.update_b = update_b


    return update_w, update_b 


# NAG GD
# https://stackoverflow.com/questions/50774683/how-is-nesterovs-accelerated-gradient-descent-implemented-in-tensorflow

class NAG:

  def __init__(self,):

    self.gamma = .9

  def compute(self,layer,dW,db,lr,epoch):

    update_w_prev = layer.update_W
    update_b_prev = layer.update_b

    update_w = self.gamma * update_w_prev + lr * dW
    update_b = self.gamma * update_b_prev + lr * db

    comp_w =  self.gamma * update_w_prev + update_w - self.gamma * update_w
    comp_b =  self.gamma * update_b_prev + update_b - self.gamma * update_b


    layer.update_W = update_w
    layer.update_b = update_b


    return comp_w, comp_b

# GD with RMS Propagation

class RMSProp:

  def __init__(self,):

    self.gamma = .9

  def compute(self,layer,dW,db,lr,epoch):

    update_w_prev = layer.update_W
    update_b_prev = layer.update_b

    update_w = self.gamma * update_w_prev + (1-self.gamma) * dW**2
    update_b = self.gamma * update_b_prev + (1-self.gamma) * db**2

    comp_w = (lr / (np.sqrt(update_w + 1e-08))) * dW
    comp_b = (lr / (np.sqrt(update_b + 1e-08))) * db


    layer.update_W = update_w
    layer.update_b = update_b


    return comp_w, comp_b

# GD with ADAM

class Adam:

  def __init__(self,):

    self.gamma1 = .9
    self.gamma2 = .999

  def compute(self,layer,dW,db,lr,epoch):
      
      mdw = layer.update_W
      mdb = layer.update_b

      vdw = layer.update_W_V
      vdb = layer.update_b_V

      

      mdw = self.gamma1 * mdw + (1 - self.gamma1) * dW
      mdb = self.gamma1 * mdb + (1 - self.gamma1) * db

      vdw = self.gamma2 * vdw + (1 - self.gamma2) * dW ** 2
      vdb = self.gamma2 * vdb + (1 - self.gamma2) * db ** 2

      mdw_corr = mdw / (1 - np.power(self.gamma1, epoch + 1))
      mdb_corr = mdb / (1 - np.power(self.gamma1, epoch + 1))

      vdw_corr = vdw / (1 - np.power(self.gamma2, epoch + 1))
      vdb_corr = vdb / (1 - np.power(self.gamma2, epoch + 1))

      comp_W = (lr / (np.sqrt(vdw_corr + 1e-08))) * mdw_corr
      comp_b = (lr / (np.sqrt(vdb_corr + 1e-08))) * mdb_corr
    
      
      layer.update_W = mdw
      layer.update_b = mdb

      layer.update_W_V = vdw
      layer.update_b_V = vdb


      return comp_W, comp_b
