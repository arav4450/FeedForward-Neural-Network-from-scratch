
import numpy as np

# class for setting up a neural network for training

class NN_exp_setup:

      def __init__(self,model,lr,optimizer):
            
          self.model = model
          self.learning_rate = lr
          self.optimizer = optimizer
        

      def train_step(self,batch):
          
          x,y = batch
          #batch_size = len(y)
          y_hat = self.model.forward(x)

         
          loss_step, acc, dloss = loss_and_acc(y, y_hat)
         
          
          return loss_step, acc, dloss

    
      def validation_step(self,batch):
          x,y = batch
          #batch_size = len(y)
          y_hat = self.model.forward(x)

         
          loss_step, acc, _ = loss_and_acc(y, y_hat)
         
          
          return loss_step, acc


      def test_step(self,batch):

          return



# loss and accuracy computation

def loss_and_acc(y, y_hat):
     
     
     N = len(y)
     i = 0
     doutput = np.zeros((N, 10))
     loss_total = 0
     acc = 0
     while(i<N):
       target = y[i]
       prediction_vec = y_hat[i]
       
       if(np.equal(np.argmax(prediction_vec),target)):
         acc += 1

       prediction = prediction_vec[target]
       loss = -np.log(prediction)
       dloss = prediction-1
       doutput[i] = prediction_vec
       doutput[i][target] = dloss
       loss_total += loss
       i += 1
     
    
     #print(loss_total / N)

     return  loss_total / N, acc / N , doutput
     
