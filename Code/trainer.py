
# class implementing training process along with back propagation

class Trainer:

  def __init__(self,batch_size, epochs):
     
     self.batch_size = batch_size
     self.epochs = epochs
     self.train_loss = {}
     self.train_acc = {}
     self.val_loss = {}
     self.val_acc = {}


  def fit(self,X,y,NN_train, validation_data = None):

    self.NN_train = NN_train
    

    if(type(NN_train.optimizer).__name__ == 'SGD'):
      self.batch_size = 1

    for epoch in range(self.epochs):
        
        self.train_loss[epoch] = []
        self.train_acc[epoch] = []
        self.val_loss[epoch] = []
        self.val_acc[epoch] = []

        batch_index = 0

        while (batch_index < X.shape[0]):
          start = batch_index
          stop = batch_index + self.batch_size
          if( batch_index + self.batch_size >= X.shape[0]):
            stop = X.shape[0]
          data_batch = X[start:stop,:]
          label_batch = y[start:stop]

          loss_step, acc_step, dloss = self.NN_train.train_step((data_batch,label_batch))
          self.train_loss[epoch].append(loss_step)
          self.train_acc[epoch].append(acc_step)
          
          
          dH = dloss
          for layer in reversed(self.NN_train.model.layers):

            dH,dW,db = layer.backprop(dH)

            dW, db = self.NN_train.optimizer.compute(layer,dW,db,self.NN_train.learning_rate,epoch)

            layer.updation(dW,db)
            
          
          batch_index += self.batch_size
        

        val_loss_step, val_acc_step = self.NN_train.validation_step(validation_data)
        self.val_loss[epoch].append(val_loss_step)
        self.val_acc[epoch].append(val_acc_step)
          
        
        tr_loss_epoch = sum(self.train_loss[epoch]) / len(self.train_loss[epoch])
        tr_acc_epoch = sum(self.train_acc[epoch]) / len(self.train_acc[epoch])

        print(f"Epoch: {epoch}   Loss: {tr_loss_epoch:.4f}   Accuracy: {tr_acc_epoch:.4f}  Val_Loss: {self.val_loss[epoch][0]:.4f}   Val_Accuracy: {self.val_acc[epoch][0]:.4f}")
       
        

    
    return self.train_loss, self.train_acc, self.val_loss, self.val_acc


  def test(self,):


    return
