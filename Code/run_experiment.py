# import 
import numpy as np
import tensorflow as tf
import importlib
import argparse
from trainer import Trainer
from model import Model
from exp_setup import NN_exp_setup

#load data

def load_and_preprocess_data(normalize = False, debug = True):

  fashion_mnist = tf.keras.datasets.fashion_mnist

  (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

  if debug:
        print('Before preprocessing:')
        print(' - X_train.shape = {}, y_train.shape = {}'.format(train_images.shape, train_labels.shape))
        print(' - X_test.shape = {}, y_test.shape = {}'.format(test_images.shape, test_labels.shape))


  # randomly sort train_images/train_labels
  indexes = np.arange(train_images.shape[0])
  for _ in range(5): indexes = np.random.permutation(indexes)  # shuffle 5 times!
  train_images = train_images[indexes]
  train_labels = train_labels[indexes]
    
  # 'split' into cross-val & train sets (use 6000(10%) records in cross-val set)    
  val_count = 6000
  val_images = train_images[:val_count]
  val_labels = train_labels[:val_count]
  train_images = train_images[val_count:]
  train_labels = train_labels[val_count:]

  if debug:
        print('After preprocessing:')
        print(' - X_train.shape = {}, y_train.shape = {}'.format(train_images.shape, train_labels.shape))
        print(' - X_val.shape = {}, y_val.shape = {}'.format(val_images.shape, val_labels.shape))
        print(' - X_test.shape = {}, y_test.shape = {}'.format(test_images.shape, test_labels.shape))
        


  if(normalize == True):

    train_images = train_images / 255.0
    val_images = val_images / 255.0
    test_images = test_images / 255.0


  return train_images, train_labels, val_images, val_labels, test_images, test_labels


def create_and_run_experiment(args):

    train_images, train_labels, val_images, val_labels, test_images, test_labels = load_and_preprocess_data(True,True)

    epochs = args.epoch
    batch_size = args.batch_size
    learning_rate = args.lr
    num_layers = args.nlayer
    num_neurons = args.size_layer

    act_class_ = getattr(importlib.import_module('activations'),args.act_fn)
    activation = act_class_()


    opt_class_ = getattr(importlib.import_module('optimizer'),args.optimizer)
    optimizer = opt_class_()
    
   
    

    X_train = train_images
    y_train = train_labels
    X_train = X_train.reshape(train_images.shape[0],784)

    X_val = val_images
    y_val = val_labels
    X_val = X_val.reshape(val_images.shape[0],784)

    input_dim = X_train.shape[1]
    num_classes  = 10
    

    model = Model(input_dim, num_classes, num_layers, num_neurons, activation)
    NN_train = NN_exp_setup(model,learning_rate,optimizer)

    trainer  = Trainer(batch_size,epochs)


    train_loss, train_acc, val_loss, val_acc = trainer.fit(X_train,y_train,NN_train, validation_data = (X_val,y_val) )



def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch",type=int, help=" Provide number of epochs.", default = 10)
    parser.add_argument("--batch_size",type=int, help=" Provide batch size.", default = 32)
    parser.add_argument("--lr",type=float, help=" Provide learning rate", default = .001)
    parser.add_argument("--nlayer",type=int, help=" Provide number of layers.", default = 1)
    parser.add_argument("--size_layer",type=int, help=" Provide size of layers.", default = 3)
    parser.add_argument("--act_fn", type=str, help=" Provide activation function.", default= "Relu")
    parser.add_argument("--optimizer", type=str, help=" Provide activation function.", default = "MinibatchGD")
    

    args = parser.parse_args()
    return args


def main():
    """Get arguments and set up experiment."""

    args = _parse_args()
   
    create_and_run_experiment(args)

if __name__ == "__main__":
    main()