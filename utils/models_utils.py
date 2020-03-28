# imports:

import keras

from keras.layers import Activation, Flatten, Dense
from keras.layers import Reshape, Dropout, Lambda, concatenate, Add
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization

from keras import initializers

import numpy as np

import matplotlib.pyplot as plt
from IPython.display import clear_output


# functions:


# updatable plot:

class PlotLosses(keras.callbacks.Callback):
    
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        
        plt.plot(self.x, np.log10(self.losses), label="train_loss")
        plt.plot(self.x, np.log10(self.val_losses), label="val_loss")
#         plt.plot(self.x, self.losses, label="loss")
#         plt.plot(self.x, self.val_losses, label="val_loss")
        plt.xlabel('epoch')
#         plt.ylabel('loss')
        plt.ylabel('loss (log-scale)')
        plt.legend()
        plt.show();


# models and model blocks:

def conv_block_2d(x, 
                  filters, kernel_size, strides=(1, 1), padding='same', kernel_initializer='glorot_uniform',
                  do_batch_norm=False,
                  activation=None, 
                  pool_size=None, 
                  dropout=None):
    
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer)(x)
    
    if do_batch_norm:
        x = BatchNormalization(axis=3)(x)
    if activation:
        x = Activation(activation)(x)
    if pool_size:
        x = MaxPooling2D(pool_size=pool_size, padding=padding)(x)
    if dropout:
        x = Dropout(dropout)(x)
    
    return x


def identity_res_block_2d(x, kernel_size, filters):
    
    # source: https://github.com/priya-dwivedi/Deep-Learning/blob/master/resnet_keras/Residual_Networks_yourself.ipynb
    
    """
    Implementation of an identity residual block
    
    Arguments:
    x -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    
    Returns:
    x -- output of the identity block, tensor of shape (m, n_H, n_W, n_C)
    """
    
    # shortcut - identity:
    s = x
    
    # main #1 - 1x1 conv, BN, activation:
    x = conv_block_2d(x, 
                      filters[0], kernel_size=(1, 1), padding='same', 
                      kernel_initializer=initializers.random_normal(stddev=0.01), # optional
                      do_batch_norm=True,
                      activation='relu')
    
    # main #2 - spatial conv, BN, activation:
    x = conv_block_2d(x, 
                      filters[1], kernel_size=kernel_size, padding='same', 
                      kernel_initializer=initializers.random_normal(stddev=0.01), # optional
                      do_batch_norm=True,
                      activation='relu')
    
    # main #3 - 1x1 conv, BN:
    x = conv_block_2d(x, 
                      filters[2], kernel_size=(1, 1), padding='same', 
                      kernel_initializer=initializers.random_normal(stddev=0.01), # optional
                      do_batch_norm=True)

    # concatenate:
    x = Add()([x, s])
    x = Activation('relu')(x)
    
    return x


def conv_res_block_2d(x, kernel_size, filters, strides):
    
    # source: https://github.com/priya-dwivedi/Deep-Learning/blob/master/resnet_keras/Residual_Networks_yourself.ipynb
    
    """
    Implementation of a convolution residual block
    
    Arguments:
    x -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    
    Returns:
    x -- output of the convolution block, tensor of shape (m, n_H, n_W, n_C)
    """
    
    # shortcut - spatial conv, strides, BN:
    s = conv_block_2d(x, 
                      filters[2], kernel_size=(1, 1), padding='same', strides=strides,
                      do_batch_norm=True)
    
    # main #1 - 1x1 conv, strides, BN, activation:
    x = conv_block_2d(x, 
                      filters[0], kernel_size=(1, 1), padding='same', strides=strides, 
                      kernel_initializer=initializers.random_normal(stddev=0.01), # optional
                      do_batch_norm=True,
                      activation='relu')
    
    # main #2 - spatial conv, BN, activation:
    x = conv_block_2d(x, 
                      filters[1], kernel_size=kernel_size, padding='same', 
                      kernel_initializer=initializers.random_normal(stddev=0.01), # optional
                      do_batch_norm=True,
                      activation='relu')
    
    # main #3 - 1x1 conv, BN, activation:
    x = conv_block_2d(x, 
                      filters[2], kernel_size=(1, 1), padding='same', 
                      kernel_initializer=initializers.random_normal(stddev=0.01), # optional
                      do_batch_norm=True)

    # concatenate:
    x = Add()([x, s])
    x = Activation('relu')(x)
    
    return x