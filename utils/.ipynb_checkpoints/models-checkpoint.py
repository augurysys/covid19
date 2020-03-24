# in this file:
# 'raw-wave to classification' models: EnvNet, EnvNetv2, WaveNet
# 'raw_spec to classification' models: spec_cnn_1, spec_cnn_2
# 'raw-wave to raw-wave' models: wave_AE, (WaveNetAE - TODO)
# 'raw-spec to raw-spec' models: spec_AE

# imports:

import os
import numpy as np
import keras
import tensorflow as tf

from keras.layers import Activation
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Conv1D, MaxPooling1D, Convolution1D, Lambda, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense
from keras.layers.core import Permute

import numpy as np
import pandas as pd
import sys

import tensorflow as tf
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Activation, Dropout, Add, TimeDistributed, Multiply 
from keras.layers import Conv1D, Conv2D, MaxPooling1D, AveragePooling1D
from keras.models import Model, Sequential, load_model
from keras import backend as K
from keras import metrics
from keras import optimizers
from keras.callbacks import History, ModelCheckpoint

import keras
import tensorflow as tf
from keras.models import Sequential, Model, load_model, save_model
from keras.layers import Input, Dense, TimeDistributed, LSTM, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.optimizers import SGD, Adam

from os.path import isfile
from tensorflow.python.client import device_lib
import h5py
from keras import backend as K

from keras import Input, metrics
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.engine import Model
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D, MaxPooling2D, UpSampling2D, K
from keras.optimizers import Adadelta, SGD

os.environ['KERAS_BACKEND'] = 'tensorflow'
keras.backend.clear_session()
np.random.seed(1)
tf.set_random_seed(1)

# models:

##############################################################################################

def EnvNet(x_shape, num_classes):
    # model: raw-wave to classification
    # x_shape: (input_length, ), with input_length = n_t
    
    filters_raw = 40
    poolsize_raw = 160
    new_dim = 149

    inp = keras.engine.Input(shape=x_shape, name='input')
    inp2 = keras.layers.core.RepeatVector(1)(inp)
    inp2 = Permute((2, 1))(inp2)

    # conv1
    x = Convolution1D(filters=filters_raw, kernel_size=(8), strides=1, padding='valid', name='conv1')(inp2)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # conv2
    x = Convolution1D(filters=filters_raw, kernel_size=8, strides=1, padding='valid', name='conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # maxpooling + swap
    print(x.shape)
    x = MaxPooling1D(pool_size=(poolsize_raw))(x)
    print(x.shape)
    x = Permute((2, 1))(x)
    print(x.shape)
#     new_shape = (filters_raw, int(x.shape[1]/poolsize_raw), 1)
    new_shape = (filters_raw, new_dim, 1)
    print(new_shape)
    x = Reshape(new_shape)(x)
    # model.add(Reshape((model.output_shape[1], model.output_shape[2], 1)))

    # conv3
    x = Convolution2D(filters=50, kernel_size=(8,13), strides=1, name='conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # maxpooling
    x = MaxPooling2D(pool_size=(3,3))(x)

    # covn4
    x = Convolution2D(filters=50, kernel_size=(1,5), strides=1, name='conv4')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # maxpooling
    x = MaxPooling2D(pool_size=(1,3))(x)

    # fc5
    x = Flatten()(x)
    # x = Dense(4096, activation='relu',  name='fc5')(x)
    # x =  Dropout(0.5)(x)

    # fc6
    # x = Dense(4096, activation='relu',  name='fc6')(x)
    # x = Dropout(0.5)(x)

    # fc7
    x = Dense(num_classes,  name='fc7')(x)#, activation='relu')(x)

    # to categorical, softmax
    x = Activation(tf.nn.softmax)(x)

    return keras.engine.Model(input=inp, output=x)

##############################################################################################

def EnvNetv2(x_shape, num_classes):
    # model: raw-wave to classification
    # x_shape: (input_length, ), with input_length = n_t

    inp = keras.engine.Input(shape=x_shape, name='input')
    inp2 = keras.layers.core.RepeatVector(1)(inp)
    inp2 = Permute((2, 1))(inp2)

    # conv1
    x = Convolution1D(filters=32, kernel_size=64, strides=2, padding='valid', name='conv1', kernel_initializer='he_normal')(inp2)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # conv2
    x = Convolution1D(filters=64, kernel_size=16, strides=2, padding='valid', name='conv2', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # maxpooling + swap
    poolsize = 64
    x = MaxPooling1D(pool_size=poolsize)(x)
    reshape_size = x.get_shape().as_list()[1]
    x = Permute((2, 1))(x)
    x = Reshape((64, reshape_size, 1))(x)

    # conv3
    x = Convolution2D(filters=32, kernel_size=(8, 8), strides=1, name='conv3', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # conv4
    x = Convolution2D(filters=32, kernel_size=(8, 8), strides=1, name='conv4', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # maxpooling
    x = MaxPooling2D(pool_size=(5, 3))(x)

    # covn5
    x = Convolution2D(filters=64, kernel_size=(1, 4), strides=1, name='conv5', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # covn6
    x = Convolution2D(filters=64, kernel_size=(1, 4), strides=1, name='conv6', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # maxpooling
    x = MaxPooling2D(pool_size=(1, 2))(x)

    # covn7
    x = Convolution2D(filters=128, kernel_size=(1, 2), strides=1, name='conv7', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # covn8
    x = Convolution2D(filters=128, kernel_size=(1, 2), strides=1, name='conv8', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # maxpooling
    x = MaxPooling2D(pool_size=(1, 2))(x)

    # covn9
    x = Convolution2D(filters=256, kernel_size=(1, 2), strides=1, name='conv9', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # covn10
    x = Convolution2D(filters=256, kernel_size=(1, 2), strides=1, name='conv10', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # maxpooling
    x = MaxPooling2D(pool_size=(1, 2))(x)

    # fc11
    x = Flatten()(x)
    # x = Dense(4096, activation='relu', name='fc11')(x)
    # x = Dropout(0.5)(x)
    #
    # # fc12
    # x = Dense(4096, activation='relu', name='fc12')(x)
    # x = Dropout(0.5)(x)

    # fc13
    x = Dense(num_classes, name='fc_classes')(x)  # , activation='relu')(x)

    # to categorical, softmax
    x = Activation(tf.nn.softmax)(x)

    return keras.engine.Model(input=inp, output=x)

##############################################################################################

def WaveNet(x_shape, num_classes):
    # model: raw-wave to classification
    # x_shape: (input_length, ), with input_length = n_t
    wnc = WaveNetClassifier(x_shape, (num_classes,), kernel_size=2, dilation_depth=9, n_filters=40)
    model = wnc.get_model()
    return model

class WaveNetClassifier():

    def __init__(self, input_shape, output_shape, kernel_size=2, dilation_depth=9, n_filters=40):
        self.activation = 'softmax'
        self.scale_ratio = 1
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.kernel_size = kernel_size
        self.dilation_depth = dilation_depth
        self.n_filters = n_filters
        self.model = self.construct_model()
        self.start_idx = 0
        self.history = None
        self.prev_history = None

    def residual_block(self, x, i):
        tanh_out = Conv1D(self.n_filters,
                          self.kernel_size,
                          dilation_rate=self.kernel_size ** i,
                          padding='causal',
                          name='dilated_conv_%d_tanh' % (self.kernel_size ** i), 
                          activation='tanh')(x)
        sigm_out = Conv1D(self.n_filters,
                          self.kernel_size,
                          dilation_rate=self.kernel_size ** i,
                          padding='causal',
                          name='dilated_conv_%d_sigm' % (self.kernel_size ** i), 
                          activation='sigmoid')(x)
        z = Multiply(name='gated_activation_%d' % (i))([tanh_out, sigm_out])
        skip = Conv1D(self.n_filters, 1, name='skip_%d' % (i))(z)
        res = Add(name='residual_block_%d' % (i))([skip, x])
        return res, skip

    def construct_model(self):
        x = Input(shape=self.input_shape, name='original_input')
        x_reshaped = Reshape(self.input_shape + (1,), name='reshaped_input')(x)
        skip_connections = []
        out = Conv1D(self.n_filters, 2, dilation_rate=1, padding='causal', name='dilated_conv_1')(x_reshaped)
        for i in range(1, self.dilation_depth + 1):
            out, skip = self.residual_block(out, i)
            skip_connections.append(skip)
        out = Add(name='skip_connections')(skip_connections)
        out = Activation('relu')(out)
        out = Conv1D(self.n_filters, 80, strides=1, padding='same', name='conv_5ms', activation='relu')(out)
        out = AveragePooling1D(80, padding='same', name='downsample_to_200Hz')(out)
        out = Conv1D(self.n_filters, 100, padding='same', activation='relu', name='conv_500ms')(out)
        out = Conv1D(self.output_shape[0], 100, padding='same', activation='relu', name='conv_500ms_target_shape')(out)
        out = AveragePooling1D(100, padding='same', name='downsample_to_2Hz')(out)
        out = Conv1D(self.output_shape[0], (int)(self.input_shape[0] / 8000), padding='same', name='final_conv')(out)
        out = AveragePooling1D((int)(self.input_shape[0] / 8000), name='final_pooling')(out)
        out = Reshape(self.output_shape)(out)
        out = Activation(self.activation)(out)
        model = Model(x, out)
        return model

    def get_model(self):
        return self.model

##############################################################################################

### from: https://github.com/drscotthawley/audio-classifier-keras-cnn

# original model used in reproducing Stein et al

def spec_cnn_1(x_shape, num_classes):  
    # model: raw-spec to classification
    
    n_layers = 5
#     n_layers = 4 
    
    K.set_image_data_format('channels_first')   # old model used channels_first, new one uses channels_last. see make_melgram utils in datautils.py

    nb_filters = 32  # number of convolutional filters to use
    pool_size = (2, 2)  # size of pooling area for max pooling
    kernel_size = (3, 3)  # convolution kernel size
    input_shape = (x_shape[0], x_shape[1], x_shape[2])

    model = Sequential()
    model.add(Conv2D(nb_filters, kernel_size, padding='valid', input_shape=input_shape))

    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))

    for layer in range(n_layers-1):
        model.add(Convolution2D(nb_filters, kernel_size))
        #model.add(BatchNormalization(axis=1))
        #model.add(ELU(alpha=1.0))
        model.add(Activation('elu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('elu'))
    #model.add(ELU(alpha=1.0))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))
    return model

##############################################################################################

### from: https://github.com/drscotthawley/audio-classifier-keras-cnn

# This is a VGG-style network that I made by 'dumbing down' @keunwoochoi's compact_cnn code
# I have not attempted much optimization, however it *is* fairly understandable

def spec_cnn_2(x_shape, num_classes):
    # model: raw-spec to classification
    
    # Inputs:
    #    x_shape = [ # audio channels, # spectrogram freq bins, # spectrogram time bins ]
    #    num_classes = number of output n_classes
    #    n_layers = number of conv-pooling sets in the CNN
    
    n_layers = 5
#     n_layers = 4 
    
    from keras import backend as K
#     K.set_image_data_format('channels_last')                   # SHH changed on 3/1/2018 b/c tensorflow prefers channels_last
    K.set_image_data_format('channels_first')
    
    nb_filters = 32  # number of convolutional filters = "feature maps"
    kernel_size = (3, 3)  # convolution kernel size
    pool_size = (2, 2)  # size of pooling area for max pooling
    cl_dropout = 0.5    # conv. layer dropout
    dl_dropout = 0.6    # dense layer dropout
    
    input_shape = (x_shape[0], x_shape[1], x_shape[2])
    
    model = Sequential()
    model.add(Conv2D(nb_filters, kernel_size, padding='valid', input_shape=input_shape, name="Input"))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))     # Leave this relu & BN here.  ELU is not good here (my experience)

    for layer in range(n_layers-1):   # add more layers than just the first
        model.add(Conv2D(nb_filters, kernel_size))
        #model.add(BatchNormalization(axis=1))  # ELU authors reccommend no BatchNorm. I confirm.
        model.add(Activation('elu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(cl_dropout))
    
    model.add(Flatten())
#     model.add(Dense(128))            # 128 is 'arbitrary' for now
    model.add(Dense(32))
    #model.add(Activation('relu'))   # relu (no BN) works ok here, however ELU works a bit better...
    model.add(Activation('elu'))
    model.add(Dropout(dl_dropout))
    model.add(Dense(num_classes))
    model.add(Activation("softmax",name="Output"))
    return model

##############################################################################################

def wave_AE(x_shape):
    # model: raw-wave to raw-wave
    
    inputs = Input(x_shape)
    
    # Encoder
    conv = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)  # 16384x16
    conv = MaxPooling1D(pool_size=2, padding='same')(conv)

    conv = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(conv)  # 8192x128
    conv = MaxPooling1D(pool_size=2, padding='same')(conv)

    conv = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(conv)  # 4096x256
    encoded = MaxPooling1D(pool_size=2, padding='same')(conv)
    
    # Decoder
    conv = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(encoded)  # 2048x64
    conv = UpSampling1D(size=2)(conv)

    conv = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(conv)  # 4096x32
    conv = UpSampling1D(size=2)(conv)

    conv = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(conv)  # 8192x16
    conv = UpSampling1D(size=2)(conv)

    outputs = Conv1D(filters=1, kernel_size=3, padding='same', activation='sigmoid')(conv)  # 16384x1
    
#     print(outputs.shape)
#     outputs = Reshape(x_shape)(outputs)
#     print(outputs.shape)

    model = Model(inputs, outputs)
    encoder = Model(inputs, encoded)
    
    return model, encoder

##############################################################################################

def spec_AE(x_shape):
    # model: raw-spec to raw-spec
    
    inputs = Input(x_shape)
    
    # Encoder
    conv = Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')(inputs)
    conv = MaxPooling2D(pool_size=(2,2), padding='same')(conv)

    conv = Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')(conv)
    conv = MaxPooling2D(pool_size=(2,2), padding='same')(conv)

    conv = Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu')(conv)
    encoded = MaxPooling2D(pool_size=(2,2), padding='same')(conv)
    
    # Decoder
    conv = Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu')(encoded)
    conv = UpSampling2D(size=(2,2))(conv)

    conv = Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')(conv)
    conv = UpSampling2D(size=(2,2))(conv)

    conv = Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')(conv)
    conv = UpSampling2D(size=(2,2))(conv)

    outputs = Conv2D(filters=1, kernel_size=(3,3), padding='same', activation='sigmoid')(conv)
    
#     print(outputs.shape)
#     outputs = Reshape(x_shape)(outputs)
#     print(outputs.shape)

    model = Model(inputs, outputs)
    encoder = Model(inputs, encoded)
    
    return model, encoder

##############################################################################################

