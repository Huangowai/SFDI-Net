# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import regularizers
 
class MyRegularizer(regularizers.Regularizer):

    def __init__(self, strength):
        self.strength = strength

    def __call__(self, x):
        # batch_size, c, h, w = x.size()
        return ( tf.reduce_sum(self.strength[0] * tf.image.total_variation(x[:,:,:,0]))+
                  tf.reduce_sum(self.strength[1] * tf.image.total_variation(x[:,:,:,1]))+
                  tf.reduce_sum(self.strength[2] * tf.image.total_variation(x[:,:,:,2]))+
                  tf.reduce_sum(self.strength[3] * tf.image.total_variation(x[:,:,:,3]))+
                  tf.reduce_sum(self.strength[4] * tf.image.total_variation(x[:,:,:,4]))+
                  tf.reduce_sum(self.strength[5] * tf.image.total_variation(x[:,:,:,5]))+
                   tf.reduce_sum(self.strength[6] * tf.image.total_variation(x[:,:,:,6]))  
                )
    def get_config(self):
        return {'strength': self.strength}


loss="MeanSquaredError"
activation="tanh"
img_size = (None, None)
num_classes = 7
batch_size = 32
channel =6  
conv_s = 1    

filters1 = [2*channel, 4*channel]
filters2 = 8*channel
filters3 = [4*channel, 2*channel]
flag1 = [0,0,0,0];
flag2 = [0,0];
flag3 = [0,0,0,0];

x_max, x_min = [[0.231, 0.172, 0.123, 0.114, 0.102, 0.0853]], [[0.0622, 0.0229, 0.00922, 0.01207, 0.00795, 0.00354]]
y_max, y_min = [[0.025, 1., 1.5, 2.5, 1.5, 1.5, 0.2]], [[0.0001, 0.3, 0.4, 0.4, -1.5, 0., 0.05]]
x_max = np.array(x_max, dtype=np.float32)
x_min = np.array(x_min, dtype=np.float32)
y_max = np.array(y_max, dtype=np.float32)
y_min = np.array(y_min, dtype=np.float32)
 
weight = np.array([0.0000001,0.00000025,0.00000005,0.00000001,0.0000001,0.00000001,0.000002], dtype=np.float32)

tf.random.set_seed(15)    #15
def get_model(img_size, num_classes,weight):
    inputs = keras.Input(shape=img_size + (6,))
    X = inputs
    for i, filters in enumerate(filters1):
        X = layers.Conv2D(filters, conv_s, strides=1, padding="same",activation=activation)(X)
        if flag1[2*i] ==1 : X = layers.Dropout(0.5)(X)
        X = layers.Conv2D(filters, conv_s, strides=1, padding="same",activation=activation)(X)
        locals()["X"+str(i)] = X
        if flag1[2*i+1] ==1 : X= layers.Dropout(0.5)(X)
        X = layers.MaxPooling2D(2, strides=2, padding="valid")(X)
        
    X = layers.Conv2D(filters2,conv_s, strides=1, padding="same",activation=activation)(X)
    if flag2[0] ==1 : X = layers.Dropout(0.5)(X)
    X = layers.Conv2D(filters2,conv_s, strides=1, padding="same",activation=activation)(X)
    if flag2[1] ==1 : X = layers.Dropout(0.5)(X)
    
    for j, filters in enumerate(filters3):
        X = layers.UpSampling2D(2)(X)
        X = layers.concatenate([X,locals()["X"+str(i)]], axis = 3)
        X = layers.Conv2D(filters, conv_s, strides=1, padding="same",activation=activation)(X)
        if flag3[2*j] ==1 : X = layers.Dropout(0.5)(X)
        X = layers.Conv2D(filters, conv_s, strides=1, padding="same",activation=activation)(X)
        if flag3[2*j+1] ==1 : X= layers.Dropout(0.5)(X)
        i -= 1
        
    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 1, padding="same", activity_regularizer=MyRegularizer(weight), 
                            activation=activation)(X) 

    # Define the model
    model = keras.Model(inputs, outputs)
    return model

# Build model
model = get_model(img_size, num_classes, weight)
model.load_weights('my_model_SFDInet.h5')
