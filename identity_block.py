import numpy as np
from keras.layers import Conv2D, Activation, BatchNormalization, Add

def identity_block(activation_prev_layer, filter_size, filters_array, stage):
    
    f1, f2, f3 = filters_array
    
    
    
    # First Conv layer
    X = Conv2D(filters = f1, kernel_size = (1,1), padding = 'valid', strides=(1,1), kernel_initializer = glorot_uniform(seed=1))(activation_prev_layer)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    
    # Second Conv layer
    X = Conv2D(filters = f2, kernel_size = (filter_size,filter_size), padding = 'same', strides=(1,1), kernel_initializer = glorot_uniform(seed=1))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    
    # Third Conv layer    
    X = Conv2D(filters = f3, kernel_size = (1,1), padding = 'valid', strides=(1,1), kernel_initializer = glorot_uniform(seed=1))(X)
    X = BatchNormalization(axis = 3)(X)
    
    #Adding skip connection
    X = Add()([ X, activation_prev_layer])
    activation_third_layer = Activation("relu")(X)
    
   	return activation_third_layer