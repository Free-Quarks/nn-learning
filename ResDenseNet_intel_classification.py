# This code we are going to try to implement the Residual part of a Resnet onto the DenseNet at the 
# point of the tranisition layers, allowing us to "bypass" each dense block if needed to (hopefully) improve
# the learning rate and reduce the generalization error while training over a large number of epochs


# In[1] imports
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, AveragePooling2D, Flatten, Add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import time


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

NAME = "ResDenseNet_intel-4x6x0x2-{}".format(int(time.time())) # layers x growth_rate x dropout x bottleneck

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

# In[2] importing the data
# Get the training and testing data
train_datagen = ImageDataGenerator(rescale = 1./255, 
                                   shear_range = 0.2, 
                                   zoom_range = 0.2, 
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('Data/seg_train', 
                                                    target_size = (150, 150), 
                                                    batch_size = 18)
test_set = test_datagen.flow_from_directory('Data/seg_test',
                                                target_size = (150, 150),
                                                 batch_size = 18)


# In[3] Variables and hyperparameters

img_input = Input(shape=(150,150,3)) # image input shape
growth_rate = 6 # how the number of filters grows
nb_channels = 2*growth_rate # number of starting channels/filters
weight_decay = 1e-4 # decay rate for weights
dropout_rate = 0 # dropout rate
bottleneck_width = 2 # relates to size of bottleneck in filters
nb_classes = 6 # number of classes
compression_level = 1 # how much compression is used, 1=none
activation_func = 'swish' # what the activation function is for the whole net
nb_layers = 4 # number of densely connected layers in one dense block




# In[4] setting up functions for network

def dense_block(x, nb_layers, nb_channels, growth_rate, dropout_rate, bottleneck_width, weight_decay):
    # This is the dense block function which concats the different sub-blocks being the conv blocks
    x_list = [x]
    for i in range(nb_layers):
        cb = convolution_block(x, growth_rate, dropout_rate, bottleneck_width, weight_decay)
        x_list.append(cb)
        x = Concatenate(axis=-1)(x_list)
        nb_channels += growth_rate
    return x, nb_channels

def convolution_block(x, nb_channels, dropout_rate, bottleneck_width, weight_decay):
    # this 1 subblock which gets concat'd to other blocks like itself. includes bottleneck in filters too
    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation(activation_func)(x)
    x = Conv2D(nb_channels * bottleneck_width, (1, 1), padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(x)       
    x = Dropout(dropout_rate)(x)
    
    # Standard (BN-ReLU-Conv), debating if last convolution should have an activation function
    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation(activation_func)(x)
    x = Conv2D(nb_channels, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(x)       
    x = Dropout(dropout_rate)(x)
    
    return x


def transition_layer(x, nb_channels, dropout_rate, compression_level, weight_decay):
        # this is the transition later between dense blocks, it reduces the image size by half each run and can
        # compress the number of filters if need be 
    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation(activation_func)(x)
    x = Conv2D(int(nb_channels*compression_level), (1, 1), padding='same',
                      use_bias=False, kernel_regularizer=l2(weight_decay))(x)    
    x = Dropout(dropout_rate)(x)    
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x

# In[5] This is the net 

# this starts off the net, note we sent linear outputs into the dense blocks since they apply activation in them
x = Conv2D(nb_channels, (3,3), padding='same',strides=(1,1),
                      use_bias=False, kernel_regularizer=l2(weight_decay))(img_input)

# in order to get dimensional matches for the residual adding we are going to run an additional conv to get a bypass 
# path, each dense block spits out nb_channels + 4*growth_rate (32) for the new max channel numbers
x_bypass =  Conv2D(nb_channels+nb_layers*growth_rate, (1,1), padding='same',strides=(1,1),
                      use_bias=False, kernel_regularizer=l2(weight_decay))(x)
# this starts the series of dense blockd and transition layers
x,nb_channels = dense_block(x, nb_layers, nb_channels, growth_rate, dropout_rate, bottleneck_width, weight_decay)
# here is the residual bypass
x = Add()([x, x_bypass])
# transition layers take in a linear input
x = transition_layer(x, nb_channels, dropout_rate, compression_level, weight_decay)
# settin up new bypass of correct dimensions
x_bypass =  Conv2D(nb_channels+nb_layers*growth_rate, (1,1), padding='same',strides=(1,1),
                      use_bias=False, kernel_regularizer=l2(weight_decay))(x)
# repeat
x,nb_channels = dense_block(x, nb_layers, nb_channels, growth_rate, dropout_rate, bottleneck_width, weight_decay)

x = Add()([x, x_bypass])

x = transition_layer(x, nb_channels, dropout_rate, compression_level, weight_decay)

x_bypass =  Conv2D(nb_channels+nb_layers*growth_rate, (1,1), padding='same',strides=(1,1),
                      use_bias=False, kernel_regularizer=l2(weight_decay))(x)

x,nb_channels = dense_block(x, nb_layers, nb_channels, growth_rate, dropout_rate, bottleneck_width, weight_decay)

x = Add()([x, x_bypass])

x = transition_layer(x, nb_channels, dropout_rate, compression_level, weight_decay)

x_bypass =  Conv2D(nb_channels+nb_layers*growth_rate, (1,1), padding='same',strides=(1,1),
                      use_bias=False, kernel_regularizer=l2(weight_decay))(x)

x,nb_channels = dense_block(x, nb_layers, nb_channels, growth_rate, dropout_rate, bottleneck_width, weight_decay)

x = Add()([x, x_bypass])

x = transition_layer(x, nb_channels, dropout_rate, compression_level, weight_decay)

#x_bypass =  Conv2D(nb_channels+nb_layers*growth_rate, (1,1), padding='same',strides=(1,1),
#                      use_bias=False, kernel_regularizer=l2(weight_decay))(x)
#
#x,nb_channels = dense_block(x, nb_layers, nb_channels, growth_rate, dropout_rate, bottleneck_width, weight_decay)
#
#x = Add()([x, x_bypass])
#
#x = transition_layer(x, nb_channels, dropout_rate, compression_level, weight_decay)
#
x_bypass =  Conv2D(nb_channels+nb_layers*growth_rate, (1,1), padding='same',strides=(1,1),
                      use_bias=False, kernel_regularizer=l2(weight_decay))(x)

x,nb_channels = dense_block(x, nb_layers, nb_channels, growth_rate, dropout_rate, bottleneck_width, weight_decay)
# adding an additional bypass here
x = Add()([x, x_bypass])
# this ends the net
x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
x = Activation(activation_func)(x)
x = GlobalAveragePooling2D()(x) # this "flattens" the images into an average value for each filter 
x = Dense(nb_classes, activation='softmax', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)

Net = Model(img_input, x)

Net.compile(optimizer= 'adam', loss= 'categorical_crossentropy', metrics= ['categorical_accuracy'])

Net.summary()

# In[6]: training of net

net_model = Net.fit_generator(training_set,
                        steps_per_epoch = 80,
                        epochs = 50,
                        validation_data = test_set,
                        validation_steps = 80,
                        callbacks=[tensorboard])

# opening tensorboard: conda activate tf_gpu && tensorboard --logdir=logs/
# %%
