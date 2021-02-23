
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

NAME = "ResNet_intel-4x6-{}".format(int(time.time())) # block_per_dim x first channel size

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
nb_channels = [6, 12] # number of starting channels/filters
weight_decay = 1e-4 # decay rate for weights
nb_classes = 6 # number of classes
activation_func = 'swish' # what the activation function is for the whole net
blocks_per_dim = 4

# In[4] This is defining the residual block function

def res_block(x, nb_channels, activation_func,weight_decay): 
    #renet block where dimension doesnot change.
    #The skip connection is just simple identity conncection
    #we will have 3 blocks and then input will be added

    x_skip = x # this will be used for addition with the residual block 
    f1, f2 = nb_channels

    #first block 
    x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation(activation_func)(x)

    #second block # bottleneck (but size kept same with padding)
    x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation(activation_func)(x)

    # third block activation used after adding the input
    x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)

    # add the input 
    x = Add()([x, x_skip])
    x = Activation(activation_func)(x)

    return x,f2
# In[5] This is the net 

# this starts off the net, we wil reduce dimension off the bat
x = Conv2D(nb_channels[1], (3,3), activation=activation_func, padding='valid', strides=2,
                      use_bias=False, kernel_regularizer=l2(weight_decay))(img_input)

# This will run blocks_per_dim number of residual blocks 
for i in range(blocks_per_dim):
    # this runs the residual block which maintains the dimension, we will run it 
    x, last_filter = res_block(x, nb_channels, activation_func, weight_decay)


# setup for increasing the number of channels for next blocks, always doubling (for now) 
nb_channels = [last_filter, 2*last_filter]
# now to reduce the dimension before running another residual block
x = Conv2D(nb_channels[1], (3,3), activation=activation_func, padding='valid',strides=2,
                      use_bias=False, kernel_regularizer=l2(weight_decay))(x)

# This will run blocks_per_dim number of residual blocks 
for i in range(blocks_per_dim):
    # this runs the residual block which maintains the dimension, we will run it 
    x, last_filter = res_block(x, nb_channels, activation_func, weight_decay)

# setup for increasing the number of channels for next blocks, always doubling (for now) 
nb_channels = [last_filter, 2*last_filter]
# now to reduce the dimension before running another residual block
x = Conv2D(nb_channels[1], (3,3), activation=activation_func, padding='valid',strides=2,
                      use_bias=False, kernel_regularizer=l2(weight_decay))(x)

# This will run blocks_per_dim number of residual blocks 
for i in range(blocks_per_dim):
    # this runs the residual block which maintains the dimension, we will run it 
    x, last_filter = res_block(x, nb_channels, activation_func, weight_decay)

# setup for increasing the number of channels for next blocks, always doubling (for now) 
nb_channels = [last_filter, 2*last_filter]
# now to reduce the dimension before running another residual block
x = Conv2D(nb_channels[1], (3,3), activation=activation_func, padding='valid',strides=2,
                      use_bias=False, kernel_regularizer=l2(weight_decay))(x)

# This will run blocks_per_dim number of residual blocks 
for i in range(blocks_per_dim):
    # this runs the residual block which maintains the dimension, we will run it 
    x, last_filter = res_block(x, nb_channels, activation_func, weight_decay)

# this ends the net
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
