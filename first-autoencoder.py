# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 00:39:54 2021

@author: xjust
"""

# In[1]:


# keras: Sequential is the neural-network class, Dense is
# the standard network layer
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D, UpSampling2D
from tensorflow.keras import optimizers # to choose more advanced optimizers like 'adam'

import numpy as np

import matplotlib.pyplot as plt # for plotting
import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display

# for subplots within subplots:
from matplotlib import gridspec

# for nice inset colorbars: (approach changed from lecture 1 'Visualization' notebook)
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

# for updating display 
# (very simple animation)
from IPython.display import clear_output
from time import sleep

steps = 1000

batchsize = 30

M = 27 # resolution

vals = np.linspace(-1,1,M)

x,y = np.meshgrid(vals,vals)

y_in = np.zeros([batchsize,M,M,1])

costs = np.zeros(steps)

# In[2]: 

def my_generator(batchsize,x,y):
    R = np.random.uniform(size=batchsize)
    x0 = np.random.uniform(size=batchsize,low=-1,high=1)
    y0 = np.random.uniform(size=batchsize,low=-1,high=1)
    return( 1.0*((x[None,:,:]-x0[:,None,None])**2 + (y[None,:,:]-y0[:,None,None])**2 < R[:,None,None]**2) )

def get_layer_activation_extractor(network):
    return(Model(inputs=network.inputs,
                            outputs=[layer.output for layer in network.layers]))

def get_layer_activations(network, y_in):
    """
    Call this on some test images y_in, to get the intermediate 
    layer neuron values. These are returned in a list, with one
    entry for each layer (the entries are arrays).
    """
    extractor=get_layer_activation_extractor(network)
    layer_features = extractor(y_in)
    return(layer_features)

def print_layers(network, y_in):
    """
    Call this on some test images y_in, to get a print-out of
    the layer sizes. Shapes shown are (batchsize,pixels,pixels,channels).
    After a call to the visualization routine, y_target will contain
    the last set of training images, so you could feed those in here.
    """
    layer_features=get_layer_activations(network,y_in)
    for idx,feature in enumerate(layer_features):
        s=np.shape(feature)
        print("Layer "+str(idx)+": "+str(s[1]*s[2]*s[3])+" neurons / ", s)


# In[3]:

Net=Sequential()
# 3x3 kernel size, 10 channels in first hidden layer:
Net.add(Conv2D(6,5,input_shape=(None,None,1),
               activation="swish",padding='same'))
Net.add(AveragePooling2D(pool_size=(3,3),padding='same')) # down
Net.add(Conv2D(3,5,
               activation="swish",padding='same'))
Net.add(Conv2D(3,5,
               activation="swish",padding='same'))
Net.add(Conv2D(3,5,
               activation="swish",padding='same'))
Net.add(AveragePooling2D(pool_size=(3,3),padding='same')) # down
Net.add(Conv2D(1,3,
               activation="swish",padding='same'))
Net.add(Conv2D(3,3,
               activation="swish",padding='same'))
Net.add(UpSampling2D(size=(3,3))) # up
Net.add(Conv2D(3,5,
               activation="swish",padding='same'))
Net.add(Conv2D(3,5,
               activation="swish",padding='same'))
Net.add(UpSampling2D(size=(3,3))) # up
Net.add(Conv2D(3,5,
               activation="swish",padding='same'))
Net.add(Conv2D(3,5,
               activation="swish",padding='same'))
Net.add(Conv2D(1,3,activation="linear",padding='same'))
Net.compile(loss='mean_squared_error',
              optimizer='adam')

Net.summary()

# In[4]:


for j in range(steps):
        # produce samples:
        y_in[:,:,:,0] = my_generator(batchsize,x,y)
        y_target = np.copy(y_in) # autoencoder wants to reproduce its input!
        
        # do one training step on this batch of samples:
        costs[j] = Net.train_on_batch(y_in,y_target)
        
        
        
# In[5]:        

print("Final cost value (averaged over last 50 batches): ", np.average(costs[-50:]))  

print_layers(Net,y_target)
    
plt.plot(costs)
plt.title("Cost function during training")
plt.show()       

        

