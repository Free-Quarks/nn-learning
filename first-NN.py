# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 20:54:01 2021

@author: xjust
"""

# This is my making of a simple backpropogation algorithm with stocastic
# gradient descent for the training and a simple quadratic cost fucntion with 
# the goal of reproducing a 2D function being the input 

# # first off we impor numpy 
# In[1]
from numpy import array, zeros, exp, random, dot, shape, reshape, meshgrid, transpose, linspace, maximum

import matplotlib.pyplot as plt # for plotting
import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display

# I am still not great with the matplotlib items but am learning 

# # Now we need to initialize some variables and tables for implementing the
# # higher level functions for the network. 

NumLayers = 2 # This is the number of layers besides the input layer
LayerSizes = [2,200,1] # input, 1st hidden, 2nd hid, 3rd hid, and output
# I am picking different sizes and more than needed just to get more practice

# Now to initialize the Weights and baises for the network. Need to be careful with relu since 
# if any entire vector for df is 0 it can derail the training, hence only random positive to start

Weights = [random.uniform(low=0, high=0.1, size=[LayerSizes[j], LayerSizes[j+1]]) for j in range(NumLayers)]
# The heavy lifting is done by the size option here, getting several matrix dimensions to pool together into an
# array. This results in a vector of arrays. 

Biases = [random.uniform(low= 0, high= +0.1, size=[LayerSizes[j+1]]) for j in range(NumLayers)]
# indexing starts at +1 since we don't add baises to input. This results in a vector of vectors. 

# # # Now to initialize variables for SGD as well # # #

y_layer = [zeros(LayerSizes[j]) for j in range(NumLayers+1)] # need to record first and last layers too
df_layer = [zeros(LayerSizes[j+1]) for j in range(NumLayers)] # no derivatives of nonlinear functions for input
dw_layer = [zeros([LayerSizes[j], LayerSizes[j+1]]) for j in range(NumLayers)] # This is for the cost function
# it has the same size as the Weights array since each Weight needs to be adjusted 
db_layer = [zeros(LayerSizes[j+1]) for j in range(NumLayers)] # no biases in input to train

# # Now to define the batchsize and initialize the corresponding inputs and outputs
# # define the number of batches we will run and the learning rate for our SGD as well

batchsize = 250

y_in = random.uniform(low = 0, high = +0.1, size = [batchsize,LayerSizes[0]]) 
y_out = random.uniform(low = 0, high = +0.1, size = [batchsize,LayerSizes[-1]]) # note: the -1 index means last

eta = 0.001 # this is the learning rate
batches = 5000 # This is the number of batches we will use to train of the data

costs=zeros(batches) # array to store the costs

# # # Now to setup the functions for the various parts of the algorithm # # #

# this fucntion should taking a neuron value and apply our nonlinear function
# we are using for this network. I will be using a sigmoid for this example. 

# In[2]

#def net_f_df(z): 
#   val = 1/(1 + exp(-z)) # This is the sigmoid nonlinear function
#    return(val, exp(-z)*val**2) # This returned the value and also the
                                # derivative which we will need for the SGD
                                
# For a change: Set up rectified linear units (relu) 
# instead of sigmoid
def net_f_df(z): # calculate f(z) and f'(z)
    
    return(maximum(0,z),1*(z>0)) # return both f and f'


# now we want to have a function to move the network forward one step

def forward_step(y,w,b):
    z = dot(y,w) + b # note: we have it y,w so adding since we adding a
    return(net_f_df(z)) # vector which needs to have same dimension as the
                        # side of the opertion (I think)

# now let's define the function to apply the network 

def apply_net(y_in, Weights, Biases, NumLayers, y_layer, df_layer):
    
    y = y_in 
    y_layer[0] = y
    for j in range(NumLayers):
        y,df = forward_step(y,Weights[j],Biases[j])
        df_layer[j] = df # this makes df[0] the first hidden layer differentials
        y_layer[j+1] = y # just recording values for later use in SGD
    return(y, y_layer, df_layer)


# now let's define the backward step used in training with SGD

def backward_step(delta, w, df):
    
    return(dot(delta,transpose(w))*df)


# now to define one of the more complicated functions. The full back propogation
# needed in order to define the cost function for each weights and then implement
# the training

def backprop(y_target, y_layer, df_layer, Weights, Biases, NumLayers, dw_layer, db_layer, batchsize):
    
    delta = (y_layer[-1] - y_target)*df_layer[-1] # defining delta, related to the derivative of the cost function
    dw_layer[-1] = dot(transpose(y_layer[-2]),delta)/batchsize # setting up top of dc/dw
    db_layer[-1] = delta.sum(0)/batchsize # setting up top of dc/db
    
    for j in range(NumLayers-1): # NumLayers-1 since the top is already accounted for above
        delta = backward_step(delta,Weights[-1-j],df_layer[-2-j]) # issuing the recursion to get all derivatives
        dw_layer[-2-j] = dot(transpose(y_layer[-3-j]),delta) # recording results (Do I need to still divide by batchsize?)
        db_layer[-2-j] = delta.sum(0) / batchsize
    return(dw_layer, db_layer)
        
# now to define the function to update the weights and baises as SGD dictates

def gradient_step(eta, dw_layer, db_layer, Weights, Biases):
    
    for j in range(NumLayers):
        Weights[j] -= eta*dw_layer[j] # nicely can act and replace whole arrays in this fashion
        Biases[j] -= eta*db_layer[j]
    return(Weights, Biases)
    
# Now to define the function for the training of the network, biggest difference of not using global should 
# be here 

def train_net(y_in, y_target, eta, Weights, Biases, NumLayers, y_layer, df_layer, dw_layer, db_layer, batchsize):
    
    y_out_result,y_layer,df_layer = apply_net(y_in, Weights, Biases, NumLayers, y_layer, df_layer)
    dw_layer,db_layer = backprop(y_target, y_layer, df_layer, Weights, Biases, NumLayers, dw_layer, db_layer, batchsize)
    Weights,Biases = gradient_step(eta, dw_layer, db_layer, Weights, Biases)
    cost = ((y_target-y_out_result)**2).sum()/batchsize
    return(cost, y_layer, df_layer, dw_layer, db_layer, Weights, Biases, y_out_result)

    
# # Now to define the function which is our data in this case and also out funtion to generate batches for training


def myFunc(x0, x1):
    r2 = x0**2 + x1**2
    return(exp(-5*r2)*abs(x0+x1))


def make_batch(batchsize):
    
    inputs = random.uniform(low = -0.5, high = +0.5, size = [batchsize,2]) # this generates the x, y values for fucntion
    targets = zeros([batchsize, 1])  # initialize the target variable
    targets[:, 0] = myFunc(inputs[:, 0],inputs[:, 1]) # this generates the target value at the x, y points
    return(inputs, targets) # think of this as y_in and y_target for the other functions

# In[3] ## Debugging
#y_in,y_target = make_batch(batchsize)
#test1,test2,test3 = apply_net(y_in, Weights, Biases, NumLayers, y_layer, df_layer)
#test4,test5 = backprop(y_target, y_layer, df_layer, Weights, Biases, NumLayers, dw_layer, db_layer, batchsize)
#Weights,Biases = gradient_step(eta, dw_layer, db_layer, Weights, Biases)

# In[4]
# # finally we shoud be at the point where we run the network and train it

for k in range(batches):
    y_in,y_target = make_batch(batchsize)
    costs[k],y_layer,df_layer,dw_layer,db_layer,Weights,Biases,y_out_result = train_net(y_in, y_target, eta, Weights, Biases, NumLayers, y_layer, df_layer, dw_layer, db_layer, batchsize)
    
    
plt.plot(costs)
plt.title("Cost function during training")
plt.show()

# In[5]

# # Lastly is to figure out how to plot the result!

# import functions for updating display 
# (simple animation)
from IPython.display import clear_output
from time import sleep

eta1 = 0.005 # learning rate
nsteps = 100

xrange=linspace(-0.5,0.5,40)
X0,X1=meshgrid(xrange,xrange)
test_batchsize=shape(X0)[0]*shape(X0)[1]
testsample=zeros([test_batchsize,2])
testsample[:,0]=X0.flatten()
testsample[:,1]=X1.flatten()


costs=zeros(nsteps)
for j in range(nsteps):
    clear_output(wait=True)
    fig,ax = plt.subplots(ncols=3, nrows=1, figsize=(8, 4)) # prepare figure
    ax[1].axis('off') # no axes
    ax[2].axis('off')
    
    # the crucial lines:
    y_in,y_target = make_batch(batchsize) # random samples (points in 2D)
    costs[j],y_layer,df_layer,dw_layer,db_layer,Weights,Biases,y_out_result = train_net(y_in, y_target, eta1, Weights, Biases, NumLayers, y_layer, df_layer, dw_layer, db_layer, batchsize) # train network (one step, on this batch)
    testoutput,one,two=apply_net(testsample, Weights, Biases, NumLayers, y_layer, df_layer) # check the new network output in the plane
    
    img=ax[1].imshow(reshape(testoutput,shape(X0)),interpolation='nearest',origin='lower') # plot image
    ax[2].imshow(myFunc(X0,X1),interpolation='nearest',origin='lower')
    ax[0].plot(costs)
    
    ax[0].set_title("Cost during training")
    ax[0].set_xlabel("number of batches")
    ax[1].set_title("Current network prediction")
    ax[2].set_title("Target Result")
    plt.show()
    sleep(0.05)