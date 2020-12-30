# This is a small bit from the first lecture of Machine Learning for Physicists
from numpy import *

N0 = 3 # input layer
N1 = 2 # output layer

w = random.uniform(low = -1, high = +1, size = (N1,N0)) #weight matrix
b = random.uniform(low = -1, high = +1, size = (N1)) #bias

y_in = array([0.2,0.4,-0.1]) # input vector / values

z = dot(w,y_in) + b # result value with weights

y_out = 1/(1+exp(-z)) # application of nonlinear function

print("y_in:", y_in)
print("w:", w)
print("b:", b)
print("z:", z)
print("y_out:", y_out)

# Now to apply multiple hidden layers and to plot neuron activation with a threshold of 0

def apply_layer(y_in,w,b): # This is the function to apply a layer to a neural network
    z = dot(w,y_in)+b
    return(1/(1+exp(-z)))

N0_1 = 2 # input layer size
N1_1 = 30 # hidden layer size
N2_1 = 1 # output layer size

# from input layer to hidden layer
w1 = random.uniform(low=-10,high=+10,size=(N1_1,N0_1))
b1 = random.uniform(low=-1,high=+1,size=N1_1)

# from hidden layer to output layer
w2 = random.uniform(low=-10,high=+10,size=(N2_1,N1_1))
b2 = random.uniform(low=-1,high=+1,size=N2_1)

def apply_net(y_in):
    global w1,b1,w2,b2 # I know using global is bad, just following video
    
    y1 = apply_layer(y_in,w1,b1)    
    y2 = apply_layer(y1,w2,b2)
    return(y2)

import matplotlib.pyplot as plt

M=50 # This sets the image size as MxM
y_out = zeros([M,M]) # defining an array to hold result

for j1 in range(M):
    for j2 in range(M):
        value0=float(j1)/M-0.5
        value1=float(j2)/M-0.5
        y_out[j1,j2]=apply_net([value0,value1])[0]
        
# display image
plt.imshow(y_out,origin='lower',extent=(-0.5,0.5,-0.5,0.5))
plt.colorbar()
plt.show()


# now to work on a network with more hidden layers and plotting to see the result
Nlayers = 20
LayerSize = 100

Weights = random.uniform(low = -3, high = +3, size=[Nlayers,LayerSize,LayerSize])
Biases =  random.uniform(low=-1,high=+1,size=[Nlayers,LayerSize])

# for first hidden layer from input
WeightFirst = random.uniform(low=-1,high=+1,size=[2,LayerSize])    
BiasFirst =  random.uniform(low=-1,high=+1,size=[LayerSize])

# for last hidden layer to output

WeightFinal = random.uniform(low=-1,high=+1,size=[LayerSize,1])    
BiasFinal =  random.uniform(low=-1,high=+1,size=1)
def apply_layer_new(y_in,w,b): # a function that applies a layer    
    z=dot(y_in,w)+b # note different order in matrix product!
    return(1/(1+exp(-z)))



def apply_net_new(y_in): # same as before, but with new layer function
    global w1,b1,w2,b2
    
    y1=apply_layer_new(y_in,w1,b1)
    y2=apply_layer_new(y1,w2,b2)
    return(y2)

def apply_multi_net(y_in):
    global Weights, Biases, WeightFinal, BiasFinal, Nlayers, WeightFirst, BiasFirst
    
    y = apply_layer_new(y_in,WeightFirst,BiasFirst)
    for j in range(Nlayers):
        y = apply_layer_new(y,Weights[j,:,:],Biases[j,:])
    output = apply_layer_new(y,WeightFinal,BiasFinal)
    return(output)

M=40
v0,v1 = meshgrid(linspace(-0.5,0.5,M),linspace(-0.5,0.5,M))
batchsize = M**2 # number of smaples=  number of pixels = M^2
y_in = zeros([batchsize,2])
y_in[:,0] = v0.flatten()
y_in[:,1] = v1.flatten()

y_out = apply_multi_net(y_in)

y_2D = reshape(y_out[:,0],[M,M])

plt.imshow(y_2D,origin='lower',extent = [-0.5,0.5,-0.5,0.5],interpolation='nearest')
plt.colorbar()
plt.show()


M=400
# Generate a 'mesh grid', i.e. x,y values in an image
v0,v1=meshgrid(linspace(-0.5,0.5,M),linspace(-0.5,0.5,M))
batchsize=M**2 # number of samples = number of pixels = M^2
y_in=zeros([batchsize,2])
y_in[:,0]=v0.flatten() # fill first component (index 0)
y_in[:,1]=v1.flatten() # fill second component


# The next function takes a few seconds:


# use the MxM input grid that we generated above 
y_out=apply_multi_net(y_in) # apply net to all these samples!



y_2D=reshape(y_out[:,0],[M,M]) # back to 2D image



plt.figure(figsize=[10,10])
plt.axes([0,0,1,1]) # fill all of the picture with the image
plt.imshow(y_2D,origin='lower',extent=[-0.5,0.5,-0.5,0.5],interpolation='nearest')
plt.axis('off') # no axes
plt.show()




    

