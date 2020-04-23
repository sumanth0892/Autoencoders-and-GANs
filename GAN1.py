import numpy as np 
from keras.datasets import mnist 
from keras.layers import Input,Dense,Lambda 
from keras.losses import binary_crossentropy as bc 
from keras import backend as K 
from keras.models import Model 

#Hyperparameters 
originalDim = 28*28 #Depends on the application and the dataset 
latentDim = 2
intermediateDim = 256 #usually 32 is also fine 
epochs = 50
batch_size = 100

def sampling(args):
	z_mean,z_log_var = args 
	epsilon = K.random_normal(shape = (batch_size,latentDim),mean = 0.0)
	return z_mean + K.exp(z_log_var/2)*epsilon


#Encoder 
x = Input(shape = (originalDim,))
h = Dense(intermediateDim,activation = 'relu')(x)
z_mean = Dense(latentDim,name = 'mean')(h)
z_log_var = Dense(latentDim,name = 'log-variance')(h)
z = Lambda(sampling,output_shape = (latentDim,))([z_mean,z_log_var])
Encoder = Model(x,[z_mean,z_log_var,z])

#Decoder 
decoderInput = Input(shape = (latentDim,))
hDecoder = Dense(intermediateDim,activation = 'relu')(decoderInput)
xDecoder = Dense(originalDim,activation = 'sigmoid')(hDecoder)
Decoder = Model(decoderInput,xDecoder)

#Putting it together 
outputDecoder = Decoder(Encoder(x)[2])
vae = Model(x,outputDecoder)

def vaeLoss(x,x_decoder_mean,z_mean=z_mean,z_log_var=z_log_var,originalDim=originalDim):
	xent_loss = originalDim*bc(x,x_decoder_mean)
	kl_loss = -0.5*K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var),axis = -1)
	return xent_loss + kl_loss 

vae.compile(optimizer = 'rmsprop',loss = vaeLoss)
vae.summary()