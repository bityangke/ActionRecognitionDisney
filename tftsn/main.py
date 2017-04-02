import tensorflow as tf
import numpy as np
import Network

#Function to train the network
def train(model):
    print('The rgb model is', model.rgbmodel.summary())
    print('The flow model is', model.flowmodel.summary())

def main():
    model = Network.Network('Train')
    #Defining our RGB place holder here
    rgbimg = tf.placeholder(tf.float32, shape=(None,224,224,3))
    #Defining our flow placeholder here
    flowimg = tf.placeholder(tf.float32, shape=(None,299,299,10))
    #Defining our target vector here
    #rgbpred = model.rgbmodel.predict(rgbimg) #rgb prediction
    flowpred = model.flowmodel.predict(flowimg) #flow prediction
    targ = tf.placeholder(tf.float32, shape=(None, 200))
    train(model)


if __name__ == '__main__':
    main()
