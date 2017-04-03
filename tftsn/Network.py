#This file will contain the models used in this architecture
import tensorflow as tf
from models.ResNet import ResNet
from models.Inception import Inception
import numpy as np

class Network:
    
    def __init__(self, mode):
        img = np.zeros((1,224,224,3),dtype=np.float32)
        resnet = ResNet()
        self.rgbmodel = resnet.resnet()
        #print( self.rgbmodel.predict(img))
        inception = Inception()
        self.flowmodel= inception.inception()
        img = np.zeros((1,299,299,10),dtype=np.float32)
        #print(self.flowmodel.predict(img))




