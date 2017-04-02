#This file will contain the models used in this architecture
import tensorflow as tf
from models.ResNet import ResNet
from models.Inception import Inception
import numpy as np

class Network:
    
    def __init__(self, mode):
        img = np.zeros((1,224,224,3),dtype=np.float32)
        self.resnet = ResNet()
        self.rgbmodel = self.resnet.resnet()
        print( self.rgbmodel.predict(img))
        self.inception = Inception()
        self.flowmodel= self.inception.inception()
        print(self.flowmodel.summary())




