#This file will contain the models used in this architecture
import tensorflow as tf
from models.ResNet import ResNet

class Network:
    
    def __init__(self, mode):
        self.rgbmodel = ResNet()
        spatialmodel = self.rgbmodel.resnet()
        print('The spatial model is', spatialmodel.summary())
        self.flowmodel= None




