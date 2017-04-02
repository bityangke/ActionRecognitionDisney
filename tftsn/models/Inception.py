"""
This script will hold the Inception model which will be used ofr the spatial model
author : Sree Harsha Kalli
"""
from keras.models import Model
from keras import layers
from keras.layers import Activation, Dense, Input, BatchNormalization, Conv2D
from keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D, ZeroPadding2D, Flatten
import keras.backend as K
from models.blocks import buildblocks

class Inception:
    def __init__(self):
        pass

    #Creating the Inception model with 200 classes as default as of now, change 
    #this if the model changes
    
    def inception(input_tensor=None, input_shape=None, classes=200):
    #Will add functionality to load the pre-trained weights as well later
        input_shape = (299,299,10)

        #if input_tensor is None:
        img_input = Input(shape=input_shape)
        #By default the number of axes has to be three 
        channel_axis = 3

        #Creating the model now
        x = buildblocks.conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
        x = buildblocks.conv2d_bn(x, 32, 3, 3, padding='valid')
        x = buildblocks.conv2d_bn(x, 64, 3, 3)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = buildblocks.conv2d_bn(x, 80, 1, 1, padding='valid')
        x = buildblocks.conv2d_bn(x, 192, 3, 3, padding='valid')
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        # mixed 0, 1, 2: 35 x 35 x 256
        branch1x1 = buildblocks.conv2d_bn(x, 64, 1, 1)

        branch5x5 = buildblocks.conv2d_bn(x, 48, 1, 1)
        branch5x5 = buildblocks.conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = buildblocks.conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = buildblocks.conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = buildblocks.conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = buildblocks.conv2d_bn(branch_pool, 32, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed0')

        # mixed 1: 35 x 35 x 256
        branch1x1 = buildblocks.conv2d_bn(x, 64, 1, 1)

        branch5x5 = buildblocks.conv2d_bn(x, 48, 1, 1)
        branch5x5 = buildblocks.conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = buildblocks.conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = buildblocks.conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = buildblocks.conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = buildblocks.conv2d_bn(branch_pool, 64, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed1')

        # mixed 2: 35 x 35 x 256
        branch1x1 = buildblocks.conv2d_bn(x, 64, 1, 1)

        branch5x5 = buildblocks.conv2d_bn(x, 48, 1, 1)
        branch5x5 = buildblocks.conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = buildblocks.conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = buildblocks.conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = buildblocks.conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = buildblocks.conv2d_bn(branch_pool, 64, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed2') 
        
        # mixed 3: 17 x 17 x 768
        branch3x3 = buildblocks.conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

        branch3x3dbl = buildblocks.conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = buildblocks.conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = buildblocks.conv2d_bn(
            branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

        branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = layers.concatenate(
            [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

        # mixed 4: 17 x 17 x 768
        branch1x1 = buildblocks.conv2d_bn(x, 192, 1, 1)

        branch7x7 = buildblocks.conv2d_bn(x, 128, 1, 1)
        branch7x7 = buildblocks.conv2d_bn(branch7x7, 128, 1, 7)
        branch7x7 = buildblocks.conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = buildblocks.conv2d_bn(x, 128, 1, 1)
        branch7x7dbl = buildblocks.conv2d_bn(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = buildblocks.conv2d_bn(branch7x7dbl, 128, 1, 7)
        branch7x7dbl = buildblocks.conv2d_bn(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = buildblocks.conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = buildblocks.conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed4') 
         
        # mixed 5, 6: 17 x 17 x 768
        for i in range(2):
            branch1x1 = buildblocks.conv2d_bn(x, 192, 1, 1)

            branch7x7 = buildblocks.conv2d_bn(x, 160, 1, 1)
            branch7x7 = buildblocks.conv2d_bn(branch7x7, 160, 1, 7)
            branch7x7 = buildblocks.conv2d_bn(branch7x7, 192, 7, 1)

            branch7x7dbl = buildblocks.conv2d_bn(x, 160, 1, 1)
            branch7x7dbl = buildblocks.conv2d_bn(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = buildblocks.conv2d_bn(branch7x7dbl, 160, 1, 7)
            branch7x7dbl = buildblocks.conv2d_bn(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = buildblocks.conv2d_bn(branch7x7dbl, 192, 1, 7)

            branch_pool = AveragePooling2D(
                (3, 3), strides=(1, 1), padding='same')(x)
            branch_pool = buildblocks.conv2d_bn(branch_pool, 192, 1, 1)
            x = layers.concatenate(
                [branch1x1, branch7x7, branch7x7dbl, branch_pool],
                axis=channel_axis,
                name='mixed' + str(5 + i))        
        
        # mixed 7: 17 x 17 x 768
        branch1x1 = buildblocks.conv2d_bn(x, 192, 1, 1)

        branch7x7 = buildblocks.conv2d_bn(x, 192, 1, 1)
        branch7x7 = buildblocks.conv2d_bn(branch7x7, 192, 1, 7)
        branch7x7 = buildblocks.conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = buildblocks.conv2d_bn(x, 192, 1, 1)
        branch7x7dbl = buildblocks.conv2d_bn(branch7x7dbl, 192, 7, 1)
        branch7x7dbl = buildblocks.conv2d_bn(branch7x7dbl, 192, 1, 7)
        branch7x7dbl = buildblocks.conv2d_bn(branch7x7dbl, 192, 7, 1)
        branch7x7dbl = buildblocks.conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = buildblocks.conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed7')

        # mixed 8: 8 x 8 x 1280
        branch3x3 = buildblocks.conv2d_bn(x, 192, 1, 1)
        branch3x3 = buildblocks.conv2d_bn(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid')

        branch7x7x3 = buildblocks.conv2d_bn(x, 192, 1, 1)
        branch7x7x3 = buildblocks.conv2d_bn(branch7x7x3, 192, 1, 7)
        branch7x7x3 = buildblocks.conv2d_bn(branch7x7x3, 192, 7, 1)
        branch7x7x3 = buildblocks.conv2d_bn(
            branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

        branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = layers.concatenate(
            [branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')

        # mixed 9: 8 x 8 x 2048
        for i in range(2):
            branch1x1 = buildblocks.conv2d_bn(x, 320, 1, 1)

            branch3x3 = buildblocks.conv2d_bn(x, 384, 1, 1)
            branch3x3_1 = buildblocks.conv2d_bn(branch3x3, 384, 1, 3)
            branch3x3_2 = buildblocks.conv2d_bn(branch3x3, 384, 3, 1)
            branch3x3 = layers.concatenate(
                [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

            branch3x3dbl = buildblocks.conv2d_bn(x, 448, 1, 1)
            branch3x3dbl = buildblocks.conv2d_bn(branch3x3dbl, 384, 3, 3)
            branch3x3dbl_1 = buildblocks.conv2d_bn(branch3x3dbl, 384, 1, 3)
            branch3x3dbl_2 = buildblocks.conv2d_bn(branch3x3dbl, 384, 3, 1)
            branch3x3dbl = layers.concatenate(
                [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

            branch_pool = AveragePooling2D(
                (3, 3), strides=(1, 1), padding='same')(x)
            branch_pool = buildblocks.conv2d_bn(branch_pool, 192, 1, 1)
            x = layers.concatenate(
                [branch1x1, branch3x3, branch3x3dbl, branch_pool],
                axis=channel_axis,
                name='mixed' + str(9 + i))

        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)

        # Create model.
        model = Model(img_input, x, name='resnet')
        return model
        
