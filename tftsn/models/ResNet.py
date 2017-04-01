#This script will hold the ResNet model which will be used ofr the spatial model
from keras.models import Model
from keras import layers
from keras.layers import Activation, Dense, Input, BatchNormalization, Conv2D
from keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D, ZeroPadding2D
import keras.backend as K
from blocks import buildblocks

class ResNet:
    def __init__(self):
        pass
    #Creating the ResNet model with 200 classes as default as of now, change 
    #this if the model changes
    def resnet(input_tensor=None, input_shape=None, classes=200):
    #Will add functionality to load the pre-trained weights as well later
        input_shape = (224,224,3)

        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor
        #By default the number of axes has to be three
        bn_axis = 3

        #Creating the model now
        print('The shape of the input image is', img_input)
        x = ZeroPadding2D((3, 3))(img_input)
        x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
        x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = buildblocks.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = buildblocks.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = buildblocks.identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = buildblocks.conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = buildblocks.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = buildblocks.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = buildblocks.identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        x = buildblocks.conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = buildblocks.identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = buildblocks.identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = buildblocks.identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = buildblocks.identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = buildblocks.identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        x = buildblocks.conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = buildblocks.identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = buildblocks.identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        x = AveragePooling2D((7, 7), name='avg_pool')(x)
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc200')(x)
        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)
        else:
            inputs = img_input
        # Create model.
        model = Model(inputs, x, name='resnet')
        print('The model is', model.summary())
        return models
        
