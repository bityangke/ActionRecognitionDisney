import tensorflow as tf
import numpy as np
import Network
import os
import sys
from keras.models import Model, Sequential
from keras.layers import Activation, Dense, Input, BatchNormalization, Conv2D
from keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D, ZeroPadding2D, Flatten
import keras.backend as K

from VideoRecog.VideoRecog.data.activity_net import DataActivityNet
from VideoRecog.VideoRecog.TSNUtils.frame_sampler import *

#Function to train the network
def train(model):
    pass


def sampler_usage():
    # initailize data manager and sampler
    annotation_path = '/data01/mscvproject/data/ActivityNetTrimflow/.scripts/activity_net.v1-3.min.json'
    frame_folder = '/data01/mscvproject/data/ActivityNetTrimflow/view'
    data_manager = DataActivityNet(annotation_path, frame_folder)
    sampler = UniformSegmentSampler(num_segments=4)

    iteration = 0
    max_iteration = 1000
    while iteration < max_iteration
        # sample a mini-batch
        # frame_type: 0 for rgb, 1 for flow_x, 2 for flow_y, 3 for both flow_x and flow_y
        # mini_batch is of shape: (n, h, w, c)
        mini_batch = data_manager.get_minibatch(sampler, frame_type = 1)
        #train
        #.....

def main():
    writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph())
    model = Network.Network('Trainer')
    img = np.random.random((2,224,224,3))
    rgbimg = tf.placeholder(tf.float32, shape=(None,224,224,3))
    flowimg = tf.placeholder(tf.float32, shape=(None,299,299,10))
    with tf.name_scope('rgb'):
        rgbpred = model.rgbmodel(rgbimg) #rgb prediction
    with tf.name_scope('flow'):
        flowpred = model.flowmodel(flowimg) #Flow prediction
    targetpred = tf.placeholder(tf.float32, shape=(None,200)) #The target values
    mulval = targetpred * tf.log(rgbpred)
    rgbce = tf.reduce_mean(-tf.reduce_sum(mulval, reduction_indices=[1]))
    flowce = tf.reduce_mean(-tf.reduce_sum(targetpred * tf.log(flowpred), reduction_indices=[1]))
    tf.summary.scalar("Loss", rgbce)
    if sys.argv[1] == 'rgb':
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(rgbce)
    elif sys.argv[1] == 'flow':
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(flowce)
    else:
        print('Invalid modality requested has to be either rgb or flow')
        return 0
    merged_summary_op = tf.summary.merge_all()
    with tf.Session() as sess:
        K.set_session(sess)
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('logs', graph=sess.graph)
        _, mersum = sess.run([train_step, merged_summary_op], feed_dict={rgbimg: img, targetpred: np.zeros((2,200),dtype=np.float32)})
        writer.add_summary(mersum, 1)


if __name__ == '__main__':
    main()
