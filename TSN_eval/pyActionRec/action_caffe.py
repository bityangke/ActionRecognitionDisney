from config import ANET_CFG

import sys

sys.path.append(ANET_CFG.CAFFE_ROOT+'/python')

import caffe
from caffe.io import oversample
import numpy as np
from utils.io import flow_stack_oversample
from caffe.io import resize_image
import cv2


class CaffeNet(object):

    def __init__(self, net_proto, net_weights, device_id):
        """
        Initialize CaffeNet
        :param net_proto: file path for deploy protxt
        :param net_weights: file path for model weights
        :param device_id: GPU device ID
        :param input_size: size(H, W) of the input frame
        """
        caffe.set_mode_gpu()
        caffe.set_device(device_id)

        # initialize net
        self._net = caffe.Net(net_proto, net_weights, caffe.TEST)

        # initialize transformer
        input_shape = self._net.blobs['data'].data.shape
        transformer = caffe.io.Transformer({'data': input_shape})
        if self._net.blobs['data'].data.shape[1] == 3:
            transformer.set_transpose('data', (2, 0, 1))  # (h, w, c) -> (c, h, w)
            transformer.set_mean('data', np.array([104, 117, 123]))  # mean subtraction
        else:
            pass  # non RGB data need not use transformer
        self._transformer = transformer
        self._sample_shape = self._net.blobs['data'].data.shape

    def predict_single_frame(self, frames, score_name, over_sample=True, multiscale=None):
        """
        predict the class scores for a single frame, but can also accept a list of frames as input
        :param frames: batch data
        :param score_name: name of the read-out layer
        :param over_sample: data augmentation by over-sampling (cropping the image at center, corners and double the amount by mirroring)
        :param multiscale: data augmentation by multi-scaling, should be a list of values between [0, 1]
        :param frame_size: frame size (H,W) [frame_size should be >= sample_region_size in order to get multi-scale oversampling working]
        Note that after augmentation, batch size changes from len(frames) to 10 * multi-scale * len(frames)
        :return:
        """

        if over_sample:
            if multiscale is None:
                os_frame = oversample(frames, (self._sample_shape[2], self._sample_shape[3]))
            else:
                os_frame = []
                for scale in multiscale:
                    resized_frame = [cv2.resize(x, (0,0), fx=1.0/scale, fy=1.0/scale) for x in frames]
                    os_frame.extend(oversample(resized_frame, (self._sample_shape[2], self._sample_shape[3])))
        else:
            os_frame = np.array(frames)
        data = np.array([self._transformer.preprocess('data', x) for x in os_frame])

        self._net.blobs['data'].reshape(*data.shape)  # adjust the batch number
        self._net.reshape()

        out = self._net.forward(blobs=[score_name,], data=data)
        return out[score_name].copy()

    def predict_single_flow_stack(self, frames, score_name, over_sample=True):
        """
        Predict from a single frame stack
        :param frames: (N,H,W)
        :param score_name: name of the read-out layer
        :param over_sample: data augmentation with oversampling + horizontal flipping
        :return:
        """
        if over_sample:
            os_frame = flow_stack_oversample(frames, (self._sample_shape[2], self._sample_shape[3]))
        else:
            os_frame = np.array([frames,])

        # centering
        data = os_frame - 128

        self._net.blobs['data'].reshape(*data.shape)
        self._net.reshape()
        out = self._net.forward(blobs=[score_name,], data=data)
        return out[score_name].copy()

