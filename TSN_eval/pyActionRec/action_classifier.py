from action_caffe import CaffeNet
from utils.video_funcs import sliding_window_aggregation_func, default_fusion_func
import numpy as np
import time


class ActionClassifier(object):
    """
    This class provides and end-to-end interface to classifying videos into activity classes
    """

    def __init__(self, models=list(), total_norm_weights=None, score_name='fc-action', dev_id=0, data_manager=None):
        """
        Contruct an action classifier
        Args:
            models: list of tuples in the form of
                    (model_proto, model_params, model_fusion_weight, input_type, data_augmentation).
                    input_type is: 0-RGB, 1-Optical flow.
                    data_augmentation indicates whether we will augment the data via oversampling
            total_norm_weights: sum of all model_fusion_weights when normalization is wanted, otherwise use None
        """

        self.__net_vec = [CaffeNet(x[0], x[1], dev_id) for x in models]
        self.__net_weights = [float(x[2]) for x in models]
        if total_norm_weights is not None:
            s = sum(self.__net_weights)
            self.__net_weights = [x/s for x in self.__net_weights]
        self.__input_type = [x[3] for x in models]
        self.__data_aug = [x[4] for x in models]
        self.__num_net = len(models)
        # whether we should prepare flow stack
        self.__need_flow = max(self.__input_type) > 0
        # the name in the proto for action classes
        self.__score_name = score_name
        self.__data_manager = data_manager

    def classify(self, video_name, model_mask=None, verbose=False):
        """
        Input a file on harddisk
        Args:
            filename:

        Returns:
            cls: classification scores
            frm_scores: frame-wise classification scores
        """
        video_idx = self.__data_manager.get_video_idx_by_name(video_name)
        rgb_frm_it = self.__data_manager.vidoe_frame_iterator(video_idx, frame_type=0, batch_size=1, step=5)
        flow_frm_it = None
        if self.__need_flow:
            flow_frm_it = self.__data_manager.vidoe_frame_iterator(video_idx, frame_type=3, batch_size=10, step=1)
        all_scores = []
        all_start = time.clock()

        cnt = 0

        # process model mask
        mask = [True] * self.__num_net
        n_model = self.__num_net
        if model_mask is not None:
            for i in range(len(model_mask)):
                mask[i] = model_mask[i]
                if not mask[i]:
                    n_model -= 1

        for rgb_stack in rgb_frm_it:
            start = time.clock()
            cnt += 1
            frm_scores = []
            flow_stack = None
            if self.__need_flow:
                assert (flow_frm_it is not None)
                flow_stack = flow_frm_it.next()
                if len(flow_stack) < 10:  # the spatial net's input channel is 10, so discard the stack
                    continue
            for net, run, in_type, data_aug in zip(self.__net_vec, mask, self.__input_type, self.__data_aug):
                if not run:
                    continue
                if in_type == 0:  # RGB input
                    frm_scores.append(net.predict_single_frame(rgb_stack[:1], self.__score_name, over_sample=data_aug))
                elif in_type == 1:  # Flow input
                    assert (flow_stack is not None)
                    frm_scores.append(net.predict_single_flow_stack(flow_stack, self.__score_name, over_sample=data_aug))
            all_scores.append(frm_scores)
            end = time.clock()
            elapsed = end - start
            if verbose:
                print("frame sample {}: {} second".format(cnt, elapsed))

        if len(all_scores) == 0: # all_score is of size (#model, #frames, #models, #classes)
            if verbose:
                print('warning: no frames found for ' + video_name)
            return None, None, None, None

        # aggregate frame-wise scores
        model_scores = []
        for i in range(n_model):
            model_scores.append(sliding_window_aggregation_func(np.array([x[i] for x in all_scores]), norm=False))

        final_scores = default_fusion_func(np.zeros_like(model_scores[0]), model_scores, [w for w, m in zip(self.__net_weights, mask) if m])

        all_end = time.clock()
        total_time = all_end - all_start
        if verbose:
            print("total time: {} second".format(total_time))

        return final_scores, model_scores, all_scores, total_time
