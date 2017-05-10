"""
This scripts demos how to do single video classification using the framework
Before using this scripts, please download the model files using

bash models/get_reference_models.sh

Usage:

python classify_video.py <video name>
"""

import os
anet_home = os.environ['ANET_HOME']
import sys
sys.path.append(anet_home)
from pyActionRec.action_classifier import ActionClassifier
sys.path.append('/home/mscvproject/mscvproject/code/ActionRecognitionDisney/VideoRecog')
from VideoRecog.data.activity_net import DataActivityNet
from VideoRecog.eval.score_io import save_scores
from VideoRecog.eval.score_io import ScoreDumpMeta
import VideoRecog.config as config
import argparse
import multiprocessing
import math
import numpy as np

def init_data_manager_ACNet():
    """
    Create and initialize data_manager for ActivityNet
    :return:
    """
    data_manager = DataActivityNet(annotation_file=config.activitynet_annotation_file,
                                   frame_folders=config.activitynet_frame_folder,
                                   trimmed=False)

    data_manager.init()
    return data_manager


def build_classifier(data_manager, gpu_index):
    """
    build the end-to-end classifier
    :return:
    """
    # initialize classifier
    models = []
    models.append((spatial_deploy_prototxt_file, spatial_model_file, 1.0, 0, False))
    models.append((temporal_deploy_prototxt_file, temporal_model_file, 0.2, 1, True))
    cls = ActionClassifier(models, dev_id=gpu_index, score_name=score_layer_name, data_manager=data_manager)
    return cls

cls = None
shared_video_processed = multiprocessing.Value('i', 0)
counter_lock = multiprocessing.Lock()
video_total = None

def worker_cls_initializer(gpu_ids):
    """
    initialize cls in the worker process
    """
    gpu_id = gpu_ids.get()
    initialize_cls(gpu_id)

def initialize_cls(gpu_id):
    global cls
    cls = build_classifier(dm, gpu_id)

def infer_video(video_name):
    """
    infer the class scores of a video
    """
    global counter_lock, shared_video_processed
    counter_lock.acquire()
    shared_video_processed.value += 1
    counter_lock.release()
    print('inferring ' + video_name + ' (' + str(shared_video_processed.value) +'/' + str(video_total) + ')...')
    scores, model_scores, all_scores, _ = cls.classify(video_name, verbose=False)
    return scores, model_scores, all_scores

def inference():
    video_names, labels = dm.get_validation_set()

    # map the tasks to workers
    scores = []
    global video_total
    video_total = len(video_names)
    if num_gpu == 1:
        initialize_cls(gpu_idx_start)
        results = [infer_video(video_name) for video_name in video_names]
    else:
        gpu_ids = multiprocessing.Queue()
        for i in range(num_gpu):
            gpu_ids.put(gpu_idx_start + i)
        p = multiprocessing.Pool(processes=num_gpu, initializer=worker_cls_initializer, initargs=(gpu_ids,))
        results = p.map(infer_video, video_names)
    scores = [result[0] for result in results]         
    model_scores = [result[1] for result in results]    # num_vid x num_model x num_classes
    all_scores = [result[2] for result in results]      # num_vid x num_frames x num_model x num_classes

    # find out all the cases where no frames are presented...
    bad_videos = []
    for i in reversed(range(len(scores))):
        if scores[i] is None:
            scores.pop(i)
            model_scores.pop(i)
            all_scores.pop(i)
            labels.pop(i)
            bad_videos.append(video_names[i])

    # save fused scores
    meta = ScoreDumpMeta(scores_dataset, scores_title)
    scores = np.array(scores)
    save_scores(scores_dump_folder, scores=scores, labels=labels, meta=meta)

    # save scores from each model
    model_scores = np.transpose(np.array(model_scores), (1, 0, 2))   # num_model x num_vid x num_classes
    assert(len(model_scores) == 2 and model_scores[0].shape == scores.shape)
    for i in range(len(model_scores)):
        meta = ScoreDumpMeta(scores_dataset, scores_title + '_model_{0}'.format(i))
        save_scores(scores_dump_folder, scores=model_scores[i], labels=labels, meta=meta)

    # save scores from each model for each frames of each video (for detailed analysis)
    meta = ScoreDumpMeta(scores_dataset, scores_title + '_all')
    save_scores(scores_dump_folder, scores=all_scores, labels=labels, meta=meta)

    print('{0} Videos Tested'.format(len(scores)))
    print('{0} Bad Videos Found'.format(len(bad_videos)))
    print('scores dumped to {0}'.format(scores_dump_folder))


parser = argparse.ArgumentParser()
parser.add_argument("spatial_deploy_prototxt_file", type=str)
parser.add_argument("spatial_model_file", type=str)
parser.add_argument("temporal_deploy_prototxt_file", type=str)
parser.add_argument("temporal_model_file", type=str)
parser.add_argument("--num_gpu", type=int, default=1)
parser.add_argument("--gpu_start_idx", type=int, default=0)
parser.add_argument("--score_layer_name", type=str, default='fc-action')
parser.add_argument("--scores_dump_folder", type=str, default='scores/default')
parser.add_argument('--scores_title', type=str, default="Unknown")
parser.add_argument('--scores_dataset', type=str, default="Unknown")

args = parser.parse_args()
spatial_deploy_prototxt_file = args.spatial_deploy_prototxt_file
spatial_model_file = args.spatial_model_file
temporal_deploy_prototxt_file = args.temporal_deploy_prototxt_file
temporal_model_file = args.temporal_model_file
num_gpu = args.num_gpu
gpu_idx_start = args.gpu_start_idx
score_layer_name = args.score_layer_name
scores_dump_folder = args.scores_dump_folder
scores_title = args.scores_title
scores_dataset = args.scores_dataset

dm = init_data_manager_ACNet()  # initialize data manager
inference()                     # start inference

