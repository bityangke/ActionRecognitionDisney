import flask
import sys
from flask import request
import json
import numpy as np
import os
import argparse

from skimage.io import imsave
import cv2

sys.path.append('../VideoRecog')
import VideoRecog.eval.metrics as metrics
import VideoRecog.eval.score_io as score_io
import VideoRecog.data.activity_net as activity_net
import VideoRecog.data.hmdb51 as hmdb51
import VideoRecog.visualization as vis


app = flask.Flask(__name__)

def generate_perframe_prediction_visualization(score_file_path):
    # shape of all_scores: (num_vid x num_frames x num_model x num_classes)
    print('loading all scores, this might take a while, or forever...')
    scores, labels, meta = score_io.load_scores(score_file_path)
    print('all scores loaded, start visualizing per-frame prediction heatmap...')
    dump_path = 'perframe_prediction.png'

    dm = dm_reg['activity_net']
    names, labels = dm.get_validation_set()
    if 'seg5' in score_file_path:
        mapping = dm.get_nasty_mapping()
        labels = np.array([mapping[label] for label in labels])

    for validation_video_idx in range(1):
        video_idx=dm.get_video_idx_by_name(names[validation_video_idx])
        video_scores = np.array(scores[validation_video_idx])

        # rgb frames
        rgb_frames = list(dm.vidoe_frame_iterator(video_idx=video_idx, frame_type=0, step=5))
        rgb_frames = [cv2.cvtColor(frame[0,:,:,:], cv2.COLOR_BGR2RGB) for frame in rgb_frames]

        # flow_x
        flow_x_frames = list(dm.vidoe_frame_iterator(video_idx=video_idx, frame_type=1, step=1))
        flow_x_frames = [frame[0,:,:] for frame in flow_x_frames]

        # flow_y
        flow_y_frames = list(dm.vidoe_frame_iterator(video_idx=video_idx, frame_type=2, step=1))
        flow_y_frames = [frame[0,:,:] for frame in flow_y_frames]

        # spatial stream responses
        model_idx = 0
        model_scores = np.array(video_scores[:, model_idx])
        normalized_model_scores = np.array([metrics.softmax(model_frame_scores[0, :]) for model_frame_scores in model_scores])
        spatial_stream_responses = normalized_model_scores[:, labels[validation_video_idx]]

        # temporal stream responses
        model_idx = 1
        model_scores = np.array(video_scores[:, model_idx])
        normalized_model_scores = np.array([metrics.softmax(model_frame_scores[0, :]) for model_frame_scores in model_scores])
        temporal_stream_responses = normalized_model_scores[:, labels[validation_video_idx]]


        plot = vis.plot_frame_prediction_heat_map(spatial_stream_responses, rgb_frames,
                                                  temporal_stream_responses, flow_x_frames, flow_y_frames)


        imsave('test_{0}.png'.format(validation_video_idx), plot)


def generate_report_data(data_manager, score_file_path):
    # load scores and labels
    scores, labels, meta = score_io.load_scores(score_file_path)
    if 'seg5' in score_file_path:
        mapping = data_manager.get_nasty_mapping()
        labels = np.array([mapping[label] for label in labels])

    # evaluate performance
    total_performance, class_performance, confusion_matrix = metrics.eval_all(scores, labels, data_manager.get_num_classes())
    for class_result in class_performance:
        class_result['class_name'] = data_manager.label_idx_to_name(class_result['class_id'])
    # plot confusion matrix
    cm_file_path = 'confusion_mat.png'
    vis.plot_confusion_mat(confusion_matrix, 'static/' + cm_file_path)
    return total_performance, class_performance, cm_file_path


def register_dm():
    """
    register all data managers
    """
    dm_acnet = activity_net.DataActivityNet(annotation_file='/home/mscvproject/mscvproject/data/ActivityNetTrimflow/.scripts/activity_net.v1-3.min.json',
                                            frame_folders='/home/mscvproject/mscvproject/data/ActivityNetUntrimflow/view',
                                            trimmed=False)
    dm_reg = {}
    dm_reg['activity_net'] = dm_acnet
    for key, val in dm_reg.iteritems():
        val.init()
    return dm_reg


# intiailize data managers
dm_reg = register_dm()
score_folder = '/home/mscvproject/mscvproject/code/ActionRecognitionDisney/TSN_eval/scores/untrimmed_val/ref'

# simple routing
@app.route('/')
@app.route('/index')
@app.route('/hello')
def hello_template():
    return "Hello"


@app.route('/request_video_list/<dataset>/<class_id>', methods=['GET'])
def handle_video_list_request(dataset, class_id):
    class_id = int(class_id)
    dm = dm_reg[dataset]
    video_list = dm.get_video_paths_by_class_id(class_id)
    video_list = ['/static/video/' + path for path in video_list]
    json_str = json.dumps(video_list)
    return json_str

@app.route('/scores/<score_file_name>')
def generate_performance_analysis(score_file_name):
    dataset='activity_net' # TODO: pull the dataset info from score meta
    score_file_path = os.path.join(score_folder, score_file_name)
    total_performance, class_performance, cm_file_path = generate_report_data(dm_reg[dataset], score_file_path)
    return flask.render_template('index.html',
                                 total_performance=total_performance,
                                 class_performance=class_performance,
                                 confusion_mat_file=flask.url_for('static', filename=cm_file_path))    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='web', choices=['web', 'pfpv'],
                        help='generate perframe prediction visualization')
    parser.add_argument('--pfpv_score_path', type=str, default=None, 
                        help='path to load the perframe prediction scores')
    args = parser.parse_args()

    if args.mode == 'pfpv':
        assert(args.pfpv_score_path)
        generate_perframe_prediction_visualization(args.pfpv_score_path)
    elif args.mode == 'web':
        app.run(host='0.0.0.0', debug=True)
    else:
        pass

