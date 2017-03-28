import flask
import sys
from flask import request
import json
import numpy as np
import os

sys.path.append('../VideoRecog')
import VideoRecog.eval.metrics as metrics
import VideoRecog.eval.score_io as score_io
import VideoRecog.data.activity_net as activity_net
import VideoRecog.data.hmdb51 as hmdb51
import VideoRecog.visualization as vis


app = flask.Flask(__name__)


def generate_report_data(data_manager, score_file_path):
    # load scores and labels
    scores, labels, meta = score_io.load_scores(score_file_path)

    if 'scores_seg5.npz' in score_file_path:
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
    dm_acnet = activity_net.DataActivityNet(annotation_file='/data01/mscvproject/data/ActivityNet/activity_net.v1-3.min.json',
                                            frame_folders='/data01/mscvproject/data/ActivityNetTrimflow/view')

    dm_hmdb51 = hmdb51.DataHMDB51('/data01/mscvproject/data/HMDB/class_list.txt',
                                  '/data01/mscvproject/data/HMDB/test_train_splits',
                                  '/data01/mscvproject/data/HMDB/videos',
                                  '/data01/mscvproject/data/hmdb51_frame_and_flow')
    dm_reg = {}
    dm_reg['activity_net'] = dm_acnet
    dm_reg['hmdb51'] = dm_hmdb51
    for key, val in dm_reg.iteritems():
        val.init()
    return dm_reg


# intiailize data managers
dm_reg = register_dm()
score_folder = '/data01/mscvproject/code/ActionRecognitionDisney/TSN_eval/scores'


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
    app.run(host='0.0.0.0', debug=True)

