import flask
import sys
from flask import request
import json

sys.path.append('../VideoRecog')
import VideoRecog.eval.metrics as metrics
import VideoRecog.eval.score_io as score_io
import VideoRecog.data.activity_net as activity_net
import VideoRecog.data.hmdb51 as hmdb51
import VideoRecog.visualization as vis


app = flask.Flask(__name__)


def generate_report_data(data_manager, score_file_path):
    # load scores and labels
    scores, labels = score_io.load_scores_an(score_file_path)
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
    dm_acnet  = activity_net.DataActivityNet('../VideoRecog/VideoRecog/data/acnet.json', None)
    dm_hmdb51 = hmdb51.DataHMDB51('/data01/mscvproject/data/HMDB/class_list.txt',
                                  '/data01/mscvproject/data/HMDB/test_train_splits',
                                  '/data01/mscvproject/data/HMDB/videos')
    dm_reg = {}
    dm_reg['activity_net'] = dm_acnet
    dm_reg['hmdb51'] = dm_hmdb51
    for key, val in dm_reg.iteritems():
        val.init()
    return dm_reg

# test code
dm_reg = register_dm()
score_file_path = '/data01/mscvproject/code/temporal-segment-networks/results/scores/score_hmdb51_flow_1.npz'
total_performance, class_performance, cm_file_path = generate_report_data(dm_reg['hmdb51'], score_file_path)


# simple routing
@app.route('/')
@app.route('/index')
def main_page():
    return flask.render_template('index.html',
                                 total_performance=total_performance,
                                 class_performance=class_performance,
                                 confusion_mat_file=flask.url_for('static', filename=cm_file_path))


@app.route('/hello')
def hello_template():
    return flask.render_template('hello_world.html', name='MSCV')


@app.route('/request_video_list/<dataset>/<class_id>', methods=['GET'])
def handle_video_list_request(dataset, class_id):
    class_id = int(class_id)
    dm = dm_reg[dataset]
    video_list = dm.get_video_paths_by_class_id(class_id)
    video_list = ['/static/video/' + path for path in video_list]
    json_str = json.dumps(video_list)
    return json_str


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

