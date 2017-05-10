""" This script analyze the correlation between the number of shots and its scores
"""
import sys
import argparse
import numpy as np
from scipy.stats.stats import pearsonr
sys.path.append('../VideoRecog')
import VideoRecog.eval.score_io as score_io
import VideoRecog.visualization as vis
import VideoRecog.eval.metrics as metrics
import VideoRecog.config as config
from VideoRecog.data.shots import ShotsInfoMgr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--score_path', type=str, default=None, 
                        help='path to load the video prediction scores (default as None)')
    parser.add_argument('--plot_dump_path', type=str, default='shot_scores_correlation.png')
    args = parser.parse_args()
    assert args.score_path is not None, 'please specify --score_path <SCORE_PATH>'

    score_file_path = args.score_path
    plot_dump_path = args.plot_dump_path

    # get validation set video names
    print('loading data manager...')
    from varap_common import dm_reg
    from varap_common import score_folder
    dm = dm_reg['activity_net']
    names, gt_labels = dm.get_validation_set()
    if 'seg5' in score_file_path:
        mapping = dm.get_nasty_mapping()
        gt_labels = np.array([mapping[label] for label in gt_labels])

    # load video scores
    print('loading scores...')
    scores, _, _ = score_io.load_scores(score_file_path)
    for i in range(len(scores)):
        scores[i] = metrics.softmax(scores[i])
    gt_scores = [scores[i, gt_labels[i]] for i in range(len(scores))]

    # load shots information
    mgr = ShotsInfoMgr()
    print('loading shots info...')
    mgr.load(config.activity_net_validation_shots_folder)
    shots_per_minute = [float(mgr.get_num_shots(name)) / mgr.get_num_frames(name) * 60 for name in names]

    # compute correlation
    print('correlation analysis...')
    assert(len(shots_per_minute) == len(gt_scores))
    correlation = pearsonr(shots_per_minute, gt_scores)[0]
    print('correlation: {0}'.format(correlation))

    # plot
    vis.plot_xy(shots_per_minute, gt_scores, 'shots per minute', 'score on GT label', '', plot_dump_path, xy_range=[0,2,0,1])

