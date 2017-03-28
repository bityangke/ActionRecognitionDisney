"""
routines for loading & writing prediction scores
"""
import numpy as np
import datetime
import os

class ScoreDumpMeta():
    def __init__(self, dataset, title):
        self.dataset = dataset
        self.title = title
        self.date = None
    def __str__(self):
        return '(' + ', '.join([self.dataset, self.title, self.date]) + ')'


def load_scores(file_path):
    """
    load prediction scores
    :return: a tuple containing (scores, labels) or (scores, None) if the dump contains no labels.
    """
    result = np.load(file_path)
    scores = result['scores']
    labels = result['labels']
    meta = result['meta']
    return scores, labels, meta


def save_scores(folder, scores, labels, meta, auto_time_stamp=True):
    """
    save prediciton scores to folder
    :return:
    """
    if auto_time_stamp:
        meta.date = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
    if not os.path.isdir(folder):
        os.mkdir(folder)
    file_path = os.path.join(folder, meta.title + '.npz')
    np.savez(file_path, meta=meta, scores=scores, labels=labels)


def test():
    score_file_path = '/data01/mscvproject/code/temporal-segment-networks/results/scores/score_hmdb51_flow_1.npz'
    scores, labels = load_scores_an(score_file_path)
    print(scores.shape)

if __name__ == '__main__':
    test()

