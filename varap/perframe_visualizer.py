
import sys
import numpy as np
import os
import argparse

from skimage.io import imread
from skimage.io import imsave
import cv2

sys.path.append('../VideoRecog')
import VideoRecog.eval.metrics as metrics
import VideoRecog.eval.score_io as score_io
import VideoRecog.visualization as vis

from varap_common import dm_reg
from varap_common import score_folder

import imageio

def generate_perframe_prediction_visualization(score_file_path, subset, subset_video_idx, frame_size):
    # shape of all_scores: (num_vid x num_frames x num_model x num_classes)
    print('loading all scores, this might take a while, or forever...')
    scores, labels, meta = score_io.load_scores(score_file_path)
    print('all scores loaded, start visualizing per-frame prediction heatmap...')
    dump_path = 'perframe_prediction.png'

    dm = dm_reg['activity_net']

    if subset == 'train':
        names, labels = dm.get_training_set()
    elif subset == 'val':
        names, labels = dm.get_validation_set()
    else:
        raise Exception('unrecognized subset str: {0}'.format(subset))

    names, labels = dm.get_validation_set()
    if 'seg5' in score_file_path:
        mapping = dm.get_nasty_mapping()
        labels = np.array([mapping[label] for label in labels])

    video_name = names[subset_video_idx]
    video_idx=dm.get_video_idx_by_name(video_name)
    video_scores = np.array(scores[subset_video_idx])
    video_segment_tags = dm.get_video_segments_tags(video_name)
    print(video_segment_tags)

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
    spatial_stream_responses = normalized_model_scores[:, labels[subset_video_idx]]

    # temporal stream responses
    model_idx = 1
    model_scores = np.array(video_scores[:, model_idx])
    normalized_model_scores = np.array([metrics.softmax(model_frame_scores[0, :]) for model_frame_scores in model_scores])
    temporal_stream_responses = normalized_model_scores[:, labels[subset_video_idx]]


    plot = vis.plot_frame_prediction_heat_map(spatial_stream_responses, rgb_frames,
                                              temporal_stream_responses, flow_x_frames, flow_y_frames,
                                              video_segment_tags,
                                              img_size=frame_size)
    return plot

def decompose_plot_into_frames(plot, image_size):
    """
    decompose the big plot into a list of frames
    """
    h, w, c = plot.shape
    num_frames = w/image_size[1]
    assert(w % image_size[1] == 0)
    frames = []
    for i in range(num_frames):
        frames.append(plot[:, i * image_size[0]: (i + 1) * image_size[0], :])
    return frames

def generate_video(frames, output_video_path):
    writer = imageio.get_writer(output_video_path, fps=10)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--score_path', type=str, default=None, 
                        help='path to load the perframe prediction scores (default as None)')
    parser.add_argument('--dump_path', type=str, default='perframe',
                        help='path to dump the visualization result (default as perframe)')
    parser.add_argument('--subset', type=str, choices=['train', 'val', 'test'], default='val',
                        help='subset of the dataset (default as val)')
    parser.add_argument('--idx_within_subset', type=int, default=0,
                        help='idx of the video in the subset (default as 0)')
    parser.add_argument('--mode', type=str, default='both', choices=['video', 'image', 'both'],
                        help='running mode (default as video)')
    parser.add_argument('--height', type=int, default=150, 
                        help='height of each visualized frame (default as 150)')
    parser.add_argument('--width', type=int, default=150,
                        help='width of each visualized frame (default as 150)')
    args = parser.parse_args()
    assert args.score_path is not None, 'please specify --score_path <SCORE_PATH>'

    # unpack all arguments
    score_path = args.score_path
    dump_path = args.dump_path
    subset = args.subset
    idx_within_subset = args.idx_within_subset
    mode = args.mode
    frame_size = (args.height, args.width)

    dump_plot= (mode == 'image' or mode == 'both')
    dump_video= (mode == 'video' or mode == 'both')

    # per-frame visualization
    plot = generate_perframe_prediction_visualization(score_path, subset, idx_within_subset, frame_size)

    if dump_plot:
        image_dump_path = dump_path + '.png'
        imsave(image_dump_path, plot)
        print('plot saved to {0}'.format(image_dump_path))
    if dump_video:
        video_dump_path = dump_path + '.mp4'
        frames = decompose_plot_into_frames(plot, frame_size)
        generate_video(frames, video_dump_path)
        print('video saved to {0}'.format(video_dump_path))

