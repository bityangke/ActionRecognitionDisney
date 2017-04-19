"""
This module provides all the visualization routines
@Yu Mao
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.color import rgb2grey
from skimage.exposure import adjust_gamma


def plot_confusion_mat(confusion_mat, dump_file):
    """
    Plot and dump confusion matrix.
    :param confusion_mat: confusion_mat of shape (num_classes, num_classes)
    :param dump_file: output image file
    :return:
    """
    fig, ax = plt.subplots()
    ax.pcolor(confusion_mat, cmap=plt.cm.Blues)
    # put the major ticks at the middle of each cell
    ax.set_xlim([0, confusion_mat.shape[1]])
    ax.set_ylim([0, confusion_mat.shape[0]])
    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    plt.savefig(dump_file)


def plot_image_grid(frames, grid_size = (3,3)):
    frame_h = frames[0].shape[0]; frame_w = frames[0].shape[1]
    result = np.zeros((frame_h * grid_size[0], frame_w * grid_size[1]))
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            if (i * grid_size[1] + j) >= len(frames):
                break
            result[frame_h*i: frame_h*(i+1), frame_w*j: frame_w*(j+1)] = frames[i * grid_size[1] + j]
    return result


def plot_frame_prediction_heat_map(spatial_stream_responses, rgb_frames,
                                   temporal_stream_responses, flow_x_frames, flow_y_frames, 
                                   img_size=(255,255),
                                   flow_gamma = 0.3, flow_step = 5, flow_grid_size=(2,2)):
    """
    Visualize prediciton heat map of video
    :param spatial_stream_responses: per-frame prediction of shape (num_frame,) of spatial stream
    :param frames: list of rgb frames to be concatenated
    :param step: sampling interval of frames
    :param img_size: new image size
    :return: plot
    """
    img_size = (img_size[0] - img_size[0]%flow_grid_size[0], img_size[1] - img_size[1]%flow_grid_size[1])
    h, w = img_size
    flow_h, flow_w = h/flow_grid_size[0], w/flow_grid_size[1]

    rgb_frames_resized = [resize(frame, img_size) for frame in rgb_frames]
    flow_x_frames_resized = [resize(frame, (flow_h, flow_w)) for frame in flow_x_frames]
    flow_y_frames_resized = [resize(frame, (flow_h, flow_w)) for frame in flow_y_frames]

    # concatenate rgb frames horizontally
    num_frame = len(spatial_stream_responses)
    num_frames_sampled = len(range(num_frame))
    frames_concatenated = np.zeros((h * 3, w * num_frames_sampled, 3))
    for idx, i in enumerate(range(num_frame)):
        frames_concatenated[0: h, w * idx: w * (idx+1), :] = rgb2grey(rgb_frames_resized[i])[:,:,np.newaxis]
    for idx, i in enumerate(range(num_frame)):
        flow_x_grid = plot_image_grid(flow_x_frames_resized[i * flow_step: min((i + 1) * flow_step, len(flow_x_frames))], grid_size=flow_grid_size)
        frames_concatenated[h: h * 2, w * idx: w * (idx+1), :] = flow_x_grid[:,:,np.newaxis]
    for idx, i in enumerate(range(num_frame)):
        flow_y_grid = plot_image_grid(flow_y_frames_resized[i * flow_step: min((i + 1) * flow_step, len(flow_y_frames))], grid_size=flow_grid_size)
        frames_concatenated[h * 2: h * 3, w * idx: w * (idx+1), :] = flow_y_grid[:,:,np.newaxis]

    # create confidence map
    mask_confidence = np.zeros((h * 3, w * num_frames_sampled, 3))
    color = np.array([1,0,0])
    for idx, i in enumerate(range(num_frame)):
        mask_confidence[0: h, w * idx: w * (idx+1), :] = color * spatial_stream_responses[i]
    for idx, i in enumerate(range(num_frame)):
        mask_confidence[h: h * 2, w * idx: w * (idx+1), :] = color * temporal_stream_responses[i]
    for idx, i in enumerate(range(num_frame)):
        mask_confidence[h * 2: h * 3, w * idx: w * (idx+1), :] = color * temporal_stream_responses[i]

    # blend them together
    frames_concatenated = np.clip(frames_concatenated + mask_confidence, 0, 1)

    # save
    return frames_concatenated

