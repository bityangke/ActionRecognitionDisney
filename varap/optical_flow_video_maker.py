""" This script visualizes frames under a folder into a video
"""
import argparse
import os
import skimage.transform
import skimage.io
import imageio
import numpy as np


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('input_folder')
	parser.add_argument('output_path')
	parser.add_argument('--frame_width', default=300)
	parser.add_argument('--frame_height', default=300)
	args = parser.parse_args()
	input_folder = args.input_folder
	output_path = args.output_path
	frame_height = args.frame_height
	frame_width = args.frame_width

	num_files = len(os.listdir(input_folder))
	assert(num_files%3==0)
	num_frames = num_files/3
	video_frame = np.zeros((frame_height, frame_width * 3, 3))

	writer = imageio.get_writer(output_path, fps=30)
	print('writing video frames...')
	for i in range(num_frames):
		img_path = os.path.join(input_folder, 'img_%05d.jpg' % (i + 1))
		flow_x_path = os.path.join(input_folder, 'flow_x_%05d.jpg' % (i + 1))
		flow_y_path = os.path.join(input_folder, 'flow_y_%05d.jpg' % (i + 1))
		rgb_frame = skimage.transform.resize(skimage.io.imread(img_path), (frame_height, frame_width))
		flow_x_frame = skimage.transform.resize(skimage.io.imread(flow_x_path), (frame_height, frame_width))
		flow_x_frame = np.expand_dims(flow_x_frame, axis=2)
		flow_y_frame = skimage.transform.resize(skimage.io.imread(flow_y_path), (frame_height, frame_width))
		flow_y_frame = np.expand_dims(flow_y_frame, axis=2)
		video_frame[:, 0:frame_width, :] = rgb_frame
		video_frame[:, frame_width:2*frame_width, :] = flow_x_frame
		video_frame[:, 2*frame_width:3*frame_width, :] = flow_y_frame
		writer.append_data(video_frame)
		if (i+1) % 100 == 0:
			print('{0}/{1} frames processed.'.format(i+1, num_frames))
	writer.close()
	print('done.')
	
