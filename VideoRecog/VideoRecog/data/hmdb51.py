""" HMDB51 Data Manager
@Yu Mao
"""

import os
import cv2
import numpy as np


class VideoMetaInfo:
    def __init__(self, full_path):
        self.full_path = full_path
        splits = self.full_path.split('/')
        self.label_name = splits[-2]
        self.name = splits[-1]
        self.name = self.name.split('.')[-2]  # remove suffix


class DataHMDB51:
    def __init__(self, class_list, class_splits_folder, video_folder, frame_folder):
        self.class_list_file = class_list
        self.video_folder = video_folder
        self.frame_folder = frame_folder
        self.class_splits_folder = class_splits_folder
        self.label_names = []                                            # class names
        self.label_to_idx = {}                                           # class name to its index in self.label_names
        self.video_info = []                                             # video info
        self.video_labels = []                                           # video labels corresponding to the videos in video_info
        self.split_indices = [{'train': [], 'test': []}] * 3             # indices of videos in each splits
        self.video_name_to_idx = {}                                      # video name to index
        self.class_to_videos = {}                                        # class idx to list of video names
        self.class_to_video_paths = {}                                   # class idx to list of video paths

    def init(self):
        self.__parse_video_folder()
        self.__parse_class_list()
        self.__parse_splits_folder()
        self.__establish_mapping_class_to_videos()

    def get_num_classes(self):
        return len(self.label_names)

    def label_idx_to_name(self, label_idx):
        """
        convert the label index to its name
        :param label_idx:
        :return: label name
        """
        assert (label_idx < len(self.label_names))
        return self.label_names[label_idx]

    def label_name_to_idx(self, label_name):
        """
        convert the label name to its index
        :param label_name:
        :return:
        """
        return self.label_to_idx[label_name]

    def get_video_names_by_class_id(self, class_id):
        """
        get a list of video names by class id
        """
        return self.class_to_videos[class_id]

    def get_video_names_by_class_name(self, class_name):
        """
        get a list of video names by class name
        """
        class_id = self.label_name_to_idx(class_name)
        return self.get_video_names_by_class_id(class_id)

    def get_video_paths_by_class_id(self, class_id):
        """
        get a list of video paths by class id
        """
        return self.class_to_video_paths[class_id]

    def get_video_paths_by_class_name(self, class_name):
        """
        get a list of video paths by class name
        """
        class_id = self.label_name_to_idx(class_name)
        return self.get_video_paths_by_class_id(class_id)

    def get_video_name_by_idx(self, idx):
        """
        get name of the video by its index
        """
        return None

    def get_video_idx_by_name(self, name):
        """
        get idex of the video by its name
        """
        return None

    frame_prefixes = ('img_', 'flow_x_', 'flow_y_')
    def get_frame_paths(self, video_idx, frame_type):
        frame_paths = []
        video_name = self.video_info[video_idx].name
        prefix = DataHMDB51.frame_prefixes[frame_type]
        frame_folder = os.path.join(self.frame_folder, video_name)
        for root, dirs, files in os.walk(frame_folder):
            files.sort()
            for file_name in files:
                full_path = os.path.join(root, file_name)
                if file_name.startswith(prefix):
                    frame_paths.append(full_path)
        return frame_paths

    def vidoe_frame_iterator(self, video_idx, frame_type, batch_size=1, step=1):
        """
        create an iterator that iterate through frames of a video
        :param video_idx: the index of the video
        :param frame_type: the type of the frame (0 for rgb, 1 for flow_x, 2 for flow_y, 3 for interleaved flow_x and flow_y)
        :param batch_size:
        :param step:
        :return: (n, h, w, c) where n is less or equal to batch_size
        """
        # fetch the paths of all frames
        frame_paths = []
        if 0 <= frame_type <=2:
            frame_paths = self.get_frame_paths(video_idx, frame_type)
        elif frame_type == 3:
            frame_paths_flow_x = self.get_frame_paths(video_idx, 1)
            frame_paths_flow_y = self.get_frame_paths(video_idx, 2)
            assert(len(frame_paths_flow_x) == len(frame_paths_flow_y))
            for i in range(len(frame_paths_flow_x)):
                frame_paths.append(frame_paths_flow_x[i])
                frame_paths.append(frame_paths_flow_y[i])

        # loading images
        cnt = 0
        result = []
        for i in range(0, len(frame_paths), step):
            cnt += 1
            result.append(cv2.imread(frame_paths[i], cv2.IMREAD_UNCHANGED))
            if cnt == batch_size:
                yield np.array(result)
                result = []
                cnt = 0
        if len(result) !=0:
            yield np.array(result)
        return

    def __parse_video_folder(self):
        for root, dirs, files in os.walk(self.video_folder):
            for file in files:
                if not file.endswith('.mp4'):
                    continue
                full_path = os.path.join(root, file)
                self.video_info.append(VideoMetaInfo(full_path))
                self.video_name_to_idx[self.video_info[-1].name] = len(self.video_info)-1

    def __parse_class_list(self):
        for line in open(self.class_list_file):
            label_name = line.strip()
            self.label_names.append(label_name)
            self.label_to_idx[label_name] = len(self.label_names)-1

    def __parse_splits_folder(self):
        if self.class_splits_folder is not None:
            for root, dirs, files in os.walk(self.class_splits_folder):
                for file_name in files:
                    if not file_name.endswith('.txt'):
                        continue
                    label_name, split_group = DataHMDB51.__parse_split_file_name(file_name)
                    file_name = os.path.join(root, file_name)
                    for line in open(file_name):
                        fields = line.strip().split()
                        if fields[1] == '0':
                            continue
                        video_name = fields[0].split('.')[-2]
                        section = 'train' if fields[1] == '1' else 'test'
                        self.split_indices[split_group][section].append(self.video_name_to_idx[video_name])

    def __establish_mapping_class_to_videos(self):
        # class_to_videos
        self.class_to_videos = []
        self.class_to_video_paths = []
        for i in range(self.get_num_classes()):
            self.class_to_videos.append([])
            self.class_to_video_paths.append([])

        for video_meta in self.video_info:
            video_name = video_meta.name
            label_name = video_meta.label_name
            label_id = self.label_name_to_idx(label_name)
            self.class_to_videos[label_id].append(video_name)
            self.class_to_video_paths[label_id].append('hmdb51_video/' + label_name + '/' + video_name + '.mp4')

    @staticmethod
    def __parse_split_file_name(file_name):
        idx = file_name.rfind('.')
        split_group = int(file_name[idx-1:idx])
        idx = file_name.rfind('_test_split')
        label_name = file_name[:idx]
        return label_name, split_group-1


if __name__ == '__main__':
    datamanager = DataHMDB51('/data01/mscvproject/data/HMDB/class_list.txt',
                             '/data01/mscvproject/data/HMDB/test_train_splits',
                             '/data01/mscvproject/data/HMDB/videos',
                             '/data01/mscvproject/data/hmdb51_frame_and_flow')
    datamanager.init()
    print(datamanager.get_num_classes())

    for i in range(datamanager.get_num_classes()):
        print(datamanager.label_idx_to_name(i))
        result = datamanager.get_video_names_by_class_id(i)
        print(len(result))
        print(result[0])

    for i in range(datamanager.get_num_classes()):
        print(datamanager.label_idx_to_name(i))
        result = datamanager.get_video_paths_by_class_id(i)
        print(len(result))
        print(result[0])

    print('total video number: ' + str(len(datamanager.video_info)))
    print(datamanager.video_info[0].name)

    # test frame iterator
    for frame_stack in datamanager.vidoe_frame_iterator(video_idx=0, frame_type=0, batch_size=20):
        print(frame_stack.shape)




