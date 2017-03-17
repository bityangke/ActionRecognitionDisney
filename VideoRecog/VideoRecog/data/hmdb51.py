""" HMDB51 Data Manager
@Yu Mao
"""

import os


class VideoMetaInfo:
    def __init__(self, full_path):
        self.full_path = full_path
        splits = self.full_path.split('/')
        self.label_name = splits[-2]
        self.name = splits[-1]
        self.name = self.name.split('.')[-2]  # remove suffix


class DataHMDB51:
    def __init__(self, class_list, class_splits_folder, video_folder):
        self.class_list_file = class_list
        self.video_folder = video_folder
        self.class_splits_folder = class_splits_folder
        self.label_names = []
        self.label_to_idx = {}
        self.video_file_list = []
        self.video_labels = []
        self.video_info = []
        self.split_indices = [{'train': [], 'test': []}] * 3
        self.video_name_to_idx = {}
        self.class_to_videos = {}
        self.class_path_to_videos = {}

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
        return self.class_path_to_videos[class_id]

    def get_video_paths_by_class_name(self, class_name):
        """
        get a list of video paths by class name
        """
        class_id = self.label_name_to_idx(class_name)
        return self.get_video_paths_by_class_id(class_id)

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
        self.class_path_to_videos = []
        for i in range(self.get_num_classes()):
            self.class_to_videos.append([])
            self.class_path_to_videos.append([])

        for video_meta in self.video_info:
            video_name = video_meta.name
            label_name = video_meta.label_name
            label_id = self.label_name_to_idx(label_name)
            self.class_to_videos[label_id].append(video_name)
            self.class_path_to_videos[label_id].append('hmdb51_video/' + label_name + '/' + video_name + '.mp4')


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
                             '/data01/mscvproject/data/HMDB/videos')
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
