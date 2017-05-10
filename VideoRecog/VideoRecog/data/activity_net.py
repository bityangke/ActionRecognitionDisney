""" ActivityNet Data Manager
@Yu Mao
"""

import json
import numpy as np
import os
import cv2
import random

# training video class idx file
# (the class idx used for training is different from the one used for evaluation
# thus we need this file to generate the mapping. The ultimate goal is to remove
# this hack.)
# activity_net_training_class_idx_file = '/data01/mscvproject/data/ActivityNetTrimflow/.scripts/ActNetClassesInd.txt'
activity_net_training_class_idx_file = '/home/mscvproject/mscvproject/code/temporal-segment-networks/data/ActNet_splits/ActNetClasses.txt'

class LabelRawInfo:
    """
    Raw label info parsed from the JSON file
    """
    def __init__(self, acnet_id, name, parent_id, parent_name):
        self.acnet_id = acnet_id  # the id is assigned by AcNet data-set
        self.name = name
        self.parent_id = parent_id
        self.parent_name = parent_name

    def __str__(self):
        return '(' + self.name + ', ' + str(self.acnet_id) + ')'


class LabelNode:
    """
    A node representing a label in the taxonomy hierarchy
    """
    def __init__(self, label):
        self.label = label
        self.childs = []
        self.parent = None
        self.level = 0

    def add_child(self, node):
        self.childs.append(node)

    def set_parent(self, parent):
        self.parent = parent
        self.level = self.parent.level + 1

    def __str__(self):
        ret = '\t' * (self.level) + ' ' + str(self.label)
        ret += '\n'
        for child in self.childs:
            ret += str(child)
        return ret


class LabelHierarchy:
    """
    A tree representing the label hierarchy
    """
    def __init__(self):
        self.root = None

    def build(self, labels):
        """
        build the label hierarchy
        :param labels:
        :return:
        """
        i = 0
        labels = list(labels)
        while i < len(labels):
            self.__add_label(labels[i], labels)
            i += 1

    def __add_label(self, label, labels):
        if self.find_node(label.acnet_id) is not None:
            raise Exception('duplicated node added: ' + str(label))

        node = LabelNode(label)
        if label.parent_id is None:
            self.root = node
            return self.root
        parent_node = self.find_node(label.parent_id)
        if parent_node is None:
            for i in range(len(labels)):
                if labels[i].acnet_id == label.parent_id:
                    parent_label = labels.pop(i)
                    parent_node = self.__add_label(parent_label, labels)
                    break
        if parent_node is None:
            raise Exception('node with id ' + str(label.parent_id) + ' not found.')
        parent_node.add_child(node)
        node.set_parent(parent_node)
        return node

    def find_node(self, id):
        """
        find a node by id
        :param id: the node id
        :return: the reference to the node
        """
        return self.__find_node_recursively(id, self.root)

    def get_leaf_labels(self):
        leaf_nodes = self.__get_leaf_nodes()
        return [node.label.name for node in leaf_nodes]

    def __get_leaf_nodes(self):
        return self.__get_leaf_nodes_recursive(self.root)

    def __get_leaf_nodes_recursive(self, root):
        if root is None:
            return []
        elif len(root.childs) == 0:
            return [root]
        else:
            ret = []
            for child in root.childs:
                ret.extend(self.__get_leaf_nodes_recursive(child))
            return ret

    def __find_node_recursively(self, id, root):
        if root is None:
            return None
        elif root.label.acnet_id == id:
            return root
        elif len(root.childs) == 0:
            return None
        else:
            for child in root.childs:
                result = self.__find_node_recursively(id, child)
                if result is not None:
                    return result
            return None

    def __str__(self):
        return str(self.root)


class VideoMetaInfo:
    """
    a structure holding the meta info of a video
    """
    class SegmentAnnotation:
        def __init__(self):
            self.start = 0
            self.end = 0
            self.duration = 0
            self.label = None

    def __init__(self):
        self.name = None
        self.duration = None
        self.subset = None
        self.resolution = None
        self.url = None
        self.annotations = None
        self.label = None

    def __str__(self):
        ret = self.name + ': {'
        fields = [str(field) for field in [self.subset, self.label, self.duration, self.resolution, self.url, len(self.annotations)]]
        ret += ','.join(fields)
        ret += '}'
        return ret


class DataActivityNet:
    def __init__(self, annotation_file, frame_folders, trimmed=True):
        """
        initialize data manager for ActivityNet
        :param annotation_file: annotation file of ActivityNet
        :param frame_folders: a dict recording frame_type: frame_folder mappings
        The frame_folder must contain #videos of subdirectories each named as the video name.
        And each subdirectory should contain the frames extracted from the corresponding video.
        """
        self.version = None
        self.frame_folder = frame_folders
        self.labels = None
        self.label_idx_table = None
        self.label_hierarchy = None
        self.taxonomy = None
        self.video_meta = {}
        self.class_to_videos = {}
        self.class_to_video_paths = {}
        self.video_seg_names = []
        self.video_seg_labels = []
        self.video_seg_name_to_idx = {}
        self.split_indices = {'training': [], 'testing': [], 'validation': []}
        self.total_count = {'training': 0, 'validation': 0, 'testing': 0}
        self.valid_count = {'training': 0, 'validation': 0, 'testing': 0}
        self.__annotation_file = annotation_file
        self.trimmed = trimmed

    def init(self, trimmed=True):
        """
        load annotation file
        :return:
        """
        data = json.load(open(self.__annotation_file, 'r'))
        self.version = data['version'].split()[1]
        self.__build_label_hierarchy_from_taxonomy(data['taxonomy'])
        self.__create_labels()
        self.__parse_database(data['database'])
        self.__parse_video_segments()
        self.__establish_mapping_class_to_videos()

    def label_idx_to_name(self, idx):
        """
        convert the label index to its name
        :param label_id:
        :return: label name
        """
        return self.labels[idx]

    def label_name_to_idx(self, name):
        """
        convert the label name to its index
        :param name:
        :return:
        """
        if name is None:  # if no label is given (test set), then assign -1
            return -1
        return self.label_idx_table[name]

    def get_num_classes(self):
        return len(self.labels)

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
        return self.video_seg_names[idx]

    def get_video_idx_by_name(self, name):
        """
        get idex of the video by its name
        """
        return self.video_seg_name_to_idx[name]

    def get_frame_paths(self, video_idx, frame_type):
        frame_prefixes = ('img_', 'flow_x_', 'flow_y_')
        frame_paths = []
        video_name = self.video_seg_names[video_idx]
        prefix = frame_prefixes[frame_type]
        frame_folder = os.path.join(self.frame_folder, video_name)
        for root, dirs, files in os.walk(frame_folder):
            files.sort()
            for file_name in files:
                full_path = os.path.join(root, file_name)
                if file_name.startswith(prefix):
                    frame_paths.append(full_path)
        return frame_paths

    def get_minibatch(self, sampler, frame_type):
        """
        get minibatch through sampler
        :param sampler: frame sampler to be used
        :param frame_type: the type of the frame (0 for rgb, 1 for flow_x, 2 for flow_y, 3 for both flow_x and flow_y)
        :return: (n, h, w, c)
        """
        # randomly choose a video
        video_idx = random.randrange(0, len(self.video_seg_names))

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
        frame_paths = np.array(frame_paths)

        # sample mini-batch
        num_frame = len(frame_paths)
        indices = sampler.sample_minibatch_indices(num_frame)
        selected_frame_paths = frame_paths[indices]
        result = []
        for frame_path in selected_frame_paths:
            result.append(cv2.imread(frame_path, cv2.IMREAD_UNCHANGED))
        return np.array(result)

    def vidoe_frame_iterator(self, video_idx, frame_type, batch_size=1, step=1):
        """
        create an iterator that iterate through frames of a video
        :param video_idx: the index of the video
        :param frame_type: the type of the frame (0 for rgb, 1 for flow_x, 2 for flow_y, 3 for both flow_x and flow_y)
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

        # loading data
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

    def get_ordered_label_list(self):
        return self.labels

    def get_training_set(self):
        """
        get video name list and label list of training set
        :return: (video_names, labels)
        """
        indices = self.split_indices['training']
        video_seg_names = [self.video_seg_names[i] for i in indices]
        video_seg_labels = [self.video_seg_labels[i] for i in indices]
        return video_seg_names, video_seg_labels

    def get_validation_set(self):
        """
        get video name list and label list of validation set
        :return: (video_names, labels)
        """
        indices = self.split_indices['validation']
        video_seg_names = [self.video_seg_names[i] for i in indices]
        video_seg_labels = [self.video_seg_labels[i] for i in indices]
        return video_seg_names, video_seg_labels

    def get_testing_set(self):
        """
        get video name list and label list of testing set
        :return: (video_names, labels)
        """
        indices = self.split_indices['testing']
        video_seg_names = [self.video_seg_names[i] for i in indices]
        video_seg_labels = [self.video_seg_labels[i] for i in indices]
        return video_seg_names, video_seg_labels

    def get_video_segments_tags(self, video_name):
        """
        get a list of (start_frame, end_frame) tagging for videos
        :return 
        """
        assert(not self.trimmed)
        meta = self.video_meta[video_name]
        annotations = meta.annotations

        # compute FPS on the fly
        total_frames = len(os.listdir(os.path.join(self.frame_folder, video_name)))
        total_frames /= 3
        assert(total_frames % 3)
        duration = meta.duration
        FPS = float(total_frames)/duration
        result = []
        for annotation in annotations:
            result.append((int(annotation.start * FPS), int(annotation.end * FPS)))
        return result

    def get_nasty_mapping(self):
        """
        nasty hack: get the mapping from data manager's label index to training pipeline's index
        """
        f = open(activity_net_training_class_idx_file)
        cleaned_label_name_to_idx = {}
        for line in f:
            fields = line.strip().split()
            idx = int(fields[1])
            name = fields[0].replace(':','')
            cleaned_label_name_to_idx[name] = idx
        return {i: cleaned_label_name_to_idx[self.labels[i].replace(' ','')] for i in range(200)}

    def __establish_mapping_class_to_videos(self):
        """
        establish the mapping from classes to videos
        """
        self.class_to_videos = {}
        self.class_to_video_paths = {}
        for video_name, video_meta in self.video_meta.iteritems():
            label_name = video_meta.label
            if label_name not in self.label_idx_table:
                continue
            label_id = self.label_name_to_idx(label_name)
            if label_id not in self.class_to_videos:
                self.class_to_videos[label_id] = []
                self.class_to_video_paths[label_id] = []
            self.class_to_videos[label_id].append(video_name)
            self.class_to_video_paths[label_id].append('acnet_video/v_' + video_name + '.mp4')

    def __parse_video_segments(self):
        self.video_seg_names = []
        if self.trimmed:
            for name, meta in self.video_meta.iteritems():
                num_seg = 1 if len(meta.annotations) == 0 else len(meta.annotations)
                l = len(self.video_seg_names)
                self.video_seg_names.extend([name + str(i) for i in range(num_seg)])
                self.video_seg_labels.extend([self.label_name_to_idx(meta.label)] * num_seg)
                self.split_indices[meta.subset].extend([l + i for i in range(num_seg)])
                self.video_seg_name_to_idx.update({name + str(i): l + i for i in range(num_seg)})
        else:
            for name, meta in self.video_meta.iteritems():
                l = len(self.video_seg_names)
                self.video_seg_names.append(name)
                self.video_seg_labels.append(self.label_name_to_idx(meta.label))
                self.split_indices[meta.subset].append(l)
                self.video_seg_name_to_idx[name] = l

    def __build_label_hierarchy_from_taxonomy(self, taxonomy):
        label_raw_infos = []
        for item in taxonomy:
            label_raw_infos.append(LabelRawInfo(item['nodeId'], item['nodeName'], item['parentId'], item['parentName']))
        self.label_hierarchy = LabelHierarchy()
        self.label_hierarchy.build(label_raw_infos)

    def __create_labels(self):
        self.labels = sorted(self.label_hierarchy.get_leaf_labels())
        self.label_idx_table = {self.labels[i]: i for i in range(len(self.labels))}

    def __parse_database(self, database):
        self.video_meta = {}
        self.total_count = {'training': 0, 'validation': 0, 'testing': 0}
        self.valid_count = {'training': 0, 'validation': 0, 'testing': 0}
        for name in database.keys():
            meta = DataActivityNet.__construct_video_meta(name, database[name])
            self.total_count[meta.subset] += 1
            if self.__video_data_validity_check(meta):
                self.video_meta[name] = meta
                self.valid_count[meta.subset] += 1

    def report_video_data_validity(self):
        for k in self.total_count:
            print('{1}/{2} {0} videos are valid.'.format(k, self.valid_count[k], self.total_count[k]))

    def __video_data_validity_check(self, meta):
        if self.trimmed:
            for i in range(len(meta.annotations)):
                video_frame_folders.append(os.path.join(self.frame_folder, meta.name + str(i)))
        else:
            video_frame_folders = [os.path.join(self.frame_folder, meta.name)]

        for folder in video_frame_folders:
            if not (os.path.exists(folder) and len(os.listdir(folder)) != 0):
                return False
        return True

    @staticmethod
    def __construct_video_meta(name, meta_info_description):
        meta = VideoMetaInfo()
        meta.name = name
        meta.duration = meta_info_description['duration']
        meta.subset = meta_info_description['subset']
        meta.resolution = meta_info_description['resolution']
        meta.url = meta_info_description['url']
        meta.annotations = []
        for annotation_desc in meta_info_description['annotations']:
            annotation = VideoMetaInfo.SegmentAnnotation()
            annotation.start = annotation_desc['segment'][0]
            annotation.end = annotation_desc['segment'][1]
            annotation.duration = annotation.end - annotation.start
            annotation.label = annotation_desc['label']
            meta.annotations.append(annotation)
        meta.label = meta.annotations[0].label if len(meta.annotations) != 0 else None
        return meta


if __name__ == '__main__':
    # data_manager = DataActivityNet(annotation_file='/data01/mscvproject/data/ActivityNetTrimflow/.scripts/activity_net.v1-3.min.json',
                                   # frame_folders='/data01/mscvproject/data/ActivityNetUntrimflow/view')
    # data_manager.init()
    # print(data_manager.label_hierarchy)

    # for k, v in data_manager.video_meta.iteritems():
    #      print(v)
    # print(data_manager.get_video_names_by_class_name('Preparing salad'))
    # print(data_manager.get_video_paths_by_class_name('Preparing salad'))

    # print('number of videos: ' + str(len(data_manager.video_meta)))
    # print('number of instances: ' + str(len(data_manager.video_seg_names)))
    # print('training set: ' + str(len(data_manager.get_training_set()[0])))
    # print('validation set: ' + str(len(data_manager.get_validation_set()[0])))
    # print('testing set: ' + str(len(data_manager.get_testing_set()[0])))

    # cnt = 0
    # print(data_manager.video_seg_names[0])
    # for frame_stack in data_manager.vidoe_frame_iterator(video_idx=0, frame_type=3, batch_size=1000):
    #     print(len(frame_stack))
    #    print(frame_stack[0].shape)

    # names, labels = data_manager.get_validation_set()
    # print(len(names))
    # for name, label in zip(names, labels):
    #     print(name, label)

    activitynet_annotation_file='/home/mscvproject/mscvproject/data/tmp/activity_net.v1-3.min.json'
    activitynet_frame_folder='/home/mscvproject/mscvproject/data/ActivityNetUntrimlongtimeflow/view'
    data_manager = DataActivityNet(annotation_file=activitynet_annotation_file,
                                   frame_folders=activitynet_frame_folder,
                                   trimmed = False)
    data_manager.init()
    data_manager.report_video_data_validity()
    names, labels = data_manager.get_validation_set()
    print('number of validation videos: ' + str(len(names)))

    

