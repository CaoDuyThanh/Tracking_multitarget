import numpy
import os
import pickle, cPickle
from DataHelper import DatasetHelper
from FileHelper import *

class MOTDataHelper(DatasetHelper):

    def __init__(self,
                 _dataset_path = None
                 ):
        DatasetHelper.__init__(self)

        # Check parameters
        check_not_none(_dataset_path, 'datasetPath'); check_path_exist(_dataset_path)

        # Set parameters
        self.dataset_path  = _dataset_path

        # Config train | test dataset
        self.train_path    = self.dataset_path + 'train/'
        self.test_path     = self.dataset_path + 'test/'
        self.train_folders = get_all_sub_folders_path(self.train_path)
        self.test_folders  = get_all_sub_folders_path(self.test_path)

        # Load data
        self.load_train_file()
        self.load_test_file()

    # ------------------------  CHECK BBOX IN RANGE OF IMAGE  ---------------------------------------------
    def _check_bbox(self, _top_left_x, _top_left_y, _width, _height, _real_width, _real_height):
        _bottom_right_x = _top_left_x + _width
        _bottom_right_y = _top_left_y + _height

        _top_left_x     = max(0, _top_left_x)
        _top_left_y     = max(0, _top_left_y)
        _bottom_right_x = min(_bottom_right_x, _real_width)
        _bottom_right_y = min(_bottom_right_y, _real_height)

        _cx = (_top_left_x + _bottom_right_x) * 1. / 2
        _cy = (_top_left_y + _bottom_right_y) * 1. / 2
        _width    = _bottom_right_x - _top_left_x
        _height   = _bottom_right_y - _top_left_y

        cx     = _cx * 1. / _real_width + 0.000001
        cy     = _cy * 1. / _real_height + 0.000001
        width  = _width * 1. / _real_width
        height = _height * 1. / _real_height

        return [cx, cy, width, height]

    # ------------------------  LOAD TRAIN | VALID | TEST FILES--------------------------------------------
    def get_data_from_one_folder(self, _path):
        # Check folder
        check_path_exist(_path)

        # Config img1 folder exist
        _img1_folder = _path + 'img1/';      check_path_exist(_img1_folder)

        # Create empty dictionary stored all data
        all_data  = dict()

        # Get image info
        _seq_info_path = _path + 'seqinfo.ini'
        _seq_info      = read_file_ini(_seq_info_path)
        _image_info    = dict()
        _image_info['imagewidth'] = float(_seq_info['Sequence']['imWidth'])
        _image_info['imageheight'] = float(_seq_info['Sequence']['imHeight'])
        all_data['imageinfo'] = _image_info

        # Read det.txt
        _det_folder = _path + 'det/';
        if check_path_exist(_det_folder, _throw_error = False):
            _frames        = dict()
            _frames_path   = dict()
            _object_ids    = dict()
            _det_file_path = _det_folder + 'det.txt';    check_file_exist(_det_file_path)
            _all_dets      = read_file(_det_file_path)
            for _det in _all_dets:
                data = _det.split(',')

                _frame_id   = int(data[0])  # Which frame object appears
                _object_id  = float(data[1])  # Number identifies that object as belonging to a tragectory by unique ID
                _top_left_x = float(data[2])  # Topleft corner of bounding box (x)
                _top_left_y = float(data[3])  # Topleft corner of bounding box (y)
                _width      = float(data[4])  # Width of the bounding box
                _height     = float(data[5])  # Height of the bounding box
                _is_ignore  = float(data[6])  # Flag whether this particular instance is ignored in the evaluation
                _type       = float(data[7])  # Identify the type of object
                                              #     Label                ID
                                              # Pedestrian                1
                                              # Person on vehicle         2
                                              # Car                       3
                                              # Bicycle                   4
                                              # Motorbike                 5
                                              # Non motorized vehicle     6
                                              # Static person             7
                                              # Distrator                 8
                                              # Occluder                  9
                                              # Occluder on the ground   10
                                              # Occluder full            11
                                              # Reflection               12
                _im_path = os.path.join(_img1_folder, '300x300/%06d.jpg' % (_frame_id))         # Change this

                if _frame_id not in _frames:
                    _frames[_frame_id] = []
                _cx, _cy, _width, _height = self._check_bbox(_top_left_x, _top_left_y, _width, _height, _image_info['imagewidth'], _image_info['imageheight'])
                _frames[_frame_id].append([_cx, _cy, _width, _height, _is_ignore, _type, _im_path])

                if _object_id not in _object_ids:
                    _object_ids[_object_id] = [_frame_id, _frame_id]
                else:
                    _object_ids[_object_id][1] = _frame_id

                if _frame_id not in _frames_path:
                    _frames_path[_frame_id] = os.path.abspath(os.path.join(_img1_folder + '300x300/', '{0:06}'.format(_frame_id) + '.jpg'))

            det_data = dict()
            det_data['frames']     = _frames
            det_data['objectid']   = _object_ids
            det_data['framespath'] = _frames_path
            all_data['det'] = det_data

        # Read gt.txt
        _gt_folder = _path + 'gt/';
        if check_path_exist(_gt_folder, _throw_error= False):
            _frames       = dict()
            _frames_path  = dict()
            _object_ids   = dict()
            _features     = dict()
            _gt_file_path = _gt_folder + 'gt.txt';      check_file_exist(_gt_file_path)
            _all_gts      = read_file(_gt_file_path)
            for _gt in _all_gts:
                data = _gt.split(',')

                _frame_id   = int(data[0])      # Which frame object appears
                _object_id  = int(data[1])      # Number identifies that object as belonging to a tragectory by unique ID
                _top_left_x = int(data[2])      # Topleft corner of bounding box (x)
                _top_left_y = int(data[3])      # Topleft corner of bounding box (y)
                _width      = int(data[4])      # Width of the bounding box
                _height     = int(data[5])      # Height of the bounding box
                _is_ignore  = int(data[6])      # Flag whether this particular instance is ignored in the evaluation
                _type       = int(data[7])      # Identify the type of object
                _occluder   = float(data[8])    # Occluder of object
                                                #     Label                ID
                                                # Pedestrian                1
                                                # Person on vehicle         2
                                                # Car                       3
                                                # Bicycle                   4
                                                # Motorbike                 5
                                                # Non motorized vehicle     6
                                                # Static person             7
                                                # Distrator                 8
                                                # Occluder                  9
                                                # Occluder on the ground   10
                                                # Occluder full            11
                                                # Reflection               12
                _im_path    = os.path.join(_img1_folder, '300x300/%06d.jpg' % (_frame_id))            # Change this

                if _frame_id not in _frames:
                    _frames[_frame_id] = dict()
                _cx, _cy, _width, _height = self._check_bbox(_top_left_x, _top_left_y, _width, _height, _image_info['imagewidth'], _image_info['imageheight'])
                _frames[_frame_id][_object_id] = [_frame_id, _cx, _cy, _width, _height, _is_ignore, _type, _occluder, _im_path]

                if _object_id not in _object_ids:
                    _object_ids[_object_id] = [_frame_id, _frame_id]
                else:
                    _object_ids[_object_id][1] = _frame_id

                if _frame_id not in _frames_path:
                    _frames_path[_frame_id] = os.path.abspath(os.path.join(_img1_folder + '300x300/', '{0:06}'.format(_frame_id) + '.jpg'))

                if _frame_id not in _features:
                    _feature_path =  os.path.join(_img1_folder, 'feature/%06d.pkl' % (_frame_id))
                    if check_file_exist(_feature_path, _throw_error = False):
                        _file = open(_feature_path)
                        _features[_frame_id] = cPickle.load(_file)
                        _file.close()

            gt_data = dict()
            gt_data['frames']     = _frames
            gt_data['objectid']   = _object_ids
            gt_data['framespath'] = _frames_path
            gt_data['features']   = _features
            all_data['gt'] = gt_data

        return all_data

    def load_train_file(self):
        self.train_data        = dict()
        for _train_folder in self.train_folders:
            _folder_name = _train_folder.split('/')[-2]
            self.train_data[_folder_name] = self.get_data_from_one_folder(_train_folder)

    def load_test_file(self):
        self.test_data         = dict()
        for _test_folder in self.test_folders:
            _folder_name = _test_folder.split('/')[-2]
            self.test_data[_folder_name] = self.get_data_from_one_folder(_test_folder)

    # -----------------------------------------------------------------------------------------------------
    def get_frames_path(self,
                        _folder_name,
                        _start_frame = 1,  # None = Start from the first frame
                        _end_frame   = 10000000  # None = To the end of frame
                        ):
        _frame_id   = _start_frame
        frames_path = []
        while (_frame_id < _end_frame):
            if _frame_id not in self.train_data[_folder_name]['gt']['framespath']:
                break;
            frames_path.append(self.train_data[_folder_name]['gt']['framespath'][_frame_id])
            _frame_id += 1

        return frames_path

    def get_all_object_ids(self):
        if self.data_opts['data_phase'] == 'train':
            _folder_name    = self.data_opts['data_folder_name']
            _folder_type    = self.data_opts['data_folder_type']
            _data           = self.train_data[_folder_name][_folder_type]
            all_object_ids = [_object_id for _object_id in _data['objectid']]
            return all_object_ids

        if self.data_opts['data_phase'] == 'test':
            assert 'Do not support get AllObjectIds from test'

    def get_random_bbox(self):
        if self.data_opts['data_phase'] == 'train':
            _folder_name  = self.data_opts['data_folder_name']
            _folder_type  = self.data_opts['data_folder_type']
            data          = self.train_data[_folder_name][_folder_type]
            _first_frames = data['frames'][1]
            ran_object    = _first_frames[1]
            return data['framespath'], ran_object

        if self.data_opts['data_phase'] == 'test':
            _folder_name   = self.data_opts['data_folder_name']
            _folder_type   = self.data_opts['data_folder_type']
            data           = self.test_data[_folder_name][_folder_type]
            _first_frames  = data['frames'][1]
            ran_object     = _first_frames[1]
            return data['framespath'], ran_object

    def get_all_folder_names(self):
        all_folder_names = []
        if self.data_opts['data_phase'] == 'train':
            all_folder_names = [_folder_name for _folder_name in self.train_data]

        if self.data_opts['data_phase'] == 'test':
            all_folder_names = [_folder_name for _folder_name in self.test_data]

        return all_folder_names

    def get_object_ids_by_frame(self,
                                _frame_id):
        _data_phase  = self.data_opts['data_phase']
        _folder_name = self.data_opts['data_folder_name']
        _folder_type = self.data_opts['data_folder_type']

        if _data_phase == 'det':
            assert 'Get object ids by frame must in gt folder'

        if _data_phase == 'train':
            _data           = self.train_data[_folder_name]
            _frames         = _data[_folder_type]['frames']
            _selected_frame = _frames[_frame_id]

            all_bboxs      = []
            all_object_ids = []
            for (_object_id, _object_data) in _selected_frame.items():
                all_bboxs.append(_object_data[1 : 5])
                all_object_ids.append(_object_id)

            all_bboxs      = numpy.asarray(all_bboxs)
            all_object_ids = numpy.asarray(all_object_ids)
            return all_bboxs, all_object_ids

    def get_sequence_by(self,
                        _occluder_thres = 0.5):
        _data_phase  = self.data_opts['data_phase']
        _folder_name = self.data_opts['data_folder_name']
        _folder_type = self.data_opts['data_folder_type']
        _object_id   = self.data_opts['data_object_id']

        if _data_phase == 'det':
            assert 'Get sequence data must in gt folder'

        if _data_phase == 'train':
            _data          = self.train_data[_folder_name]
            _all_object_id = _data[_folder_type]['objectid']
            frame_start,\
            frame_end      = _all_object_id[_object_id]
            _frames        = _data[_folder_type]['frames']
            ims_path       = []
            bbox           = []
            frame_end = min(frame_end + 1, max(_frames))
            while frame_start <= frame_end:
                _current_frame = _frames[frame_start]
                if _object_id in _current_frame:
                    if _current_frame[_object_id][7] >= _occluder_thres:
                        ims_path.append(_current_frame[_object_id][-1])
                        bbox.append(_current_frame[_object_id][1:5])
                frame_start += 1

            return ims_path, bbox, frame_start

        if self.data_opts['data_phase'] == 'test':
            assert 'Do not support get Sequence from test'

    def get_feature_by(self,
                       _occluder_thres=0.5):
        _data_phase  = self.data_opts['data_phase']
        _folder_name = self.data_opts['data_folder_name']
        _folder_type = self.data_opts['data_folder_type']
        _object_id   = self.data_opts['data_object_id']

        if _data_phase == 'det':
            assert 'Get sequence data must in gt folder'

        if _data_phase == 'train':
            _data          = self.train_data[_folder_name]
            _all_object_id = _data[_folder_type]['objectid']
            frame_start,\
            frame_end      = _all_object_id[_object_id]
            _frames        = _data[_folder_type]['frames']
            _features      = _data[_folder_type]['features']
            features    = []
            bbox        = []
            all_bbox    = []
            ims_path    = []
            frame_end   = min(frame_end + 1, max(_frames))
            while frame_start <= frame_end:
                _current_frame = _frames[frame_start]
                if _object_id in _current_frame:
                    if _current_frame[_object_id][7] >= _occluder_thres:
                        features.append(_features[frame_start])
                        bbox.append(_current_frame[_object_id][1:5])
                        ims_path.append(_current_frame[_object_id][-1])

                        _all_bbox = []
                        for _id, _item in _current_frame.iteritems():
                            _all_bbox.append(_item[1:5])
                        all_bbox.append(_all_bbox)

                frame_start += 1

            return features, bbox, all_bbox, ims_path

        if self.data_opts['data_phase'] == 'test':
            assert 'Do not support get Sequence from test'

    def next_train_batch(self): raise NotImplementedError

    def next_valid_batch(self): raise NotImplementedError

    def next_test_batch(self, _batch_size): raise NotImplementedError