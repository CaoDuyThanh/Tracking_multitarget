import numpy
import math
import random
import thread
import timeit
import gzip
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import multiprocessing
from joblib import Parallel, delayed
from Models.SSD_300x300.SSDFeVehicleModel import *
from Models.LSTM.DAFeatModel import *
from Utils.MOTDataHelper import *
from Utils.DefaultBox import *
from Utils.BBoxHelper import *
from random import shuffle

########################################################################################################################
#                                                                                                                      #
#    CONFIGURATIONS SESSION                                                                                            #
#                                                                                                                      #
########################################################################################################################
# RECORD PATH
RECORD_PATH = 'DAFE_record.pkl'

# TRAIN | VALID | TEST RATIO
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO  = 0.1

# TRAINING HYPER PARAMETER
TRAIN_STATE        = 1    # Training state
VALID_STATE        = 0    # Validation state
BATCH_SIZE         = 1
NUM_EPOCH          = 300
MAX_ITERATION      = 100000
LEARNING_RATE      = 0.001        # Starting learning rate
DISPLAY_FREQUENCY  = 10;         INFO_DISPLAY = '\r%sLearning rate = %f - Epoch = %d - Iter = %d - Cost = %f - Best cost = %f - Mags = %f'
SAVE_FREQUENCY     = 40
VALIDATE_FREQUENCY = 40

# LSTM NETWORK CONFIG
NUM_OBJECT        = 70
DA_EN_INPUT_SIZE  = 128 + 4
DA_EN_HIDDEN_SIZE = 512
DA_DE_INPUT_SIZE  = (128 + 4) * NUM_OBJECT
DA_DE_OUTPUT_SIZE = NUM_OBJECT + 1

# DATASET CONFIGURATION
DATASET_PATH    = '/media/badapple/Data/PROJECTS/Machine Learning/Dataset/HCMT16/'
DATASET_SOURCE  = 'HCMT'

# SAVE MODEL PATH
SAVE_PATH       = '../Pretrained/Epoch=%d_Iter=%d.pkl'

# LOAD MODEL PATH
LOAD_MODEL_PATH = '../Pretrained/Epoch=%d_Iter=%d.pkl'
START_EPOCH     = 0
START_ITERATION = 0
START_FOLDER    = ''
START_OBJECTID  = 0

# STATE PATH
STATE_PATH = '../Pretrained/DAFE_CurrentState.pkl'
BEST_PATH  = '../Pretrained/DAFE_Best.pkl'

#  GLOBAL VARIABLES
dataset         = None
feature_factory = None
DAFeat_model    = None
default_bboxs   = None
is_pause        = False            # = True --> hold process

########################################################################################################################
#                                                                                                                      #
#    LOAD DATASET SESSIONS                                                                                             #
#                                                                                                                      #
########################################################################################################################
def _load_dataset():
    global dataset
    if DATASET_SOURCE == 'HCMT':
        dataset = MOTDataHelper(DATASET_PATH)
    print ('|-- Load dataset ! Completed !')

########################################################################################################################
#                                                                                                                      #
#    UTILITIES DIRTY CODE HERE                                                                                         #
#                                                                                                                      #
########################################################################################################################
def _extract_feature():
    global feature_factory
    feature_factory = SSDFeVehicleModel()
    feature_factory.load_caffe_model('../Models/SSD_300x300/Vehicle/deploy.prototxt',
                                     '../Models/SSD_300x300/Vehicle/VGG_VOC0712_SSD_300x300_iter_284365.caffemodel')

    IMAGE_PATH = '/media/badapple/Data/PROJECTS/Machine Learning/Dataset/HCMT16/train/HCMT16-02/img1/%06d.jpg'
    FILE_SAVE  = '/media/badapple/Data/PROJECTS/Machine Learning/Dataset/HCMT16/train/HCMT16-02/img1/feature/%06d.pkl'

    for k in range(1, 1600):
        print (k)
        _imgs_path = [IMAGE_PATH % k]
        _imgs      = read_images(_imgs_path, 1)

        _feature = feature_factory.feat_func(_imgs)
        _feature = _feature[0].reshape((1940, 128))

        _file = open(FILE_SAVE % (k), 'wb')
        pickle.dump(_feature, file, 2)
        _file.close()

def _mapping_feature(_feature, _bbox):
    global default_bboxs

    _anchor_bboxs = default_bboxs.list_default_boxes
    _anchor_ids   = default_bboxs.list_id_feature_boxes

    _ratio  = _fast_iou(_anchor_bboxs, _bbox)
    _choice_ratio = _ratio >= 0.45
    if numpy.sum(_choice_ratio) == 0:
        feature = _feature[_anchor_ids[numpy.argmax(_ratio)],]
    else:
        _choice_feat = _feature[_anchor_ids[_choice_ratio]]
        numpy.random.shuffle(_choice_feat)
        feature = _choice_feat[0,]

    return feature

# def _mapping_feature1(_feature, _bbox, _ims_path):
#     global default_bboxs
#
#     _anchor_bboxs = default_bboxs.list_default_boxes
#     _anchor_ids   = default_bboxs.list_id_feature_boxes
#
#     _ratio  = _fast_iou(_anchor_bboxs, _bbox)
#     return _anchor_bboxs[numpy.argmax(_ratio),]

def _fast_iou(_anchor_bboxs, _ground_truth):
    _num_bboxs     = _anchor_bboxs.shape[0]
    _size_per_bbox = _anchor_bboxs.shape[1]

    _anchor_bboxs = _anchor_bboxs.reshape((_num_bboxs, _size_per_bbox))
    _zeros        = numpy.zeros((_num_bboxs))
    _inter_x      = numpy.maximum(_zeros,
                                 numpy.minimum(
                                     _anchor_bboxs[:, 0] + _anchor_bboxs[:, 2] / 2,
                                     _ground_truth [0] + _ground_truth [2] / 2
                                 ) -
                                 numpy.maximum(
                                     _anchor_bboxs[:, 0] - _anchor_bboxs[:, 2] / 2,
                                     _ground_truth [0] - _ground_truth [2] / 2
                                 ))
    _inter_y      = numpy.maximum(_zeros,
                                 numpy.minimum(
                                     _anchor_bboxs[:, 1] + _anchor_bboxs[:, 3] / 2,
                                     _ground_truth [1] + _ground_truth [3] / 2
                                 ) -
                                 numpy.maximum(
                                     _anchor_bboxs[:, 1] - _anchor_bboxs[:, 3] / 2,
                                     _ground_truth [1] - _ground_truth [3] / 2
                                 ))
    _iter_area = _inter_x * _inter_y

    _area1 = _anchor_bboxs[:, 2] * _anchor_bboxs[:, 3]
    _area2 = _ground_truth [2] * _ground_truth [3]

    ratio = _iter_area / (_area1 + _area2 - _iter_area)
    return ratio

def _same_bbox_fast(_bboxs, _bbox):
    _list = (abs(_bboxs[:, 0] - _bbox[0]) <= 0.0001) * \
             (abs(_bboxs[:, 1] - _bbox[1]) <= 0.0001) * \
              (abs(_bboxs[:, 2] - _bbox[2]) <= 0.0001) * \
               (abs(_bboxs[:, 3] - _bbox[3]) <= 0.0001)

    if numpy.sum(_list) == 1:
        return numpy.argmax(_list)
    else:
        return -1


# def Draw(_ims_path_sequence, _bboxs, _all_bboxs):
#     fig, ax = plt.subplots(1)
#     ab = None
#
#     for _idx, _im_path in enumerate(_ims_path_sequence):
#         _raw_im = cv2.imread(_im_path)
#
#         if ab == None:
#             ab = ax.imshow(_raw_im)
#         else:
#             ab.set_data(_raw_im)
#
#         _bbox = _bboxs[_idx]
#         cx = _bbox[0]
#         cy = _bbox[1]
#         w  = _bbox[2]
#         h  = _bbox[3]
#
#         H, W, _ = _raw_im.shape
#         rect0 = patches.Rectangle(((cx - w / 2) * W, (cy - h / 2) * H), w * W, h * H, linewidth=1, edgecolor='r', facecolor='none')
#         # Add the patch to the Axes
#         ax.add_patch(rect0)
#         plt.show()
#         plt.axis('off')
#         plt.pause(5)
#         rect0.remove()
#
#         _all_bbox = _all_bboxs[_idx]
#         rect = []
#         for _bbox in _all_bbox:
#             cx = _bbox[0]
#             cy = _bbox[1]
#             w = _bbox[2]
#             h = _bbox[3]
#
#             H, W, _ = _raw_im.shape
#             rect0 = patches.Rectangle(((cx - w / 2) * W, (cy - h / 2) * H), w * W, h * H, linewidth=1, edgecolor='r',
#                                       facecolor='none')
#             # Add the patch to the Axes
#             ax.add_patch(rect0)
#             rect.append(rect0)
#         plt.show()
#         plt.axis('off')
#         plt.pause(5)
#
#         for _rec in rect:
#             _rec.remove()
#
# def Draw1(_ims_path_sequence, _bboxs, _ratio, _ids):
#     fig, ax = plt.subplots(1)
#     ab = None
#
#     for _idx, _im_path in enumerate(_ims_path_sequence):
#         _raw_im = cv2.imread(_im_path)
#
#         if ab == None:
#             ab = ax.imshow(_raw_im)
#         else:
#             ab.set_data(_raw_im)
#
#         count = 0
#         for _bbox, _rati in zip(_bboxs, _ratio):
#             cx = _bbox[0]
#             cy = _bbox[1]
#             w  = _bbox[2]
#             h  = _bbox[3]
#
#             H, W, _ = _raw_im.shape
#             rect0 = patches.Rectangle(((cx - w / 2) * W, (cy - h / 2) * H), w * W, h * H, linewidth=1, edgecolor='r', facecolor='none')
#             # Add the patch to the Axes
#             ax.add_patch(rect0)
#             print (_rati, count, _ids[count])
#             plt.show()
#             plt.axis('off')
#             plt.pause(0.01)
#             rect0.remove()
#             count += 1

def _one_frame(_feature, _frame_bboxs):
    _frame_feat_bboxs = []
    for _bbox_id, _bbox in enumerate(_frame_bboxs):
        _frame_feat_bboxs.append(_mapping_feature(_feature, _bbox))
    return _frame_feat_bboxs

def _get_data_sequence(_parallel,
                       _dataset,
                       _folder_name,
                       _object_id):
    _dataset.data_opts['data_folder_name']   = _folder_name
    _dataset.data_opts['data_object_id']     = _object_id
    _features, _bboxs, _all_bboxs, _ims_path = _dataset.get_feature_by(_occluder_thres = 0.5)

    # num_cores = multiprocessing.cpu_count()
    #
    # _feat_bboxs = _parallel(delayed(_mapping_feature)(_feature, _bbox) for _feature, _bbox in zip(_features, _bboxs))
    # _feat_bboxs = numpy.asarray(_feat_bboxs, dtype = 'float32')
    # _encode_x_sequence = numpy.concatenate((_feat_bboxs, _bboxs), axis = 1)
    #
    # _feat_all_bboxs = _parallel(delayed(_one_frame)(_feature, _frame_bboxs) for _feature, _frame_bboxs in zip(_features, _all_bboxs))

    _feat_bboxs = []
    for _feature, _bbox in zip(_features, _bboxs):
        _feat_bboxs.append(_mapping_feature(_feature, _bbox))
    _feat_bboxs = numpy.asarray(_feat_bboxs, dtype='float32')
    _encode_x_sequence = numpy.concatenate((_feat_bboxs, _bboxs), axis=1)

    _feat_all_bboxs = []
    for _feature, _frame_bboxs in zip(_features, _all_bboxs):
        _frame_feat_bboxs = []
        for _bbox_id, _bbox in enumerate(_frame_bboxs):
            _frame_feat_bboxs.append(_mapping_feature(_feature, _bbox))
        _feat_all_bboxs.append(_frame_feat_bboxs)

    _decode_x_sequence = []
    _decode_y_sequence = []
    for _id, (_frame_feat_bboxs, _frame_bboxs) in enumerate(zip(_feat_all_bboxs, _all_bboxs)):
        _frame_feat_bboxs = numpy.asarray(_frame_feat_bboxs, dtype = 'float32')
        _frame_bboxs      = numpy.asarray(_frame_bboxs, dtype = 'float32')
        _num_bboxs        = _frame_bboxs.shape[0]
        _decode_x_frame   = numpy.concatenate((_frame_feat_bboxs, _frame_bboxs), axis = 1)
        _decode_x_frame   = numpy.pad(_decode_x_frame, ((0, NUM_OBJECT - _num_bboxs), (0, 0)), mode ='constant', constant_values = 0)

        _decode_y_frame   = numpy.zeros((NUM_OBJECT + 1, ), dtype = 'int32')

        _selected_bbox_id = _same_bbox_fast(_frame_bboxs, _bboxs[_id])
        if _selected_bbox_id == -1:
            _decode_y_frame[-1] = 1
        else:
            _decode_y_frame[_selected_bbox_id] = 1

        _ids = range(len(_decode_x_frame))
        numpy.random.shuffle(_ids)
        _decode_x_frame                = _decode_x_frame[_ids,]
        _decode_y_frame[:NUM_OBJECT, ] = _decode_y_frame[_ids,]

        _decode_x_sequence.append(_decode_x_frame)
        _decode_y_sequence.append(_decode_y_frame)

    _encode_x_sequence = numpy.asarray(_encode_x_sequence, dtype = 'float32')
    _decode_x_sequence = numpy.asarray(_decode_x_sequence, dtype = 'float32')
    _decode_y_sequence = numpy.asarray(_decode_y_sequence, dtype = 'float32')

    return _encode_x_sequence, _decode_x_sequence, _decode_y_sequence


########################################################################################################################
#                                                                                                                      #
#    CREATE FEATURE FACTORY TO EXTRACT FEATURES                                                                        #
#                                                                                                                      #
########################################################################################################################
def _create_feature_factory():
    global feature_factory, \
           default_bboxs
    # feature_factory = SSDFeVehicleModel()
    # feature_factory.load_caffe_model('../Models/SSD_300x300/Vehicle/deploy.prototxt',
    #                                  '../Models/SSD_300x300/Vehicle/VGG_VOC0712_SSD_300x300_iter_284365.caffemodel')
    # print('|-- Create SSD model ! Completed !')

    _bbox_opts = BBoxOpts()
    _bbox_opts.opts['image_width']  = 300
    _bbox_opts.opts['image_height'] = 300
    _bbox_opts.opts['smin']         = 20
    _bbox_opts.opts['smax']         = 90
    _bbox_opts.opts['layer_sizes']  = [(38, 38),
                                       (19, 19),
                                       (10, 10),
                                       (5, 5),
                                       (3, 3),
                                       (1, 1)]
    _bbox_opts.opts['num_boxs']  = [4, 6, 6, 6, 6, 6]
    _bbox_opts.opts['offset']    = 0.5
    _bbox_opts.opts['steps']     = [8, 16, 32, 64, 100, 300]
    _bbox_opts.opts['min_sizes'] = [30, 60, 114, 168, 222, 276]
    _bbox_opts.opts['max_sizes'] = [0, 114, 168, 222, 276, 330]
    default_bboxs = DefaultBBox(_bbox_opts)
    print('|-- Create default bboxs ! Completed !')


########################################################################################################################
#                                                                                                                      #
#    CREATE LSTM MODEL FOR TRACKING OBJECTS                                                                            #
#                                                                                                                      #
########################################################################################################################
def _create_DAFeat_model():
    global DAFeat_model
    DAFeat_model = DAFeatModel(_dafeat_en_input_size  = DA_EN_INPUT_SIZE,
                               _dafeat_en_hidden_size = DA_EN_HIDDEN_SIZE,
                               _dafeat_de_input_size  = DA_DE_INPUT_SIZE,
                               _dafeat_de_output_size = DA_DE_OUTPUT_SIZE)

########################################################################################################################
#                                                                                                                      #
#    VALID LSTM MODEL........................                                                                          #
#                                                                                                                      #
########################################################################################################################
def _valid_model(_dataset, _valid_data):
    with Parallel(n_jobs = 16, backend = "threading") as _parallel:
        _num_valid_data     = len(_valid_data)
        _all_valid_data_ids = range(_num_valid_data)
        _num_valided_data   = 0
        _iter  = 0
        _epoch = 0
        _costs = []
        for _valid_data_idx in _all_valid_data_ids:
            # Get batch of object id
            _valid_sample = _valid_data[_valid_data_idx]
            _folder_name  = _valid_sample[0]
            _object_id    = int(_valid_sample[1])
            _num_valided_data += 1

            # Print information
            print '\r    |-- Validate %d / %d objects - Folder name = %s - Object id = %d' % (_num_valided_data, _num_valid_data, _folder_name, _object_id),

            # Get information of batch of object ids
            _encode_x_sequence, \
            _decode_x_sequence, \
            _decode_y_sequence = _get_data_sequence(_parallel    = _parallel,
                                                    _dataset     = _dataset,
                                                    _folder_name = _folder_name,
                                                    _object_id   = _object_id)

            _num_time_step = _encode_x_sequence.shape[0]
            if _num_time_step < 3:
                continue

            _encode_x_sequence = _encode_x_sequence[0: _num_time_step - 1, ]
            _decode_x_sequence = _decode_x_sequence[1:, ]
            _decode_y_sequence = _decode_y_sequence[1:, ]

            _encode_x_sequence = _encode_x_sequence.reshape(
                (_encode_x_sequence.shape[0], BATCH_SIZE, _encode_x_sequence.shape[1]))
            _decode_x_sequence = _decode_x_sequence.reshape(
                (_decode_x_sequence.shape[0], BATCH_SIZE, _decode_x_sequence.shape[1] * _decode_x_sequence.shape[2]))
            _decode_y_sequence = _decode_y_sequence.reshape(
                (_decode_y_sequence.shape[0], BATCH_SIZE, _decode_y_sequence.shape[1]))

            _iter += 1
            result = DAFeat_model.valid_func(_encode_x_sequence,
                                             _decode_x_sequence,
                                             _decode_y_sequence)
            _train_end_time = timeit.default_timer()

            # Temporary save info
            _costs.append(result[0])

            if _iter % DISPLAY_FREQUENCY == 0:
                # Print information of current training in progress
                print (INFO_DISPLAY % ('    |-- ', 0, _epoch, _iter, numpy.mean(_costs), 0, 0))
        return numpy.mean(_costs)

########################################################################################################################
#                                                                                                                      #
#    TRAIN LSTM MODEL........................                                                                          #
#                                                                                                                      #
########################################################################################################################
def _train_model():
    global dataset, \
           feature_factory, \
           DAFeat_model
    with Parallel(n_jobs = 16, backend="threading") as _parallel:
        # ===== Prepare dataset =====
        # ----- Get all data and devide into TRAIN | VALID | TEST set -----
        dataset.data_opts['data_phase'] = 'train'
        _all_folder_names = dataset.get_all_folder_names()
        _all_data         = []
        for _folder_name in _all_folder_names:
            dataset.data_opts['data_folder_name'] = _folder_name
            dataset.data_opts['data_folder_type'] = 'gt'
            _all_object_ids = dataset.get_all_object_ids()
            for _object_id in _all_object_ids:
                _all_data.append([_folder_name, _object_id])
        # ----- Shuffle data -----
        random.seed(123456)
        random.shuffle(_all_data)

        # ----- Divide into TRAIN|VALID|TEST set -----
        _idx_ratio = [0,
                      int(len(_all_data) * TRAIN_RATIO),
                      int(len(_all_data) * (TRAIN_RATIO + VALID_RATIO)),
                      len(_all_data)]
        train_data = _all_data[_idx_ratio[0] : _idx_ratio[1]]
        valid_data = _all_data[_idx_ratio[1] : _idx_ratio[2]]
        test_data  = _all_data[_idx_ratio[2] : _idx_ratio[3]]

        # ===== Load data record =====
        print ('|-- Load previous record !')
        iter_train_record = []
        cost_train_record = []
        iter_valid_record = []
        cost_valid_record = []
        best_valid_cost   = 10000
        if check_file_exist(RECORD_PATH, _throw_error = False):
            _file = open(RECORD_PATH)
            iter_train_record = pickle.load(_file)
            cost_train_record = pickle.load(_file)
            iter_valid_record = pickle.load(_file)
            cost_valid_record = pickle.load(_file)
            best_valid_cost   = pickle.load(_file)
            _file.close()
        print ('|-- Load previous record ! Completed !')

        # ===== Load state =====
        print ('|-- Load state !')
        if check_file_exist(STATE_PATH, _throw_error = False):
            _file = open(STATE_PATH)
            DAFeat_model.load_state(_file)
            _file.close()
        print ('|-- Load state ! Completed')

        # ===== Training start =====
        # ----- Temporary record -----
        _costs = []
        _mags  = []
        _epoch = START_EPOCH
        _iter  = START_ITERATION
        _learning_rate = LEARNING_RATE

        # ----- Train -----
        for _epoch in xrange(START_EPOCH, NUM_EPOCH):
            _num_train_data     = len(train_data)
            _all_train_data_ids = range(_num_train_data)
            _num_trained_data   = 0
            for _train_data_idx in _all_train_data_ids:
                # Get batch of object id
                _train_sample = train_data[_train_data_idx]
                _folder_name  =     _train_sample[0]
                _object_id    = int(_train_sample[1])
                _num_trained_data += 1

                # Print information
                print '\r|-- Trained %d / %d objects - Folder name = %s - Object id = %d' % (_num_trained_data, _num_train_data, _folder_name, _object_id),

                # Get information of batch of object ids
                _encode_x_sequence, \
                _decode_x_sequence, \
                _decode_y_sequence = _get_data_sequence(_parallel    = _parallel,
                                                        _dataset     = dataset,
                                                        _folder_name = _folder_name,
                                                        _object_id   = _object_id)
                _num_time_step = _encode_x_sequence.shape[0]
                if _num_time_step < 3:
                    continue

                _encode_x_sequence = _encode_x_sequence[0 : _num_time_step - 1, ]
                _decode_x_sequence = _decode_x_sequence[1 : , ]
                _decode_y_sequence = _decode_y_sequence[1 : , ]

                _encode_x_sequence = _encode_x_sequence.reshape((_encode_x_sequence.shape[0], BATCH_SIZE, _encode_x_sequence.shape[1]))
                _decode_x_sequence = _decode_x_sequence.reshape((_decode_x_sequence.shape[0], BATCH_SIZE, _decode_x_sequence.shape[1] * _decode_x_sequence.shape[2]))
                _decode_y_sequence = _decode_y_sequence.reshape((_decode_y_sequence.shape[0], BATCH_SIZE, _decode_y_sequence.shape[1]))

                # Update
                _train_start_time = timeit.default_timer()
                _iter += 1
                result = DAFeat_model.train_func(_learning_rate,
                                                 _encode_x_sequence,
                                                 _decode_x_sequence,
                                                 _decode_y_sequence)
                _train_end_time = timeit.default_timer()

                # Temporary save info
                _costs.append(result[0])
                _mags.append(result[1])

                if _iter % DISPLAY_FREQUENCY == 0:
                    # Print information of current training in progress
                    print (INFO_DISPLAY % ('|-- ', _learning_rate, _epoch, _iter, numpy.mean(_costs), best_valid_cost, numpy.mean(_mags)))
                    _costs = []
                    _mags  = []

                if _iter % VALIDATE_FREQUENCY == 0:
                    print ('------------------- Validate Model -------------------')
                    _cost_valid = _valid_model(_dataset    = dataset,
                                               _valid_data = valid_data)
                    iter_valid_record.append(_iter)
                    cost_valid_record.append(_cost_valid)
                    print ('\n+ Validate model finished! Cost = %f' % (_cost_valid))
                    print ('------------------- Validate Model (Done) -------------------')

                    # Save model if its cost better than old one
                    if (_cost_valid < best_valid_cost):
                        best_valid_cost = _cost_valid

                        # Save best model
                        _file = open(BEST_PATH, 'wb')
                        DAFeat_model.save_model(_file)
                        _file.close()
                        print ('+ Save best model ! Complete !')

                if _iter % SAVE_FREQUENCY == 0:
                    # Save record
                    _file = open(RECORD_PATH, 'wb')
                    pickle.dump(iter_train_record, _file, 2)
                    pickle.dump(cost_train_record, _file, 2)
                    pickle.dump(iter_valid_record, _file, 2)
                    pickle.dump(cost_valid_record, _file, 2)
                    pickle.dump(best_valid_cost, _file, 2)
                    _file.close()
                    print ('+ Save record ! Completed !')

                    # Save state
                    _file = open(STATE_PATH, 'wb')
                    DAFeat_model.save_state(_file)
                    _file.close()
                    print ('+ Save state ! Completed !')

########################################################################################################################
#                                                                                                                      #
#    EVENT HANDLING..........................                                                                          #
#                                                                                                                      #
########################################################################################################################
def _wait_event(threadName):
    global is_pause
    print ('Start pause event')
    while (1):
        input = raw_input()
        if input == 'p':
            is_pause = True

def _create_pause_event():
    try:
        thread.start_new_thread(_wait_event, ('Thread wait',))
    except:
        print ('Error: unable to start thread')

if __name__ == '__main__':
    # _create_pause_event()
    _load_dataset()
    _create_feature_factory()
    # _create_DAFeat_model()
    _train_model()