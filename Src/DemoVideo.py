import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Utils.MOTDataHelper import *
from Models.SSD_300x300.SSDModel import *
from Models.SSD_300x300.SSDFeVehicleModel import *
from Models.LSTM.DAFeatModelTruncated import *
from Utils.DefaultBox import *
from Utils.PlotManager import *

########################################################################################################################
#                                                                                                                      #
#    CONFIGURATIONS SESSION                                                                                            #
#                                                                                                                      #
########################################################################################################################
# TRAINING HYPER PARAMETER
TEST_STATE        = 0
NUM_OBJECT        = 70

# LSTM NETWORK CONFIG
DA_EN_INPUT_SIZE  = 128 * 6 + 4
DA_EN_HIDDEN_SIZE = 256

# DATASET CONFIGURATION
DATASET_PATH    = '/media/badapple/Data/PROJECTS/Machine Learning/Dataset/MOT16/'
DATASET_SOURCE  = 'MOT'

# VIDEO_TEST
VIDEO_PATH = '/media/badapple/Data/PROJECTS/Machine Learning/Dataset/Traffic Video/Human2.avi'

# STATE PATH
BEST_PREC_PATH  = '../Pretrained/DAFE_Prec_Best.pkl'

# TYPE
TEST_VEHICLE = True

# GLOBAL VARIABLES
default_bboxs     = None
SSD_model         = None
DAFeat_model      = None
plot_manager      = None

########################################################################################################################
#                                                                                                                      #
#    CREATE SSD MODEL DETECTION                                                                                        #
#                                                                                                                      #
########################################################################################################################
def _create_SSD_model():
    global SSD_model, \
           default_bboxs
    if TEST_VEHICLE:
        _bbox_opts = BBoxOpts()
        _bbox_opts.opts['image_width']  = 300
        _bbox_opts.opts['image_height'] = 300
        _bbox_opts.opts['smin'] = 20
        _bbox_opts.opts['smax'] = 90
        _bbox_opts.opts['layer_sizes']  = [(38, 38),
                                           (19, 19),
                                           (10, 10),
                                           (5, 5),
                                           (3, 3),
                                           (1, 1)]
        _bbox_opts.opts['num_boxs']     = [4, 6, 6, 6, 6, 6]
        _bbox_opts.opts['offset']       = 0.5
        _bbox_opts.opts['steps']        = [8, 16, 32, 64, 100, 300]
        _bbox_opts.opts['min_sizes']    = [30,  60, 114, 168, 222, 276]
        _bbox_opts.opts['max_sizes']    = [ 0, 114, 168, 222, 276, 330]
    else:
        _bbox_opts = BBoxOpts()
        _bbox_opts.opts['image_width']  = 300
        _bbox_opts.opts['image_height'] = 300
        _bbox_opts.opts['smin'] = 20
        _bbox_opts.opts['smax'] = 90
        _bbox_opts.opts['layer_sizes'] = [(38, 38),
                                          (19, 19),
                                          (10, 10),
                                          (5, 5),
                                          (3, 3),
                                          (1, 1)]
        _bbox_opts.opts['num_boxs'] = [4, 6, 6, 6, 4, 4]
        _bbox_opts.opts['offset']   = 0.5
        _bbox_opts.opts['steps']    = [8, 16, 32, 64, 100, 300]
    default_bboxs = DefaultBBox(_bbox_opts)

    print ('|-- Load best SSD model !')
    if TEST_VEHICLE:
        SSD_model = SSDFeVehicleModel()
        SSD_model.load_caffe_model('../Models/SSD_300x300/Vehicle/deploy.prototxt',
                                   '../Models/SSD_300x300/Vehicle/VGG_VOC0712_SSD_300x300_iter_284365.caffemodel')
    else:
        SSD_model = SSDModel()
        SSD_model.load_caffe_model('../Models/SSD_300x300/VOC0712/deploy.prototxt',
                                   '../Models/SSD_300x300/VOC0712/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel')
    print ('|-- Load best SSD model ! Completed !')


########################################################################################################################
#                                                                                                                      #
#    CREATE LSTM MODEL FOR TRACKING OBJECTS                                                                            #
#                                                                                                                      #
########################################################################################################################
def _create_DA_model():
    global DAFeat_model
    DAFeat_model = DAFeatModel(_dafeat_en_input_size  = DA_EN_INPUT_SIZE,
                               _dafeat_en_hidden_size = DA_EN_HIDDEN_SIZE)

    # Load old model if exist
    if check_file_exist(BEST_PREC_PATH, _throw_error = False):
        print ('|-- Load best DA model !')
        _file = open(BEST_PREC_PATH)
        DAFeat_model.load_model(_file)
        _file.close()
        print ('|-- Load best DA model ! Completed !')

########################################################################################################################
#                                                                                                                      #
#    UTILITIES (MANY DIRTY CODES)                                                                                      #
#                                                                                                                      #
########################################################################################################################
def _fast_inbox_check(_anchor_bboxs, _anchor_ids, _ground_truth):
    _num_bboxs     = _anchor_bboxs.shape[0]
    _size_per_bbox = _anchor_bboxs.shape[1]

    _anchor_bboxs = _anchor_bboxs.reshape((_num_bboxs, _size_per_bbox))
    _inter_x      = (_anchor_bboxs[:, 0] - _anchor_bboxs[:, 2] / 2 <= _ground_truth[0]) * \
                    (_anchor_bboxs[:, 0] + _anchor_bboxs[:, 2] / 2 >= _ground_truth[0])
    _inter_y      = (_anchor_bboxs[:, 1] - _anchor_bboxs[:, 3] / 2 <= _ground_truth[1]) * \
                    (_anchor_bboxs[:, 1] + _anchor_bboxs[:, 3] / 2 >= _ground_truth[1])
    _iter_check = _inter_x * _inter_y

    return _anchor_ids[_iter_check]

def _create_fe_pos(_index):
    indices = numpy.zeros((1940,), dtype = bool)
    indices[_index] = True
    return indices

def create_indices_bbox(_all_bbox):
    global default_bboxs
    _anchor_bboxs = default_bboxs.list_default_boxes
    _anchor_ids = default_bboxs.list_id_feature_boxes

    indices_bbox = []
    for _one_bbox in _all_bbox:
        _index    = _fast_inbox_check(_anchor_bboxs, _anchor_ids, _one_bbox)
        _fe_index = _create_fe_pos(_index)
        indices_bbox.append(_fe_index)

    return indices_bbox

def _extract_fe(_feature, _bbox, _index_pos):
    idx1 = (_index_pos[0: 1444] > 0).nonzero()[0]
    rand = numpy.random.randint(size=(1,), low = 0, high = idx1.shape[0])
    idx1 = idx1[rand]

    idx2 = (_index_pos[1444: 1805] > 0).nonzero()[0] + 1444
    rand = numpy.random.randint(size=(1,), low = 0, high = idx2.shape[0])
    idx2 = idx2[rand]

    idx3 = (_index_pos[1805: 1905] > 0).nonzero()[0] + 1805
    rand = numpy.random.randint(size=(1,), low = 0, high = idx3.shape[0])
    idx3 = idx3[rand]

    idx4 = (_index_pos[1905: 1930] > 0).nonzero()[0] + 1905
    rand = numpy.random.randint(size=(1,), low = 0, high = idx4.shape[0])
    idx4 = idx4[rand]

    idx5 = (_index_pos[1930: 1939] > 0).nonzero()[0] + 1930
    rand = numpy.random.randint(size=(1,), low = 0, high = idx5.shape[0])
    idx5 = idx5[rand]

    idx6 = (_index_pos[1939: 1940] > 0).nonzero()[0] + 1939
    rand = numpy.random.randint(size=(1,), low = 0, high = idx6.shape[0])
    idx6 = idx6[rand]

    _fe = _feature[[idx1, idx2, idx3, idx4, idx5, idx6],]
    _fe = _fe.reshape((numpy.prod(_fe.shape),))
    feature = numpy.concatenate((_fe, _bbox), axis = 0)

    return feature

def create_feature(_feature, _all_bbox, _indices_bbox):
    decode_x_batch_sequence = []
    decode_x_sequence       = []
    _fe_negs = []
    for _one_bbox, _index_bbox in zip(_all_bbox, _indices_bbox):
        _fe_neg = _extract_fe(_feature, _one_bbox, _index_bbox)
        _fe_negs.append(_fe_neg)
    decode_x_sequence.append(_fe_negs)
    decode_x_batch_sequence.append(decode_x_sequence)
    decode_x_batch_sequence = numpy.asarray(decode_x_batch_sequence, dtype = 'float32')
    return decode_x_batch_sequence

def _create_id(_selected_id):
    for _new_id in range(1, NUM_OBJECT + 1):
        if _selected_id[_new_id] == 0:
            return _new_id

def _match_bboxs(_match,
                 _encode_h_sequence,
                 _decode_x_batch_sequence,
                 _id_bboxs,
                 _bboxs,
                 _prob):
    encode_x_pos_batch_sequence = []
    encode_h_sequence           = []
    id_bboxs                    = []
    bboxs                       = []

    _selected_ids = numpy.zeros((NUM_OBJECT + 1,))
    _zero_state = numpy.zeros((DA_EN_HIDDEN_SIZE,), dtype = 'float32')
    _check = numpy.ones((len(_bboxs),), dtype = bool)
    _match = _match[0]
    _prob  = _prob[0]
    for _id_match, _one_prob in enumerate(_prob):
        _selected_id = -1
        _best_prob   = 0.95
        for _id_bbox, _bbox in enumerate(_bboxs):
            if _one_prob[_id_bbox] > _best_prob:
                _selected_id = _id_bbox
                _best_prob   = _one_prob[_id_bbox]

        if _selected_id != -1:
            encode_x_pos_batch_sequence.append(_decode_x_batch_sequence[_selected_id,])
            encode_h_sequence.append(_encode_h_sequence[_id_match,])
            id_bboxs.append(_id_bboxs[_id_match])
            bboxs.append(_bboxs[_selected_id])
            _check[_selected_id] = False
            _selected_ids[int(_id_bboxs[_id_match])] = 1

    for _id_check, _one_check in enumerate(_check):
        if _one_check:
            encode_x_pos_batch_sequence.append(_decode_x_batch_sequence[_id_check,])
            encode_h_sequence.append(_zero_state)
            _new_id = _create_id(_selected_ids)
            _selected_ids[_new_id] = 1
            id_bboxs.append(_new_id)
            bboxs.append(_bboxs[_id_check])

    encode_x_pos_batch_sequence = numpy.asarray([encode_x_pos_batch_sequence])
    encode_h_sequence           = numpy.asarray(encode_h_sequence)
    return encode_x_pos_batch_sequence, encode_h_sequence, id_bboxs, bboxs


########################################################################################################################
#                                                                                                                      #
#    CREATE PLOT MANAGER                                                                                               #
#                                                                                                                      #
########################################################################################################################
def _create_plot_manager():
    global plot_manager
    plot_manager = PlotManager()
    print ('|-- Create plot manager ! Completed !')

########################################################################################################################
#                                                                                                                      #
#    TEST LSTM MODEL........................                                                                           #
#                                                                                                                      #
########################################################################################################################
def _test_model():
    global default_bboxs, \
           SSD_model, \
           DAFeat_model, \
           plot_manager
    if TEST_VEHICLE:
        IMAGE_PATH = '/media/badapple/Data/PROJECTS/Machine Learning/Dataset/HCMT16/train/HCMT16-02/img1/%06d.jpg'
    else:
        IMAGE_PATH = '/media/badapple/Data/PROJECTS/Machine Learning/Dataset/MOT16/train/MOT16-05/img1/%06d.jpg'

    # Create id
    _pre_bboxs = numpy.zeros((NUM_OBJECT, 4))
    _id_bboxs  = numpy.zeros((NUM_OBJECT, ))

    _encode_x_pos_batch_sequence = None
    for k in range(1, 500):
        _imgs_path        = [IMAGE_PATH % k]
        _raw_im           = cv2.imread(_imgs_path[0])
        _imgs             = read_images(_imgs_path, 1)
        _feat, _all_bboxs = SSD_model.test_network(_imgs)

        _best_bbox    = default_bboxs.bbox(_all_bboxs[0], _all_bboxs[1])
        _bbou         = default_bboxs.bbou(_best_bbox)[0]
        _indices_bbox = create_indices_bbox(_bbou)

        _decode_x_sequence = create_feature(_feat, _bbou, _indices_bbox)

        if k == 1:
            _id_bboxs  = range(1, len(_bbou) + 1)
            _cur_bboxs = _bbou

            _encode_x_pos_batch_sequence = _decode_x_sequence[0, ]
            _encode_h_sequence           = numpy.zeros((_encode_x_pos_batch_sequence.shape[1], DA_EN_HIDDEN_SIZE), dtype='float32')
            # _encode_x_pos_batch_sequence = _decode_x_sequence[0, ]
            # _encode_x_pos_batch_sequence = _encode_x_pos_batch_sequence[0, 0, ]
            # _encode_x_pos_batch_sequence = numpy.asarray([[_encode_x_pos_batch_sequence]])
            # _encode_h_sequence           = numpy.zeros((1, DA_EN_HIDDEN_SIZE), dtype='float32')
        else:
            _decode_x_batch_sequence = numpy.repeat(_decode_x_sequence, _encode_x_pos_batch_sequence.shape[1], 1)
            _result = DAFeat_model.pred_func(_encode_x_pos_batch_sequence,
                                             _encode_h_sequence,
                                             _decode_x_batch_sequence)
            _match             = _result[0]
            _encode_h_sequence = _result[1]
            _prob              = _result[2]

            _encode_x_pos_batch_sequence, \
            _encode_h_sequence, \
            _id_bboxs, \
            _cur_bboxs = _match_bboxs(_match,
                                      _encode_h_sequence,
                                      _decode_x_batch_sequence[0, 0, ],
                                      _id_bboxs, _bbou,
                                      _prob)

        plot_manager.update_main_plot(_raw_im)
        plot_manager.draw_bboxs(_id_bboxs, _cur_bboxs)
        plot_manager.draw_mini_bboxs(_raw_im, _id_bboxs, _cur_bboxs)
        plot_manager.refresh()

if __name__ == '__main__':
    _create_plot_manager()
    _create_SSD_model()
    _create_DA_model()
    _test_model()