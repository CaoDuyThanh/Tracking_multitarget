import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Utils.MOTDataHelper import *
from Models.SSD_300x300.SSDModel import *
from Models.SSD_300x300.SSDVehicleModel import *
from Models.LSTM.DAModel import *
from Utils.DefaultBox import *
from Utils.PlotManager import *

########################################################################################################################
#                                                                                                                      #
#    CONFIGURATIONS SESSION                                                                                            #
#                                                                                                                      #
########################################################################################################################
# TRAINING HYPER PARAMETER
TEST_STATE        = 0
BATCH_SIZE        = 1
DISPLAY_FREQUENCY = 50

# LSTM NETWORK CONFIG
NUM_TRUNCATE      = (1, 1)
NUM_HIDDEN        = 2048
INPUTS_SIZE       = [256 + 4 + 256 + 4 + 256 + 4 + 256 + 4 + 256 + 4 + 256 + 4]
OUTPUTS_SIZE      = [1, 4]

# DATASET CONFIGURATION
DATASET_PATH    = '/media/badapple/Data/PROJECTS/Machine Learning/Dataset/MOT16/'
DATASET_SOURCE  = 'MOT'

# VIDEO_TEST
VIDEO_PATH = '/media/badapple/Data/PROJECTS/Machine Learning/Dataset/Traffic Video/Human2.avi'

# STATE PATH
BEST_PATH  = '../Models/LSTM/DA_Best.pkl'

# TYPE
TEST_VEHICLE = True

# LSTM NETWORK CONFIG
NUM_OBJECT     = 70
DA_INPUT_SIZE  = 70
DA_HIDDEN_SIZE = 512
DA_OUTPUT_SIZE = 70

# GLOBAL VARIABLES
dataset           = None
default_bboxs     = None
SSD_model         = None
DA_model          = None
plot_manager      = None


########################################################################################################################
#                                                                                                                      #
#    LOAD DATASET SESSIONS                                                                                             #
#                                                                                                                      #
########################################################################################################################
def _load_dataset():
    global dataset
    if DATASET_SOURCE == 'MOT':
        dataset = MOTDataHelper(DATASET_PATH)


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
        SSD_model = SSDVehicleModel()
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
    global DA_model
    DA_model = DAModel(da_input_size  = DA_INPUT_SIZE,
                       da_hidden_size = DA_HIDDEN_SIZE,
                       da_output_size = DA_OUTPUT_SIZE + 1)
    # Load old model if exist
    if check_file_exist(BEST_PATH,
                        _throw_error=False) == True:
        print ('|-- Load best DA model !')
        _file = open(BEST_PATH)
        DA_model.load_model(_file)
        _file.close()
        print ('|-- Load best DA model ! Completed !')

########################################################################################################################
#                                                                                                                      #
#    UTILITIES (MANY DIRTY CODES)                                                                                      #
#                                                                                                                      #
########################################################################################################################
def _Euclidean_distance(_bbox_timestamp, _bbox_next_timestamp):
    delta = _bbox_next_timestamp - _bbox_timestamp
    return numpy.sqrt((delta * delta).sum())

def _pairwise_distance(_list_bboxs1,
                       _list_bboxs2):
    distance = numpy.zeros((_list_bboxs1.shape[0], _list_bboxs2.shape[0]), dtype = 'float32')
    for _bbox1_id in range(_list_bboxs1.shape[0]):
        for _bbox2_id in range(_list_bboxs2.shape[0]):
            _bbox1 = _list_bboxs1[_bbox1_id]
            _bbox2 = _list_bboxs2[_bbox2_id]
            distance[_bbox1_id, _bbox2_id] = _Euclidean_distance(_bbox1, _bbox2)
    return distance

def _prepare_bbou(_bbou):
    _bbou = numpy.asarray(_bbou, dtype = 'float32')
    bbou  = numpy.pad(_bbou, ((0, 0), (0, NUM_OBJECT - _bbou.shape[1]), (0, 0)) , mode = 'constant', constant_values = 0)
    return bbou

def _create_id(_selected_id):
    for _new_id in range(1, NUM_OBJECT + 1):
        if _selected_id[_new_id] == 0:
            return _new_id

def _match_bboxs(_pre_bboxs, _id_bboxs, _next_bboxs, _objects_match):
    _cur_bboxs    = numpy.zeros((NUM_OBJECT, 4))
    _cur_id_bboxs = numpy.zeros((NUM_OBJECT,))
    _check       = numpy.ones((NUM_OBJECT,))
    _count       = 0
    _selected_id = numpy.zeros((NUM_OBJECT + 1,))

    for _object_id in range(NUM_OBJECT):
        _object_id_match = _objects_match[_object_id]
        if _object_id_match == NUM_OBJECT:
            continue

        if _id_bboxs[_object_id]:
            if _check[_object_id_match]:
                _check[_object_id_match] = 0
                _cur_bboxs[_count]       = _next_bboxs[_object_id_match,]
                _cur_id_bboxs[_count]    = _id_bboxs[_object_id]
                _count += 1
                _selected_id[int(_id_bboxs[_object_id])] = 1

    for _object_id in range(NUM_OBJECT):
        if _check[_object_id] and \
            _next_bboxs[_object_id][0] > 0.0001 and \
            _next_bboxs[_object_id][1] > 0.0001:
            _new_id               = _create_id(_selected_id)
            _selected_id[_new_id] = 1
            _cur_bboxs[_count]    = _next_bboxs[_object_id,]
            _cur_id_bboxs[_count] = _new_id
            _count += 1

    return _cur_bboxs, _cur_id_bboxs

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
    global dataset, \
           default_bboxs, \
           SSD_model, \
           DA_model, \
           plot_manager
    if TEST_VEHICLE:
        IMAGE_PATH = '/media/badapple/Data/PROJECTS/Machine Learning/Dataset/HCMT16/train/HCMT16-03/img1/%06d.jpg'
    else:
        IMAGE_PATH = '/media/badapple/Data/PROJECTS/Machine Learning/Dataset/MOT16/train/MOT16-05/img1/%06d.jpg'

    # Create id
    _pre_bboxs = numpy.zeros((NUM_OBJECT, 4))
    _id_bboxs  = numpy.zeros((NUM_OBJECT, ))

    for k in range(1, 500):
        _imgs_path = [IMAGE_PATH % k]
        _raw_im    = cv2.imread(_imgs_path[0])
        _imgs      = read_images(_imgs_path, 1)
        _output    = SSD_model.test_network(_imgs)

        _best_bbox = default_bboxs.bbox(_output[0], _output[1])
        _bbou      = default_bboxs.bbou(_best_bbox)
        _bbou      = _prepare_bbou(_bbou)

        _eucl_distance = _pairwise_distance(_pre_bboxs, _bbou[0])
        _eucl_distance = _eucl_distance.reshape((1, NUM_OBJECT, NUM_OBJECT))
        _result        = DA_model.pred_func(_eucl_distance)
        _match         = _result[0]

        _cur_bboxs,\
        _id_bboxs    = _match_bboxs(_pre_bboxs, _id_bboxs, _bbou[0], _match[0])
        _pre_bboxs = _cur_bboxs

        plot_manager.update_main_plot(_raw_im)
        plot_manager.draw_bboxs(_id_bboxs, _cur_bboxs)
        plot_manager.draw_mini_bboxs(_raw_im, _id_bboxs, _cur_bboxs)
        plot_manager.refresh()

if __name__ == '__main__':
    _create_plot_manager()
    _load_dataset()
    _create_SSD_model()
    _create_DA_model()
    _test_model()