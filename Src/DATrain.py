import numpy
import math
import random
import thread
import timeit
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Models.LSTM.DAModel import *
from Utils.MOTDataHelper import *
from random import shuffle

########################################################################################################################
#                                                                                                                      #
#    CONFIGURATIONS SESSION                                                                                            #
#                                                                                                                      #
########################################################################################################################
# RECORD PATH
RECORD_PATH = 'record.pkl'

# TRAIN | VALID | TEST RATIO
TRAIN_RATIO = 0.94
VALID_RATIO = 0.05
TEST_RATIO  = 0.01

# TRAINING HYPER PARAMETER
TRAIN_STATE        = 1    # Training state
VALID_STATE        = 0    # Validation state
BATCH_SIZE         = 1
NUM_EPOCH          = 300
MAX_ITERATION      = 100000
LEARNING_RATE      = 0.001        # Starting learning rate
DISPLAY_FREQUENCY  = 200;         INFO_DISPLAY = 'LearningRate = %f, Epoch = %d, iteration = %d, cost = %f, bestcost = %f'
SAVE_FREQUENCY     = 4000
VALIDATE_FREQUENCY = 2000

# LSTM NETWORK CONFIG
NUM_OBJECT     = 70
DA_INPUT_SIZE  = 70
DA_HIDDEN_SIZE = 512
DA_OUTPUT_SIZE = 70

# DATASET CONFIGURATION
DATASET_PATH    = '/media/badapple/Data/PROJECTS/Machine Learning/Dataset/MOT16/'
DATASET_SOURCE  = 'MOT'

# SAVE MODEL PATH
SAVE_PATH       = '../Pretrained/Epoch=%d_Iter=%d.pkl'

# LOAD MODEL PATH
LOAD_MODEL_PATH = '../Pretrained/Epoch=%d_Iter=%d.pkl'
START_EPOCH     = 26
START_ITERATION = 16000
START_FOLDER    = ''
START_OBJECTID  = 0

# STATE PATH
STATE_PATH = '../Pretrained/CurrentState.pkl'
BEST_PATH  = '../Pretrained/Best.pkl'

#  GLOBAL VARIABLES
dataset  = None
DA_model = None
is_pause = False            # = True --> hold process

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
#    UTILITIES DIRTY CODE HERE                                                                                         #
#                                                                                                                      #
########################################################################################################################
def _sort_data(_data):
    global dataset
    data = numpy.array(_data)

    _data_sizes = numpy.zeros((data.__len__(),))
    for _idx, _sample in enumerate(data):
        _folder_name = _sample[0]
        _object_id   = int(_sample[1])

        dataset.data_opts['data_folder_name'] = _folder_name
        dataset.data_opts['data_object_id']   = _object_id
        _ims_path, _, _ = dataset.get_sequence_by(occluderThres=0.5)
        _data_sizes[_idx] = _ims_path.__len__()

    _sorted_idx = numpy.argsort(_data_sizes, axis = 0)
    data = data[_sorted_idx]

    return data

def _create_pairwise_distance(_ims_path_sequence, _bboxs_sequence, _object_id, dataset):
    # Length of bboxs sequence
    _num_trun = len(_bboxs_sequence)

    pairwise_distance = numpy.zeros((_num_trun, NUM_OBJECT), dtype = 'float32')
    pairwise_prob     = numpy.zeros((_num_trun, NUM_OBJECT + 1), dtype = 'float32')

    for _trun_idx in range(_num_trun):
        _im_path        = _ims_path_sequence[_trun_idx]

        _bbox_timestamp = _bboxs_sequence[_trun_idx]

        _frame_id            = int(_im_path.split('/')[-1].split('.')[0])
        _frame_id_next       = _frame_id + 1
        _all_bboxs_next,\
        _all_object_ids_next = dataset.get_object_ids_by_frame(_frame_id_next)
        _idx = range(len(_all_bboxs_next))
        shuffle(_idx)
        _all_bboxs_next      = _all_bboxs_next[_idx]
        _all_object_ids_next = _all_object_ids_next[_idx]

        _all_bboxs_next      = _all_bboxs_next[:NUM_OBJECT,]
        _all_object_ids_next = _all_object_ids_next[:NUM_OBJECT,]

        _all_bboxs_next      = numpy.pad(_all_bboxs_next, ((0, NUM_OBJECT - _all_bboxs_next.shape[0]), (0, 0)), mode = 'constant', constant_values = 0)
        _all_object_ids_next = numpy.pad(_all_object_ids_next, ((0, NUM_OBJECT - _all_object_ids_next.shape[0])), mode = 'constant', constant_values = 0)

        # Calculate Euclidean distance between bbox_timestamp to all_bboxs_timmestamp
        _check = True
        for _bbox_id, _bbox_next in enumerate(_all_bboxs_next):
            _eucl_distance = _Euclidean_distance(_bbox_timestamp, _bbox_next)
            _prob          = _prob_distance(_object_id, _all_object_ids_next[_bbox_id])
            if _prob == 1:
                _check = False
            pairwise_distance[_trun_idx][_bbox_id] = _eucl_distance
            pairwise_prob[_trun_idx][_bbox_id]     = _prob

        # If there is no bounding box in next frame with same id, we set the prob at NUM_OBJECT + 1 = 1
        if _check == True:
            pairwise_prob[_trun_idx][NUM_OBJECT] = 1

    return pairwise_distance, pairwise_prob


def _Euclidean_distance(_bbox_timestamp, _bbox_next_timestamp):
    delta = _bbox_next_timestamp - _bbox_timestamp
    return numpy.sqrt((delta * delta).sum())

def _prob_distance(_bbox_id1, _bbox_id2):
    if _bbox_id1 == _bbox_id2:
        return 1
    else:
        return 0


########################################################################################################################
#                                                                                                                      #
#    CREATE LSTM MODEL FOR TRACKING OBJECTS                                                                            #
#                                                                                                                      #
########################################################################################################################
def _create_DA_model():
    global DA_model
    DA_model = DAModel(da_input_size   = DA_INPUT_SIZE,
                       da_hidden_size  = DA_HIDDEN_SIZE,
                       da_output_size  = DA_OUTPUT_SIZE + 1)

########################################################################################################################
#                                                                                                                      #
#    VALID LSTM MODEL........................                                                                          #
#                                                                                                                      #
########################################################################################################################
def _valid_model(_valid_data):
    global BATCH_SIZE

    print ('\n')
    print ('|-- VALIDATE MODEL -----------------------------------------------------------------')
    global is_pause, \
           DA_model

    _num_object_ids = _valid_data.__len__() // BATCH_SIZE
    epoch = 0
    iter  = 0
    costs = []
    for _object_idx in range(_num_object_ids):
        # Create pause event
        if (is_pause):
            print ('Pause validate process....................')
            while (1):
                input = raw_input('Enter anything to resume...........')
                if input == 'r':
                    is_pause = False
                    break;
            print ('Resume !')

        # Object to valid
        _folder_name = _valid_data[_object_idx][0]
        _object_id   = int(_valid_data[_object_idx][1])

        # Get information of batch of object ids
        dataset.data_opts['data_folder_name'] = _folder_name
        dataset.data_opts['data_object_id']   = _object_id
        _ims_path_sequence, _bboxs_sequence, _frame_start = dataset.get_sequence_by(0.5)
        if len(_ims_path_sequence) < 2:
            continue
        _pw_distance, _gt                                 = _create_pairwise_distance(_ims_path_sequence, _bboxs_sequence, _object_id, dataset)
        _pw_distance = _pw_distance.reshape((_pw_distance.shape[0], 1, _pw_distance.shape[1]))
        _gt          = _gt.reshape((_gt.shape[0], 1, _gt.shape[1]))

        _valid_start_time = timeit.default_timer()
        iter += 1
        result = DA_model.valid_func(_pw_distance,
                                     _gt)
        cost = result[0]
        costs.append(cost)
        _valid_end_time  = timeit.default_timer()
        print ('    |-- Valid mini sequence in a batch ! Done ! Train time = %f' % (_valid_end_time - _valid_start_time))

        if iter % DISPLAY_FREQUENCY == 0:
            # Print information of current training in progress
            print (INFO_DISPLAY % (LEARNING_RATE, epoch, iter, numpy.mean(costs), epoch))
    print ('|-- VALIDATE MODEL (DONE) ----------------------------------------------------------')
    print ('\n')

    return numpy.mean(costs)

########################################################################################################################
#                                                                                                                      #
#    TRAIN LSTM MODEL........................                                                                          #
#                                                                                                                      #
########################################################################################################################
def _train_model():
    global dataset,\
           DA_model,\
           is_pause

    # Get all data and devide into TRAIN | VALID | TEST set
    dataset.data_opts['data_phase'] = 'train'
    _all_folder_names = dataset.get_all_folder_names()
    _all_data         = []
    for _folder_name in _all_folder_names:
        dataset.data_opts['data_folder_name'] = _folder_name
        dataset.data_opts['data_folder_type'] = 'gt'
        _all_object_ids = dataset.get_all_object_ids()
        for _object_id in _all_object_ids:
            _all_data.append([_folder_name, _object_id])
    # Shuffle data
    random.seed(123456)
    random.shuffle(_all_data)

    # Divide into TRAIN|VALID|TEST set
    _train_data = _all_data[  0
                            : int(math.floor(_all_data.__len__() * TRAIN_RATIO))]
    _valid_data = _all_data[  int(math.floor(_all_data.__len__() * TRAIN_RATIO))
                            : int(math.floor(_all_data.__len__() * (TRAIN_RATIO + VALID_RATIO)))]
    _test_data  = _all_data[  int(math.floor(_all_data.__len__() * (TRAIN_RATIO + VALID_RATIO)))
                            :]

    # Sort Data based on its length
    _train_data = _sort_data(_train_data)
    _valid_data = _sort_data(_valid_data)

    # Load previous data record
    iter_train_record = []
    cost_train_record = []
    iter_valid_record = []
    cost_valid_record = []
    best_valid_record = 1000
    if check_file_exist(RECORD_PATH, _throw_error=False):
        file = open(RECORD_PATH)
        iter_train_record = pickle.load(file)
        cost_train_record = pickle.load(file)
        iter_valid_record = pickle.load(file)
        cost_valid_record = pickle.load(file)
        best_valid_record = pickle.load(file)
        file.close()
    print ('Load previous record !')

    # Plot training cost
    plt.ion()
    data, = plt.plot(iter_train_record, cost_train_record)

    # Load old model if exist
    if check_file_exist(STATE_PATH,
                        _throw_error=False) == True:
        print ('|-- Load state !')
        file = open(STATE_PATH)
        DA_model.load_state(file)
        file.close()
        print ('|-- Load state ! Completed !')

    costs         = []
    epoch         = START_EPOCH
    iter          = START_ITERATION
    learning_rate = LEARNING_RATE

    # ----- Training state ---------------------------------------------------------------------------------------------
    while iter < MAX_ITERATION:
        _num_object_ids = _train_data.__len__()
        for _object_idx in range(_num_object_ids):
            # Create pause event
            if (is_pause):
                print ('Pause training process...')
                while (1):
                    input = raw_input('Enter anything to resume...')
                    if input == 'r':
                        is_pause = False
                        break;
                    if input == 'l':
                        learning_rate = float(raw_input('Input learning rate = '))
                        print ('New learning rate = %f' % (learning_rate))
                print ('Resume !')

            # Object to train
            _folder_name = _train_data[_object_idx][0]
            _object_id   = int(_train_data[_object_idx][1])

            # Get information of batch of object ids
            dataset.data_opts['data_folder_name'] = _folder_name
            dataset.data_opts['data_object_id']   = _object_id
            _ims_path_sequence, _bboxs_sequence, _frame_start = dataset.get_sequence_by(0.5)

            if len(_ims_path_sequence) < 2:
                continue

            _pw_distance, _gt                                 = _create_pairwise_distance(_ims_path_sequence, _bboxs_sequence, _object_id, dataset)
            _pw_distance = _pw_distance.reshape((_pw_distance.shape[0], 1, _pw_distance.shape[1]))
            _gt          = _gt.reshape((_gt.shape[0], 1, _gt.shape[1]))

            _train_start_time = timeit.default_timer()
            iter  += 1
            result = DA_model.train_func(learning_rate,
                                         _pw_distance,
                                         _gt)
            cost = result[0]
            costs.append(cost)
            _train_end_time = timeit.default_timer()
            # print ('    |-- Train mini sequence in a batch ! Done ! Train time = %f' % (_train_end_time - _train_start_time))

            if iter % DISPLAY_FREQUENCY == 0:
                # Print information of current training in progress
                print (INFO_DISPLAY % (learning_rate, epoch, iter, numpy.mean(costs), best_valid_record))

                # Plot result in progress
                iter_train_record.append(iter)
                cost_train_record.append(numpy.mean(costs))
                data.set_xdata(numpy.append(data.get_xdata(), iter_train_record[-1]))
                data.set_ydata(numpy.append(data.get_ydata(), cost_train_record[-1]))
                yLimit = math.floor(numpy.max(cost_train_record) / 10) * 10 + 10
                plt.axis([iter_train_record[-1] - 10000, iter_train_record[-1], 0, yLimit])
                plt.draw()
                plt.pause(0.05)

                costs = []

            if iter % VALIDATE_FREQUENCY == 0:
                cost_valid = _valid_model(_valid_data= _valid_data)
                iter_valid_record.append(iter)
                cost_valid_record.append(cost_valid)
                print ('Validate model finished! Cost = %f' % (cost_valid))

                if cost_valid < best_valid_record:
                    best_valid_record = cost_valid

                    # Save model
                    file = open(BEST_PATH, 'wb')
                    DA_model.save_model(file)
                    file.close()
                    print ('|-- Save best model !')

            if iter % SAVE_FREQUENCY == 0:
                # Save record
                file = open(RECORD_PATH, 'wb')
                pickle.dump(iter_train_record, file, 0)
                pickle.dump(cost_train_record, file, 0)
                pickle.dump(iter_valid_record, file, 0)
                pickle.dump(cost_valid_record, file, 0)
                pickle.dump(best_valid_record, file, 0)
                file.close()
                print ('|-- Save record !')

                # Save state
                file = open(STATE_PATH, 'wb')
                DA_model.save_state(file)
                file.close()
                print ('|-- Save state !')
        epoch = epoch + 1



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
    _create_DA_model()
    _train_model()