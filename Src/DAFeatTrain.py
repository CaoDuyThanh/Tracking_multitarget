import numpy
import math
import random
import thread
import timeit
import gzip
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Models.LSTM.DAFeatModel import *
from Utils.MOTDataHelper import *
from random import shuffle
from Models.SSD_300x300.SSDModel import *

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
dataset         = None
feature_factory = None
DAFeat_model    = None
is_pause        = False            # = True --> hold process

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
def _extract_feature():
    global feature_factory
    feature_factory = SSDModel()

########################################################################################################################
#                                                                                                                      #
#    CREATE FEATURE FACTORY TO EXTRACT FEATURES                                                                        #
#                                                                                                                      #
########################################################################################################################
def _create_feature_factory():
    global feature_factory
    feature_factory = SSDModel()

def _extract_feature():
    global feature_factory
    IMAGE_PATH = '/media/badapple/Data/PROJECTS/Machine Learning/Dataset/MOT16/train/MOT16-09/img1/%06d.jpg'
    FEAT_PATH  = '/media/badapple/Data/PROJECTS/Machine Learning/Dataset/MOT16/train/MOT16-09/img1/feature/%06d.pkl'

    for i in range(1, 2000):
        _imgs_path = [IMAGE_PATH % i]
        _imgs      = read_images(_imgs_path, 1)
        _feat      = feature_factory.feat_func(_imgs)
        _feat      = _feat[0].reshape((1940, 256))

        _file = gzip.open(FEAT_PATH % i, 'wb')
        pickle.dump(_feat, _file, 2)
        _file.close()

########################################################################################################################
#                                                                                                                      #
#    CREATE LSTM MODEL FOR TRACKING OBJECTS                                                                            #
#                                                                                                                      #
########################################################################################################################
def _create_DAFeat_model():
    global DAFeat_model
    DAFeat_model = DAFeatModel()

########################################################################################################################
#                                                                                                                      #
#    VALID LSTM MODEL........................                                                                          #
#                                                                                                                      #
########################################################################################################################
def _valid_model(_valid_data):
    return 0

########################################################################################################################
#                                                                                                                      #
#    TRAIN LSTM MODEL........................                                                                          #
#                                                                                                                      #
########################################################################################################################
def _train_model():
    return 0

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
    _extract_feature()
    # _create_DA_model()
    # _train_model()