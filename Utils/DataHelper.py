import cv2
import numpy

class DatasetHelper:
    def __init__(self):
        self.data_opts = {}

        # Default setting for retrieve data
        self.data_opts['data_phase']       = 'train'     # Setting 'train' for training
                                                        #         'test' for testing
        self.data_opts['data_folder_name'] = ''
        self.data_opts['data_folder_type'] = 'gt'        # Setting 'gt' for tracking
                                                        # Setting 'det' for detection
        self.data_opts['data_object_id']   = '0'

    # ------------------------  LOAD TRAIN | VALID | TEST FILES--------------------------------------------
    def load_train_file(self): raise NotImplementedError

    def load_valid_file(self): raise NotImplementedError

    def load_test_file(self): raise NotImplementedError

    # -----------------------------------------------------------------------------------------------------
    def next_train_batch(self): raise NotImplementedError

    def next_valid_batch(self): raise NotImplementedError

    def next_test_batch(self, _batch_size): raise NotImplementedError
