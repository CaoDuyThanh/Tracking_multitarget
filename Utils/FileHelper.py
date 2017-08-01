import os
import configparser
import cv2
import numpy
import skimage.transform
import matplotlib.pyplot as plt

def check_file_exist(_file_path,
                     _throw_error = True):
    if not os.path.exists(_file_path):
        print ('%s does not exist ! !' % (_file_path))
        if _throw_error is True:
            assert ('%s does not exist ! !' % (_file_path))
        return False
    return True

def check_path_exist(_path,
                     _throw_error = True):
    if not os.path.isdir(_path):
        print ('%s does not exist ! Make sure you choose right location !' % (_path))
        if _throw_error is True:
            assert ('%s does not exist ! Make sure you choose right location !' % (_path))
        return False
    return True

def create_path(_path):
    os.mkdir(_path)
    return True

def check_not_none(_something,
                   _name,
                   _throw_error = True):
    if _something is None:
        print ('%s can not be none !' % (_name))
        if _throw_error is True:
            assert ('%s can not be none !' % (_name))
        return False
    return True

def get_all_sub_folders_path(_path):
    # Check parameter
    check_not_none(_path, 'path'); check_path_exist(_path)

    all_sub_folders_path = sorted([os.path.join(os.path.abspath(_path), _name + '/') for _name in os.listdir(_path)
                                   if check_path_exist(os.path.join(_path, _name))])
    return all_sub_folders_path

def get_all_files(_path):
    # Check parameters
    check_not_none(_path); check_path_exist(_path)

    all_files_path = sorted([os.path.join(os.path.abspath(_path), _filename) for _filename in os.listdir(_path)
                             if check_file_exist(os.path.join(_path, _filename))])
    return all_files_path

def read_file(_file_path):
    # Check file exist
    check_file_exist(_file_path)

    _file    = open(_file_path)
    all_data = tuple(_file)
    _file.close()

    return all_data

def read_images(_ims_path,
                _batch_size):
    ims           = []
    _num_has_data = 0
    for _im_path in _ims_path:
        if _im_path != '':
            _extension = _im_path.split('.')[-1]
            _im = plt.imread(_im_path, _extension)
            _im = cv2.resize(_im, (300, 300), interpolation = cv2.INTER_LINEAR)
            _im = _im[:, :, [2, 1, 0]]
            _im = numpy.transpose(_im, (2, 0, 1))
            ims.append(_im)
            _num_has_data += 1
    ims = numpy.asarray(ims, dtype = 'float32')

    _mean_img = numpy.asarray([104.000, 117.000, 123.00], dtype = 'float32')
    _mean_img = numpy.reshape(_mean_img, (1, 3, 1, 1))
    ims = (ims - _mean_img)

    if _num_has_data == 0:
        _num_has_data = _batch_size
        ims        = numpy.zeros((_batch_size, 3, 300, 300), dtype ='float32')
    ims = numpy.pad(ims, ((0, _batch_size - _num_has_data), (0, 0), (0, 0), (0, 0)), mode ='constant', constant_values = 0)
    return ims

def read_file_ini(_file_path):
    # Check file exist
    check_file_exist(_file_path)

    config = configparser.ConfigParser()
    config.sections()
    config.read(_file_path)
    return config