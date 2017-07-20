import cv2
from FileHelper import *

def size(_im):
    _shape = _im.shape
    if len(_shape) == 3:
        num_channel = _shape[0]
        im_height   = _shape[1]
        im_width    = _shape[2]
    else:
        num_channel = 1
        im_height   = _shape[0]
        im_width    = _shape[1]
    return [num_channel, im_width, im_height]

def resize(_im,
           _new_size     = None,
           _scale_factor = None):
    if _new_size is not None:
        new_im = cv2.resize(src   = _im,
                           dsize = _new_size)
    else:
        new_im = cv2.resize(src   = _im,
                           dsize = (0, 0),
                           fx    = _scale_factor[0],
                           fy    = _scale_factor[1])
    return new_im

def read_image(_im_path):
    check_file_exist(_im_path)
    return cv2.imread(_im_path)

def convert_image(_im, _format):
    if _format == 'gray':
        im = cv2.cvtColor(_im, cv2.COLOR_BGR2GRAY)
    return im

def scale_size(_old_size, _new_size):
    """
    Create new size from old size which is same aspect ratio with the old size

    Parameters
    ----------
    _old_size: tuple
        (width, heigh) - old size of image

    _new_size: tuple
        (width, height) - new size of image

    Returns
    -------

    """

    if _new_size[0] < 0 and _new_size[1] < 0:
        return _old_size
    else:
        if _new_size[0] < 0:
            ratio = _new_size[1] * 1.0 / _old_size[1]
            _new_size = (int(_old_size[0] * ratio), int(_old_size[1] * ratio))
            return _new_size
        else:
            ratio = _new_size[0] * 1.0 / _old_size[0]
            _new_size = (int(_old_size[0] * ratio), int(_old_size[1] * ratio))
            return _new_size