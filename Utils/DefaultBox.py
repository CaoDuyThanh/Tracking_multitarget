import math
import numpy

class BBoxOpts():
    def __init__(self):
        self.opts = {}

        self.opts['image_width']  = 800
        self.opts['image_height'] = 600
        self.opts['smin']         = 20
        self.opts['smax']         = 90
        self.opts['layer_sizes']  = None
        self.opts['num_boxs']     = None
        self.opts['offset']       = 0.5
        self.opts['steps']        = None
        self.opts['min_sizes']    = None
        self.opts['max_sizes']    = None

class DefaultBBox():
    def __init__(self,
                 _bbox_opts):
        # Save information to its layer
        self.image_width  = _bbox_opts.opts['image_width']
        self.image_height = _bbox_opts.opts['image_height']
        self.s_min        = _bbox_opts.opts['smin']
        self.s_max        = _bbox_opts.opts['smax']
        self.layer_sizes  = _bbox_opts.opts['layer_sizes']
        self.num_boxs     = _bbox_opts.opts['num_boxs']
        self.offset       = _bbox_opts.opts['offset']
        self.steps        = _bbox_opts.opts['steps']
        self.min_sizes    = _bbox_opts.opts['min_sizes']
        self.max_sizes    = _bbox_opts.opts['max_sizes']

        self.create_default_box()

    def create_default_box(self):
        self.list_default_boxes = []

        if self.steps is None:
            self.steps = []
            for _layer_size in self.layer_sizes:
                _step = round(self.image_width / _layer_size[0])
                self.steps.append(_step)

        _step = int(math.floor((self.s_max - self.s_min) / (len(self.layer_sizes) - 2)))

        if self.min_sizes == None or self.max_sizes == None:
            self.min_sizes = []
            self.max_sizes = []
            for _ratio in xrange(self.s_min, self.s_max + 1, _step):
                self.min_sizes.append(self.image_width  * _ratio / 100.)
                self.max_sizes.append(self.image_height * (_ratio + _step) / 100.)
            self.min_sizes = [self.image_width * 10 / 100.] + self.min_sizes
            self.max_sizes = [self.image_width * 20 / 100.] + self.max_sizes

        for _k, _layer_size in enumerate(self.layer_sizes):
            _layer_width  = _layer_size[0]
            _layer_height = _layer_size[1]
            _numbox       = self.num_boxs[_k]

            _min_size = self.min_sizes[_k]
            _max_size = self.max_sizes[_k]

            if _numbox == 4:
                _aspect_ratio = [1., 2., 1. / 2.]
            else:
                _aspect_ratio = [1., 2., 1. / 2., 3., 1. / 3.]
            _step_w = _step_h = self.steps[_k]
            for _h in range(_layer_height):
                for _w in range(_layer_width):
                    _center_x = (_w + self.offset) * _step_w
                    _center_y = (_h + self.offset) * _step_h

                    # first prior: aspect_ratio = 1, size = min_size
                    _box_width = _box_height = _min_size
                    # xmin
                    _x_min = (_center_x - _box_width / 2.)  / self.image_width
                    # ymin
                    _y_min = (_center_y - _box_height / 2.) / self.image_height
                    # xmax
                    _x_max = (_center_x + _box_width / 2.)  / self.image_width
                    # ymax
                    _y_max = (_center_y + _box_height / 2.) / self.image_height
                    self.list_default_boxes.append([_x_min, _y_min, _x_max, _y_max])

                    if _max_size > 0:
                        # second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
                        _box_width = _box_height = math.sqrt(_min_size * _max_size);
                        # xmin
                        _x_min = (_center_x - _box_width / 2.)  / self.image_width
                        # ymin
                        _y_min = (_center_y - _box_height / 2.) / self.image_height
                        # xmax
                        _x_max = (_center_x + _box_width / 2.)  / self.image_width
                        # ymax
                        _y_max = (_center_y + _box_height / 2.) / self.image_height
                        self.list_default_boxes.append([_x_min, _y_min, _x_max, _y_max])

                    for _ar in _aspect_ratio:
                        if _ar == 1:
                            continue

                        _box_width  = _min_size * math.sqrt(_ar)
                        _box_height = _min_size / math.sqrt(_ar)

                        # xmin
                        _x_min = (_center_x - _box_width / 2.)  / self.image_width
                        # ymin
                        _y_min = (_center_y - _box_height / 2.) / self.image_height
                        # xmax
                        _x_max = (_center_x + _box_width / 2.)  / self.image_width
                        # ymax
                        _y_max = (_center_y + _box_height / 2.) / self.image_height
                        self.list_default_boxes.append([_x_min, _y_min, _x_max, _y_max])

        self.list_default_boxes = numpy.asarray(self.list_default_boxes, dtype = 'float32')

    def bbox(self,
             _preds_batch,
             _boxes_batch):
        best_boxes_batch = []
        for _batch_id in range(_preds_batch.shape[0]):
            _best_boxes = []
            _preds = _preds_batch[_batch_id]
            _boxes = _boxes_batch[_batch_id]
            for _id, _box in enumerate(_boxes):
                _archor_box = self.list_default_boxes[_id]
                _archor_xmin = _archor_box[0]
                _archor_ymin = _archor_box[1]
                _archor_xmax = _archor_box[2]
                _archor_ymax = _archor_box[3]
                _cx = (_archor_xmin + _archor_xmax) / 2
                _cy = (_archor_ymin + _archor_ymax) / 2
                _w  = (_archor_xmax - _archor_xmin)
                _h  = (_archor_ymax - _archor_ymin)

                _offset_xmin = _box[0]
                _offset_ymin = _box[1]
                _offset_xmax = _box[2]
                _offset_ymax = _box[3]

                _cx = _offset_xmin * 0.1 * _w + _cx
                _cy = _offset_ymin * 0.1 * _h + _cy
                _w  = math.exp(_offset_xmax * 0.2) * _w
                _h  = math.exp(_offset_ymax * 0.2) * _h

                if _preds[_id] >= 1:
                    _xmin = _cx - _w / 2.
                    _ymin = _cy - _h / 2.
                    _xmax = _cx + _w / 2.
                    _ymax = _cy + _h / 2.

                    _xmin = min(max(_xmin, 0), 1)
                    _ymin = min(max(_ymin, 0), 1)
                    _xmax = min(max(_xmax, 0), 1)
                    _ymax = min(max(_ymax, 0), 1)

                    _best_boxes.append([(_xmin + _xmax) / 2,
                                        (_ymin + _ymax) / 2,
                                        (_xmax - _xmin),
                                        (_ymax - _ymin)])
                    # _best_boxes.append([_xmin, _ymin, _xmax, _ymax])
            best_boxes_batch.append(_best_boxes)

        return best_boxes_batch

    def bbou(self,
            _best_boxs_batch):
        for _best_boxs in _best_boxs_batch:
            _num_bboxs = len(_best_boxs)
            _filter_bboxs = []

            _checks = numpy.ones((_num_bboxs,))
            for _k in range(_num_bboxs):
                if (_checks[_k]):
                    for _l in range(_k + 1, _num_bboxs):
                        _bbox_1 = _best_boxs[_k]
                        _bbox_2 = _best_boxs[_l]

                        if self.iou(_bbox_1, _bbox_2) > 0.5:
                            _checks[_l] = 0
                            _checks[_k] = 2


            filter_bboxs_batch = []
            for _k in range(_num_bboxs):
                if (_checks[_k] >= 1):
                    _filter_bboxs.append(_best_boxs[_k])

            filter_bboxs_batch.append(_filter_bboxs)
        return filter_bboxs_batch

    def iou(self,
            _bbox1,
            _bbox2):
        _cx1 = _bbox1[0]
        _cy1 = _bbox1[1]
        _w1  = _bbox1[2]
        _h1  = _bbox1[3]

        _cx2 = _bbox2[0]
        _cy2 = _bbox2[1]
        _w2  = _bbox2[2]
        _h2  = _bbox2[3]

        _inter_x = max(0, min(_cx1 + _w1 / 2., _cx2 + _w2 / 2.) - max(_cx1 - _w1 / 2., _cx2 - _w2 / 2.))
        _inter_y = max(0, min(_cy1 + _h1 / 2., _cy2 + _h2 / 2.) - max(_cy1 - _h1 / 2., _cy2 - _h2 / 2.))

        _iter_area = _inter_x * _inter_y

        _area1 = _w1 * _h1
        _area2 = _w2 * _h2

        IOU = _iter_area / (_area1 + _area2 - _iter_area)
        return IOU
