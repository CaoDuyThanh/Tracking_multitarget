import theano.tensor as T
from UtilLayer import *

class NormalizeLayer():
    def __init__(self,
                 _net,
                 _input):
        # Save all information to its layer
        self.normalize_scale = _net.layer_opts['normalize_scale']
        self.filter_shape    = _net.layer_opts['normalize_filter_shape']
        self.scale_name      = _net.layer_opts['normalize_scale_name']

        self.scale = create_shared_parameter(
                        _rng      = _net.net_opts['rng'],
                        _shape    = self.filter_shape,
                        _name_var = self.scale_name
                    )

        _input2     = T.sqr(_input)
        _input_sum  = _input2.sum(axis = 1, keepdims = True)
        _input_sqrt = T.sqrt(_input_sum)

        _output_shape  = T.shape(_input)
        _scale_reshape = self.scale.reshape((1, self.filter_shape[0], 1, 1))
        _scale_reshape = T.extra_ops.repeat(
            _scale_reshape,
            _output_shape[0],
            0)
        _scale_reshape = T.extra_ops.repeat(
            _scale_reshape,
            _output_shape[2],
            2)
        _scale_reshape = T.extra_ops.repeat(
            _scale_reshape,
            _output_shape[3],
            3)

        self.output = _input / _input_sqrt * _scale_reshape