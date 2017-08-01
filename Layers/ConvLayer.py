import pickle
import cPickle
import theano.tensor as T
from UtilLayer import *
from theano.tensor.nnet import conv2d

class ConvLayer():
    def __init__(self,
                 net,
                 input):
        # Save information to its layer
        self.input          = input
        self.input_shape     = net.layer_opts['conv2D_input_shape']
        self.filter_shape    = net.layer_opts['conv2D_filter_shape']
        self.filter_flip     = net.layer_opts['conv2D_filter_flip']
        self.border_mode     = net.layer_opts['conv2D_border_mode']
        self.subsample       = net.layer_opts['conv2D_stride']
        self.filter_dilation = net.layer_opts['conv2D_filter_dilation']
        self.W               = net.layer_opts['conv2D_W']
        self.W_name          = net.layer_opts['conv2D_WName']
        self.b               = net.layer_opts['conv2D_b']
        self.b_name          = net.layer_opts['conv2D_bName']

        if self.W is None:
            self.W = create_shared_parameter(
                        _rng      = net.net_opts['rng'],
                        _shape    = self.filter_shape,
                        _name_var = self.W_name
                    )

        if self.b is None:
            bShape = (self.filter_shape[0],)
            self.b = create_shared_parameter(
                        _rng      = net.net_opts['rng'],
                        _shape    = bShape,
                        _name_var = self.b_name
                    )
        self.params = [self.W, self.b]

        _conv_output = conv2d(
                        input           = self.input,
                        input_shape     = self.input_shape,
                        filters         = self.W,
                        filter_shape    = self.filter_shape,
                        border_mode     = self.border_mode,
                        subsample       = self.subsample,
                        filter_flip     = self.filter_flip,
                        filter_dilation = self.filter_dilation
                    )

        _output_shape = T.shape(_conv_output)
        _rp_biases = self.b.reshape((1, self.filter_shape[0], 1, 1))
        _rp_biases = T.extra_ops.repeat(
            _rp_biases,
            _output_shape[0],
            0)
        _rp_biases = T.extra_ops.repeat(
            _rp_biases,
            _output_shape[2],
            2)
        _rp_biases = T.extra_ops.repeat(
            _rp_biases,
            _output_shape[3],
            3)
        self.output = _conv_output + _rp_biases

    def save_model(self, file):
        [pickle.dump(param.get_value(borrow = True), file, 2) for param in self.params]

    def load_model(self, file):
        [param.set_value(cPickle.load(file), borrow=True) for param in self.params]