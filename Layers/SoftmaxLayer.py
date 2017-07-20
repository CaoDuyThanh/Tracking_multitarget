import theano.tensor as T
import numpy

class SoftmaxLayer():
    def __init__(self,
                 _net,
                 _input):
        # Save information to its layer
        self.axis    = _net.layer_opts['softmax_axis']
        self.input   = _input

        _e_x = T.exp(_input - _input.max(axis = self.axis, keepdims = True))
        _out = _e_x / _e_x.sum(axis = self.axis, keepdims = True)

        self.output = _out