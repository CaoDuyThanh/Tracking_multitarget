import theano.tensor as T
from UtilLayer import *

class HiddenLayer():
    def __init__(self,
                 _net,
                 _input):
        # Save information to its layer
        self.input       = _input
        self.input_size  = _net.layer_opts['hidden_input_size']
        self.output_size = _net.layer_opts['hidden_output_size']
        self.W           = _net.layer_opts['hidden_W']
        self.W_name      = _net.layer_opts['hidden_WName']
        self.b           = _net.layer_opts['hidden_b']
        self.b_name      = _net.layer_opts['hidden_bName']

        if self.W is None:
            self.W = create_shared_parameter(
                        _rng      = _net.net_opts['rng'],
                        _shape    = (self.input_size, self.output_size),
                        _name_var = self.W_name
                    )

        if self.b is None:
            self.b = create_shared_parameter(
                        _rng     = _net.net_opts['rng'],
                        _shape   = (self.output_size,),
                        _name_var= self.b_name
                    )

        self.params = [self.W, self.b]

        self.output = T.dot(_input, self.W) + self.b
