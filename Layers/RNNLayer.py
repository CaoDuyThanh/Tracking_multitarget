import theano.tensor as T
import pickle
import cPickle
from UtilLayer import *
from BasicLayer import *

class RNNLayer(BasicLayer):
    def __init__(self,
                 net):
        # Save all information to its layer
        self.net_name     = net.net_name
        self.num_hidden   = net.layer_opts['rnn_hidden_size']
        self.input_size   = net.layer_opts['rnn_input_size']
        self.output_size  = net.layer_opts['rnn_output_size']
        self.params       = net.layer_opts['rnn_params']

        if self.params is None:
            # Parameters for input
            # Init Wxt | Wht | bt
            _Wx_in = create_shared_parameter(net.net_opts['rng'], (self.input_size, self.num_hidden), 0.08, 1, '%s_Wx_in' % (self.net_name))
            _Wh_in = create_shared_parameter(net.net_opts['rng'], (self.num_hidden, self.num_hidden), 0.08, 1, '%s_Wh_in' % (self.net_name))
            _bi_in = create_shared_parameter(net.net_opts['rng'], (self.num_hidden,)                , 0,    0, '%s_bi_in' % (self.net_name))

            # Parameters for output
            _Wh_ou = create_shared_parameter(net.net_opts['rng'], (self.num_hidden, self.output_size), 0.08, 1, '%s_Wh_ou' % (self.net_name))
            _bi_ou = create_shared_parameter(net.net_opts['rng'], (self.output_size,)                , 0,    0, '%s_bi_ou' % (self.net_name))

            self.params = [_Wx_in, _Wh_in, _bi_in] + \
                          [_Wh_ou, _bi_ou]

    def step(self,
             _input_x,
             _hkm1):
        # Get all weight from param
        _Wx_in, _Wh_in, _bi_in, \
        _Wh_ou, _bi_ou = self.params

        # Update state
        H = T.tanh(T.dot(_input_x, _Wx_in) + T.dot(_hkm1, _Wh_in) + _bi_in)

        # Calculate output
        output_y = T.dot(H, _Wh_ou) + _bi_ou

        return H, output_y

    def step1(self,
             _input_x,
             _hkm1):
        _hkm1 = _hkm1.reshape((_hkm1.shape[0], 1, _hkm1.shape[1]))

        # Get all weight from param
        _Wx_in, _Wh_in, _bi_in, \
        _Wh_ou, _bi_ou = self.params

        # Update state
        H = T.tanh(T.dot(_input_x, _Wx_in) + T.dot(_hkm1, _Wh_in) + _bi_in)

        # Calculate output
        output_y = T.dot(H, _Wh_ou) + _bi_ou

        return output_y