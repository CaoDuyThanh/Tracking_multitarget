import theano.tensor as T
import pickle
import cPickle
from UtilLayer import *
from BasicLayer import *

class LSTMLayer(BasicLayer):
    def __init__(self,
                 net):
        # Save all information to its layer
        self.net_name     = net.net_name
        self.num_hidden   = net.layer_opts['lstm_hidden_size']
        self.input_size   = net.layer_opts['lstm_input_size']
        self.output_size  = net.layer_opts['lstm_output_size']
        self.params       = net.layer_opts['lstm_params']

        if self.params is None:
            # Parameters for list of input
            # Init Wi | Wf | Wc | Wo
            Wi = create_shared_parameter(net.net_opts['rng'], (self.input_size, self.num_hidden), 0.08, 1, '%s_Wi' % (self.net_name))
            Wf = create_shared_parameter(net.net_opts['rng'], (self.input_size, self.num_hidden), 0.08, 1, '%s_Wf' % (self.net_name))
            Wc = create_shared_parameter(net.net_opts['rng'], (self.input_size, self.num_hidden), 0.08, 1, '%s_Wc' % (self.net_name))
            Wo = create_shared_parameter(net.net_opts['rng'], (self.input_size, self.num_hidden), 0.08, 1, '%s_Wo' % (self.net_name))

            # Init Ui | bi
            Ui = create_shared_parameter(net.net_opts['rng'], (self.num_hidden, self.num_hidden), 0.08, 1, '%s_Ui' % (self.net_name))
            bi = create_shared_parameter(net.net_opts['rng'], (self.num_hidden,), 0.08, 0, '%s_bi' % (self.net_name))

            # Init Uf | bf
            Uf = create_shared_parameter(net.net_opts['rng'], (self.num_hidden, self.num_hidden), 0.08, 1, '%s_Uf' % (self.net_name))
            bf = create_shared_parameter(net.net_opts['rng'], (self.num_hidden,), 0.08, 0, '%s_bf' % (self.net_name))

            # Init Uc | bc
            Uc = create_shared_parameter(net.net_opts['rng'], (self.num_hidden, self.num_hidden), 0.08, 1, '%s_Uc' % (self.net_name))
            bc = create_shared_parameter(net.net_opts['rng'], (self.num_hidden,), 0.08, 0, '%s_bc' % (self.net_name))

            # Init Uo | bo
            Uo = create_shared_parameter(net.net_opts['rng'], (self.num_hidden, self.num_hidden), 0.08, 1, '%s_Uo' % (self.net_name))
            bo = create_shared_parameter(net.net_opts['rng'], (self.num_hidden,), 0.08, 0, '%s_bo' % (self.net_name))

            # Parameters for list of output
            Wy = create_shared_parameter(net.net_opts['rng'], (self.num_hidden, self.output_size), 0.08, 1, '%s_Wy' % (self.net_name))
            by = create_shared_parameter(net.net_opts['rng'], (self.output_size,), 0.08, 0, '%s_by' % (self.net_name))

            self.params = [Wi, Wf, Wc, Wo] + \
                          [Wy, by] + \
                          [Ui, Uf, Uc, Uo] + \
                          [bi, bf, bc, bo]

    def step(self,
             _input_x,
             _hkm1,
             _ckm1,
             _output_y):
        # Get all weight from param
        Wi = self.params[0]
        Wf = self.params[1]
        Wc = self.params[2]
        Wo = self.params[3]
        Wy = self.params[4]
        by = self.params[5]
        Ui = self.params[6]
        Uf = self.params[7]
        Uc = self.params[8]
        Uo = self.params[9]
        bi = self.params[10]
        bf = self.params[11]
        bc = self.params[12]
        bo = self.params[13]

        _input_i = T.dot(_input_x, Wi)
        _input_f = T.dot(_input_x, Wf)
        _input_o = T.dot(_input_x, Wo)
        _input_g = T.dot(_input_x, Wc)

        # Calculate to next layer
        _i = T.nnet.sigmoid(_input_i + T.dot(_hkm1, Ui) + bi)
        _f = T.nnet.sigmoid(_input_f + T.dot(_hkm1, Uf) + bf)
        _o = T.nnet.sigmoid(_input_o + T.dot(_hkm1, Uo) + bo)
        _g = T.tanh(_input_g + T.dot(_hkm1, Uc) + bc)

        C = _ckm1 * _f + _g * _i
        H = T.tanh(C) * _o

        output_y = T.dot(H, Wy) + by
        prob_y   = T.nnet.softmax(output_y)

        return C, prob_y