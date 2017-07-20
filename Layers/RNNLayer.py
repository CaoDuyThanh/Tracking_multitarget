import theano.tensor as T
import pickle
import cPickle
from UtilLayer import *
from BasicLayer import *

class RNNLayer(BasicLayer):
    def __init__(self,
                 net):
        # Save all information to its layer
        self.net_name     = net.NetName
        self.num_hidden   = net.LayerOpts['rnn_num_hidden']
        self.input_size   = net.LayerOpts['rnn_inputs_size']
        self.output_size  = net.LayerOpts['rnn_outputs_size']
        self.params       = net.LayerOpts['rnn_params']

        if self.params is None:
            # Parameters for input
            # Init Wxt | Wht
            _Wx_in = create_shared_parameter(net.NetOpts['rng'], (self.input_size, self.num_hidden), 1, '%s_Wx_in' % (self.net_name))
            _Wh_in = create_shared_parameter(net.NetOpts['rng'], (self.num_hidden, self.num_hidden), 1, '%s_Wh_in' % (self.net_name))

            # Init Wx_star_t
            _WA_mi = create_shared_parameter(net.NetOpts['rng'], (self.num_hidden, self.num_hidden), 1, '%s_WA_mi' % (self.net_name))

            # Parameters for output
            _Wx_star_ou = create_shared_parameter(net.NetOpts['rng'], (self.num_hidden, self.output_size), 1, '%s_Wx_star_ou' % (self.net_name))
            _Wx_ou      = create_shared_parameter(net.NetOpts['rng'], (self.num_hidden, self.output_size), 1, '%s_Wx_ou' % (self.net_name))
            _Weps_ou    = create_shared_parameter(net.NetOpts['rng'], (self.num_hidden, self.output_size), 1, '%s_Weps_ou' % (self.net_name))

            self.params = [_Wx_in, _Wh_in] + \
                          [_WA_mi] + \
                          [_Wx_star_ou, _Wx_ou, _Weps_ou]

    def step(self,
             ht,
             xt,
             zt,
             At,
             ept):
        # Get all weight from param
        _Wx_in, _Wh_in, \
        _WA_mi, \
        _Wx_star_ou, _Wx_ou, _Weps_ou = self.Params

        # Prediction
        state      = T.dot(xt, _Wx_in) + T.dot(ht, _Wh_in)
        state      = T.tanh(state)
        x_star_ou  = T.dot(state, _Wx_star_ou)

        # Update
        x_hat   = T.concatenate(x_star_ou, zt)
        mapping = T.dot(x_hat, At)
        proj    = T.dot((mapping * ept), _WA_mi)
        proj1   = T.tanh(proj + ht)
        x_t_p_1 = T.dot(proj1, _Wx_ou)
        eps_t_p_1 = T.nnet.sigmoid(T.dot(proj1, _Weps_ou))
        eps_star_t_p_1 = T.abs_(ept - eps_t_p_1)

        return x_star_ou, state, x_t_p_1, eps_t_p_1, eps_star_t_p_1