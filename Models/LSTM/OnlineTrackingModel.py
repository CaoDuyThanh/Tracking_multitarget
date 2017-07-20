import theano
import theano.tensor as T
from Layers.Net import *
from Layers.LayerHelper import *

def numpy_floatX(data):
    return numpy.asarray(data, dtype = 'float32')

class OnlineTrackingModel():
    def __init__(self,
                 pub_input_size   = 4,
                 pub_hidden_size  = 256,
                 pub_output_size  = 4,
                 da_input_size    = 4,
                 da_hidden_size   = 512,
                 da_output_size   = 4):
        ####################################
        #       Create model               #
        ####################################
        # ========== Create tensor variables to store input / output data ==========
        # ===== RNN Input =====
        self.PUB_h  = None
        self.PUB_x  = None
        self.PUB_z  = None
        self.PUB_A  = None
        self.PUB_ep = None

        # ===== LSTM Input =====
        self.DA_C  = None
        self.DA_h  = None
        self.DA_c  = None

        # ===== Extract Info =====
        num_object   = self.PUB_x.shape[0]
        num_feature  = self.PUB_x.shape[1]
        num_truncate = self.PUB_h.shape[0]
        num_batch    = self.PUB_h.shape[1]

        # ========== Create encoder LSTM layer ==========
        # ===== Create Prediction - Update - Birth net =====
        self.PUB_net         = RNNNet()
        self.PUB_net.net_name = 'PUB Net'
        self.PUB_net.layer_opts['rnn_input_size']   = pub_input_size
        self.PUB_net.layer_opts['rnn_hidden_size']  = pub_hidden_size
        self.PUB_net.layer_opts['rnn_output_size']  = pub_output_size

        # ===== Create RNN layer =====
        self.PUB_net.layer_opts['rnn_trun'] = RNNLayer(self.PUB_net)

        # ===== DA step =====
        PUB_output, PUB_update = theano.scan(self.PUB_net.layer_opts['rnn_trun'].step,
                                             sequences    = [self.Input],
                                             outputs_info = [],
                                             n_steps      = num_truncate)

        # ===== Create Data Association net =====
        self.DA_net         = LSTMNet()             # Data Association ne
        self.DA_net.NetName = 'DA Net'
        self.DA_net.layer_opts['lstm_input_size']  = da_input_size
        self.DA_net.layer_opts['lstm_hidden_size'] = da_hidden_size
        self.DA_net.layer_opts['lstm_output_size'] = da_output_size

        # ===== Create LSTM layer =====
        self.DA_net.layer_opts['lstm_trun'] = LSTMLayer(self.DA_net)

        # ===== DA step =====
        DA_output, DA_update = theano.scan(self.DA_net.layer_opts['lstm_trun'].step,
                                           sequences    = [self.DA_C],
                                           outputs_info = [T.alloc(numpy_floatX(0.),
                                                                   num_truncate,
                                                                   da_hidden_size),
                                                           T.alloc(numpy_floatX(0.),
                                                                   num_truncate,
                                                                   da_hidden_size),
                                                           T.alloc(numpy_floatX(0.),
                                                                   num_truncate,
                                                                   da_hidden_size)],
                                           n_steps      = num_truncate)

        # ========== Calculate cost function ==========
        # ===== Confidence loss =====
        _cost_batch = 0
        def cost_func(x_star, x, x_gt, eps, eps_gt, eps_star):
            hyper1 = 0
            hyper2 = 0
            hyper3 = 0
            hyper4 = 0
            cost_pred        = T.sum(T.sqr(x_star - x_gt)) * hyper1 / (num_object * num_feature)
            cost_update      = T.sum(T.sqr(x - x_gt) * hyper2 / (num_object * num_feature))
            cost_eps         = eps_gt * T.log(eps) + (1 - eps_gt) * T.log(1 - eps)
            cost_birth_death = hyper3 * cost_eps + hyper4 * eps_star
            cost             = cost_pred + cost_update + cost_birth_death
            return cost
        all_cost, cost_update = theano.scan(cost_func,
                                            sequences    = [PUB_output, DA_output],
                                            outputs_info = [T.alloc(numpy_floatX(0.),
                                                            num_truncate)],
                                            n_steps      = num_truncate)
        _cost_batch = T.sum(all_cost)

        # ===== Update params =====
        _params = self.PUB_net.layer_opts['rnn_trun'].Params \
                + self.DA_net.layer_opts['lstm_trun'].Params
        _grads  = T.grad(_cost_batch, _params)

        self.optimizer = AdamGDUpdate(self.DA_net, params = _params, grads = _grads)
        _magnitude = 0
        for grad in self.optimizer.grads:
            _magnitude += T.sqr(grad).sum()
        _magnitude = T.sqrt(_magnitude)

        updates = PUB_update \
                + DA_update \
                + cost_update \
                + self.Optimizer.Updates

        # ========== Call functions ==========
        # ===== Train function =====
        self.train_func = theano.function(inputs  = None,
                                          updates = updates,
                                          outputs = [_cost_batch])

        # ===== Valid function =====
        self.valid_func = theano.function(inputs  = [None],
                                          outputs = [_cost_batch])

        # ===== Pred function =====
        self.pred_func  = theano.function(inputs  = None,
                                          outputs = [_cost_batch])

    def save_model(self, file):
        self.PUB_net.layer_opts['rnn_trun'].save_model(file)
        self.DA_net.layer_opts['lstm_trun'].save_model(file)

    def save_state(self, file):
        self.PUB_net.layer_opts['rnn_trun'].save_model(file)
        self.DA_net.layer_opts['lstm_trun'].save_model(file)
        self.optimizer.save_model(file)

    def load_model(self, file):
        self.PUB_net.layer_opts['rnn_trun'].load_model(file)
        self.DA_net.layer_opts['lstm_trun'].load_model(file)

    def load_state(self, file):
        self.PUB_net.layer_opts['rnn_trun'].load_model(file)
        self.DA_net.layer_opts['lstm_trun'].load_model(file)
        self.optimizer.load_model(file)
