import theano
import theano.tensor as T
from Layers.Net import *
from Layers.LayerHelper import *

def numpy_floatX(data):
    return numpy.asarray(data, dtype = 'float32')

class DAModel():
    def __init__(self,
                 da_input_size    = 40,
                 da_hidden_size   = 512,
                 da_output_size   = 40):
        ####################################
        #       Create model               #
        ####################################
        # ========== Create tensor variables to store input / output data ==========
        # ===== DA Input =====
        self.DA_C  = T.tensor3('DA_C')
        self.DA_A  = T.tensor3('DA_A')

        # ===== Extract Info =====
        num_truncate = self.DA_C.shape[0]
        num_object   = self.DA_C.shape[1]

        # ========== Create encoder LSTM layer ==========
        # ===== Create Data Association net =====
        self.DA_net         = LSTMNet()             # Data Association net
        self.DA_net.NetName = 'DA Net'
        self.DA_net.layer_opts['lstm_input_size']  = da_input_size
        self.DA_net.layer_opts['lstm_hidden_size'] = da_hidden_size
        self.DA_net.layer_opts['lstm_output_size'] = da_output_size

        # ===== Create LSTM layer =====
        self.DA_net.layer_opts['lstm_trun'] = LSTMLayer(self.DA_net)

        # ===== DA step =====
        _DA_output, _DA_update = theano.scan(self.DA_net.layer_opts['lstm_trun'].step,
                                           sequences    = [self.DA_C],
                                           outputs_info = [T.alloc(numpy_floatX(0.),
                                                                   num_object,
                                                                   da_hidden_size),
                                                           T.alloc(numpy_floatX(0.),
                                                                   num_object,
                                                                   da_hidden_size),
                                                           T.alloc(numpy_floatX(0.),
                                                                   num_object,
                                                                   da_output_size)],
                                           n_steps      = num_truncate)

        # ========== Calculate cost function ==========
        # ===== Confidence loss =====
        _A_prob     = _DA_output[2]
        _A_pred     = T.argmax(_A_prob, axis = 2)
        _cost_batch = T.sum(self.DA_A * -T.log(_A_prob))
        _cost_batch /= num_truncate

        # ===== Update params =====
        _params = self.DA_net.layer_opts['lstm_trun'].params
        _grads  = T.grad(_cost_batch, _params)

        self.optimizer = AdamGDUpdate(self.DA_net, params = _params, grads = _grads)
        _magnitude = 0
        for grad in self.optimizer.grads:
            _magnitude += T.sqr(grad).sum()
        _magnitude = T.sqrt(_magnitude)

        updates =    _DA_update \
                   + self.optimizer.updates

        # ========== Call functions ==========
        # ===== Train function =====
        self.train_func = theano.function(inputs  = [self.optimizer.learning_rate,
                                                     self.DA_C,
                                                     self.DA_A],
                                          updates = updates,
                                          outputs = [_cost_batch])

        # ===== Valid function =====
        self.valid_func = theano.function(inputs  = [self.DA_C,
                                                     self.DA_A],
                                          outputs = [_cost_batch])

        # ===== Pred function =====
        self.prob_func = theano.function(inputs=[self.DA_C],
                                         outputs=[_A_prob])
        self.pred_func  = theano.function(inputs  = [self.DA_C],
                                          outputs = [_A_pred])

    def save_model(self, file):
        self.DA_net.layer_opts['lstm_trun'].save_model(file)

    def save_state(self, file):
        self.DA_net.layer_opts['lstm_trun'].save_model(file)
        self.optimizer.save_model(file)

    def load_model(self, file):
        self.DA_net.layer_opts['lstm_trun'].load_model(file)

    def load_state(self, file):
        self.DA_net.layer_opts['lstm_trun'].load_model(file)
        self.optimizer.load_model(file)
